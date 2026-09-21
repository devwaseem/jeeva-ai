import asyncio
import json
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from jeeva._weights import require_local_adapter_safetensors
from jeeva.arbiters.base import BaseArbiter
from jeeva.core.cache import (
    build_branch_attention_and_positions,
    replicate_dynamic_cache,
)
from jeeva.core.candidate_scoring import (
    CandidateScoreAggregate,
    aggregate_candidate_scores,
    balanced_candidate_orders,
)
from jeeva.core.prompts import build_chat_prefill, build_contextual_metric_prompts
from jeeva.heads.base import BaseDecisionHead
from jeeva.heads.linear import LinearDecisionHead
from jeeva.heads.metric import BilinearMetricHead
from jeeva.schema.multimodal import MultimodalContext
from jeeva.schema.primitives import (
    BasePrimitive,
    Choice,
    DecisionResult,
    normalize_schema,
)
from jeeva.types import DecisionMode, DtypeType, normalize_decision_mode, resolve_dtype

# Convenience alias used throughout this module
ContextInput = str | MultimodalContext


def _subtract_score_offset(
    similarities: torch.Tensor,
    offset: Sequence[float] | None,
    *,
    field_name: str,
) -> torch.Tensor:
    if offset is None:
        return similarities
    offset_tensor = torch.as_tensor(
        offset,
        dtype=similarities.dtype,
        device=similarities.device,
    )
    if offset_tensor.shape != similarities.shape:
        raise ValueError(
            f"Calibration for field {field_name!r} has {offset_tensor.numel()} scores; "
            f"expected {similarities.numel()}."
        )
    return similarities - offset_tensor


logger = logging.getLogger("jeeva")


@dataclass
class EvaluationResult:
    """Structured output containing decisions across all schema lanes with telemetry."""

    decisions: dict[str, DecisionResult]
    telemetry: dict[str, Any]
    candidate_scores: dict[str, tuple[float, ...]] = field(default_factory=dict)

    @property
    def values(self) -> dict[str, Any]:
        """Convenience dictionary mapping field names directly to extracted values."""
        return {k: res.value for k, res in self.decisions.items()}

    @property
    def is_all_certain(self) -> bool:
        """True if all evaluated fields achieved the certainty threshold."""
        return all(res.is_certain for res in self.decisions.values())

    @property
    def is_certain(self) -> bool:
        """Convenient alias for is_all_certain."""
        return self.is_all_certain

    @property
    def is_ambiguous(self) -> bool:
        """True if any field is ambiguous."""
        return not self.is_all_certain

    @property
    def ambiguous_fields(self) -> list[str]:
        """Lists all fields flagged as ambiguous by the cascade filter."""
        return [k for k, res in self.decisions.items() if not res.is_certain]

    @property
    def margins(self) -> dict[str, float]:
        """Convenience dictionary mapping field names to their certainty margin gaps."""
        return {k: res.margin for k, res in self.decisions.items()}

    @property
    def distributions(self) -> dict[str, dict[str, float]]:
        """Convenience dictionary mapping field names to candidate option probability distributions."""
        return {k: res.distribution for k, res in self.decisions.items()}

    def __getitem__(self, key: str) -> Any:
        """Allows dictionary indexing directly into extracted decision values."""
        if key in self.decisions:
            return self.decisions[key].value
        raise KeyError(
            f"Field '{key}' not found in evaluated decisions: {list(self.decisions.keys())}"
        )

    def __getattr__(self, name: str) -> Any:
        """Allows direct attribute access to field values (e.g. result.category)."""
        if name.startswith("_") or name in ("decisions", "telemetry"):
            raise AttributeError(name)
        decisions = object.__getattribute__(self, "decisions")
        if name in decisions:
            return decisions[name].value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute or field '{name}'")

    def __contains__(self, key: str) -> bool:
        return key in self.decisions

    def __iter__(self):
        return iter(self.values.items())

    def get(self, key: str, default: Any = None) -> Any:
        """Returns the decision value for key if present, else default."""
        if key in self.decisions:
            return self.decisions[key].value
        return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": {
                k: (v.value if hasattr(v, "value") else (v.name if hasattr(v, "name") else v))
                for k, v in self.values.items()
            },
            "is_certain": self.is_certain,
            "ambiguous_fields": self.ambiguous_fields,
            "decisions": {k: res.to_dict() for k, res in self.decisions.items()},
            "telemetry": self.telemetry,
        }

    def json(self, indent: int = 2) -> str:
        """Serializes results and telemetry to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def get_default_device() -> str:
    """Detects the fastest available device with native Apple Silicon (MPS) priority."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def config_declares_multimodal(config: Any) -> bool:
    """Return whether a Transformers config explicitly declares image inputs."""
    return config is not None and any(
        getattr(config, attribute, None) is not None
        for attribute in (
            "vision_config",
            "image_token_id",
            "image_token_index",
            "vision_tower",
        )
    )


def _load_causal_or_vision_model(
    model_name_or_path: str,
    device: str,
    dtype: torch.dtype,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Any:
    """
    Loads a causal language model or multimodal vision-language model.
    Tries AutoModelForCausalLM first; if the model configuration is a VLM
    (e.g. Qwen2-VL, PaliGemma), falls back to AutoModelForImageTextToText.
    """
    loaders: list[Any] = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText

        loaders.append(AutoModelForImageTextToText)
    except ImportError:
        pass

    last_error: Exception | None = None
    for loader in loaders:
        try:
            kwargs: dict[str, Any] = {
                "revision": revision,
                "local_files_only": local_files_only,
                "use_safetensors": True,
            }
            if device == "cuda":
                kwargs["device_map"] = "cuda"

            try:
                # Attempt with hardware-accelerated SDPA attention first
                model = loader.from_pretrained(
                    pretrained_model_name_or_path=model_name_or_path,
                    dtype=dtype,
                    attn_implementation="sdpa",
                    **kwargs,
                )
            except TypeError:
                try:
                    # Fallback without attn_implementation
                    model = loader.from_pretrained(
                        pretrained_model_name_or_path=model_name_or_path,
                        dtype=dtype,
                        **kwargs,
                    )
                except TypeError:
                    model = loader.from_pretrained(
                        pretrained_model_name_or_path=model_name_or_path,
                        torch_dtype=dtype,
                        **kwargs,
                    )

            if device != "cuda":
                model = model.to(device)
            return model
        except (ValueError, KeyError) as e:
            last_error = e
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not load model '{model_name_or_path}'.")


class DecisionEngine:
    """
    Ultra-low latency, zero-token structured decision engine using
    Parallel Context Slicing (PCD) over shared DynamicCache.
    """

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        dtype: torch.dtype | str | DtypeType | None = None,
        torch_dtype: torch.dtype | str | DtypeType | None = None,
        default_cascade_threshold: float = 0.30,
        temperature: float | None = None,
        mode: DecisionMode | str | None = None,
        head: BaseDecisionHead | dict[str, BaseDecisionHead] | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        contiguous_cache: bool = True,
        candidate_position_balancing: bool = True,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.device = device or get_default_device()
        self.revision = revision
        self.tokenizer_revision = revision if tokenizer_revision is None else tokenizer_revision
        self.local_files_only = local_files_only
        self.default_cascade_threshold = default_cascade_threshold
        self._global_mode_explicit = mode is not None
        self.mode = normalize_decision_mode(DecisionMode.LETTER if mode is None else mode)
        self.head = head
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.contiguous_cache = contiguous_cache
        self.candidate_position_balancing = candidate_position_balancing
        self._branch_suffix = "\n\nAssistant:\n"
        self._kwargs = kwargs
        self._lock = threading.Lock()

        if temperature is not None:
            self.temperature = temperature
        else:
            self.temperature = 1.0 if self.mode == "letter" else 0.03

        self._letter_weights: dict[str, torch.Tensor] = {}

        # Resolve dtype from either 'dtype' or 'torch_dtype' parameter
        effective_dtype_spec = dtype if dtype is not None else torch_dtype
        self.torch_dtype = resolve_dtype(effective_dtype_spec, device=self.device)
        self.dtype = self.torch_dtype
        self.model: Any
        self.tokenizer: Any

        from pathlib import Path

        is_peft_adapter = (
            Path(model_id).is_dir() and (Path(model_id) / "adapter_config.json").exists()
        )

        if is_peft_adapter:
            require_local_adapter_safetensors(model_id)
            try:
                from peft import PeftConfig, PeftModel
            except ImportError as exc:
                raise ImportError(
                    "Loading a PEFT adapter requires the training extra. "
                    "Install it with: pip install 'jeeva-ai[training]'"
                ) from exc
            peft_cfg = PeftConfig.from_pretrained(
                model_id,
                local_files_only=self.local_files_only,
            )
            base_id = peft_cfg.base_model_name_or_path
            base_revision = peft_cfg.revision or self.revision
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=self.tokenizer_revision,
                local_files_only=self.local_files_only,
            )
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model: Any = _load_causal_or_vision_model(
                model_name_or_path=base_id or model_id,
                device=self.device,
                dtype=self.torch_dtype,
                revision=base_revision,
                local_files_only=self.local_files_only,
            )
            self.model = PeftModel.from_pretrained(
                base_model,
                model_id,
                local_files_only=self.local_files_only,
            ).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=self.tokenizer_revision,
                local_files_only=self.local_files_only,
            )
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = _load_causal_or_vision_model(
                model_name_or_path=model_id,
                device=self.device,
                dtype=self.torch_dtype,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
        self.model.eval()

        self.processor: Any = None
        self.is_multimodal: bool = False
        model_config = getattr(self.model, "config", None)
        if config_declares_multimodal(model_config):
            try:
                from transformers import AutoProcessor
            except ImportError as exc:
                raise ImportError(
                    "This model declares image inputs, but its processor dependencies "
                    "are unavailable. Install: pip install 'jeeva-ai[multimodal]'"
                ) from exc
            proc_kwargs: dict[str, Any] = {}
            if self.min_pixels is not None:
                proc_kwargs["min_pixels"] = self.min_pixels
            if self.max_pixels is not None:
                proc_kwargs["max_pixels"] = self.max_pixels
            loaded_processor = AutoProcessor.from_pretrained(
                model_id,
                revision=self.tokenizer_revision,
                local_files_only=self.local_files_only,
                **proc_kwargs,
            )
            if not hasattr(loaded_processor, "image_processor"):
                raise ValueError(
                    f"Model {model_id!r} declares image inputs, but its processor "
                    "does not expose an image_processor."
                )
            self.processor = loaded_processor
            self.is_multimodal = True
            self.tokenizer = loaded_processor.tokenizer
            logger.debug(
                "Multimodal processor loaded for '%s'; vision inputs are enabled.",
                model_id,
            )

        model_config = getattr(self.model, "config", None)
        model_name = getattr(model_config, "_name_or_path", None)
        model_commit = getattr(model_config, "_commit_hash", None)
        tokenizer_name = getattr(self.tokenizer, "name_or_path", None)
        tokenizer_kwargs = getattr(self.tokenizer, "init_kwargs", None)
        tokenizer_commit = (
            tokenizer_kwargs.get("_commit_hash") if isinstance(tokenizer_kwargs, dict) else None
        )
        self.resolved_model_id = model_name if isinstance(model_name, str) else model_id
        self.resolved_model_revision = (
            model_commit if isinstance(model_commit, str) else self.revision
        )
        self.resolved_tokenizer_id = tokenizer_name if isinstance(tokenizer_name, str) else model_id
        self.resolved_tokenizer_revision = (
            tokenizer_commit if isinstance(tokenizer_commit, str) else self.tokenizer_revision
        )

        if self.mode == "letter":
            self._precache_letter_weights()

    @classmethod
    def from_components(
        cls,
        *,
        model: Any,
        tokenizer: Any,
        model_id: str,
        device: str,
        dtype: torch.dtype,
        processor: Any = None,
        default_cascade_threshold: float = 0.30,
        temperature: float | None = None,
        mode: DecisionMode | str | None = None,
        head: BaseDecisionHead | dict[str, BaseDecisionHead] | None = None,
        contiguous_cache: bool = True,
        candidate_position_balancing: bool = True,
    ) -> "DecisionEngine":
        """Build an engine around already-loaded components without loading a checkpoint."""
        instance = cls.__new__(cls)
        instance.model_id = model_id
        instance.device = device
        instance.default_cascade_threshold = default_cascade_threshold
        instance._global_mode_explicit = mode is not None
        instance.mode = normalize_decision_mode(DecisionMode.LETTER if mode is None else mode)
        instance.head = head
        instance.min_pixels = None
        instance.max_pixels = None
        instance.contiguous_cache = contiguous_cache
        instance.candidate_position_balancing = candidate_position_balancing
        instance._kwargs = {}
        instance._lock = threading.Lock()
        instance.temperature = (
            temperature if temperature is not None else (1.0 if instance.mode == "letter" else 0.03)
        )
        instance._letter_weights = {}
        instance._branch_suffix = "\n\nAssistant:\n"
        instance.torch_dtype = dtype
        instance.dtype = dtype
        instance.model = model
        instance.processor = processor
        instance.is_multimodal = processor is not None and hasattr(processor, "image_processor")
        instance.tokenizer = getattr(processor, "tokenizer", tokenizer)
        instance.tokenizer.padding_side = "left"
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
        if instance.mode == "letter":
            instance._precache_letter_weights()
        return instance

    def _resolve_readout_modes(
        self,
        primitives: Sequence[BasePrimitive],
        mode: DecisionMode | str | None,
    ) -> list[str]:
        global_mode = normalize_decision_mode(mode) if mode is not None else self.mode
        return [global_mode] * len(primitives)

    @property
    def backbone(self) -> Any:
        """
        Returns the core transformer backbone (without lm_head projection).
        Bypasses full-vocabulary logits calculation to minimize FLOPs, memory, and latency.
        """
        if self.is_multimodal:
            return self.model

        # PEFT models: PeftModel -> base_model (LoraModel) -> model (CausalLM) -> model (Backbone)
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model"):
            inner = self.model.base_model.model
            if hasattr(inner, "model"):
                return inner.model
            if hasattr(inner, "get_decoder"):
                return inner.get_decoder()

        # Standard CausalLM models (Llama, Qwen, Mistral, Gemma, Phi, etc.)
        if hasattr(self.model, "model"):
            return self.model.model
        if hasattr(self.model, "get_decoder"):
            return self.model.get_decoder()

        return self.model

    def _get_or_compute_option_embeddings(self, primitive: BasePrimitive) -> torch.Tensor:
        """Compute dense embeddings without retaining caller-varying options."""
        candidate_texts = primitive.options
        opt_inputs = self.tokenizer(candidate_texts, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad():
            opt_out = self.model(**opt_inputs, output_hidden_states=True)
            mask = opt_inputs.attention_mask.unsqueeze(-1)
            emb = (opt_out.hidden_states[-1] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def _get_letter_weight(self, letter: str) -> torch.Tensor:
        """Return the output projection for a letter after the prompt's trailing space."""
        if letter not in self._letter_weights:
            token_ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"Letter candidate {letter!r} is not a single token for model "
                    f"{self.model_id!r}. Use mode='dense' instead."
                )
            token_id = token_ids[0]

            lm_head = getattr(self.model, "lm_head", None)
            if lm_head is None and hasattr(self.model, "get_output_embeddings"):
                lm_head = self.model.get_output_embeddings()
            if lm_head is None and hasattr(self.model, "base_model"):
                lm_head = getattr(self.model.base_model, "lm_head", None)
                if lm_head is None and hasattr(self.model.base_model, "get_output_embeddings"):
                    lm_head = self.model.base_model.get_output_embeddings()
                if lm_head is None and hasattr(self.model.base_model, "model"):
                    lm_head = getattr(self.model.base_model.model, "lm_head", None)
                    if lm_head is None and hasattr(
                        self.model.base_model.model, "get_output_embeddings"
                    ):
                        lm_head = self.model.base_model.model.get_output_embeddings()
            if lm_head is not None:
                self._letter_weights[letter] = (
                    lm_head.weight[token_id].detach().to(torch.float32).to(self.device).contiguous()
                )
            else:
                raise AttributeError(
                    "Model does not expose an lm_head or get_output_embeddings() attribute for letter readout."
                )
        return self._letter_weights[letter]

    def _get_candidate_weights(self, cand_letters: list[str]) -> torch.Tensor:
        """Return stacked projection vectors from the bounded A-Z letter cache."""
        return torch.stack(
            [self._get_letter_weight(letter) for letter in cand_letters]
        ).contiguous()

    def _precache_letter_weights(self) -> None:
        """
        Pre-computes and caches projection weight vectors for candidate letters (A-Z) on device.
        """
        import string

        for char in string.ascii_uppercase:
            try:
                self._get_letter_weight(char)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _extract_context_text(self, context: ContextInput) -> str:
        """Returns the raw text portion of a context (str or MultimodalContext)."""
        if isinstance(context, MultimodalContext):
            return context.text
        return context

    def _build_prefill_inputs(
        self,
        context: ContextInput,
        loaded_images: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Build Pass 1 inputs and retain the model-specific branch suffix."""
        if isinstance(context, MultimodalContext) and context.has_images():
            if not self.is_multimodal:
                raise ValueError(
                    "MultimodalContext with images requires a vision-language model "
                    "(e.g. 'Qwen/Qwen2-VL-2B-Instruct'), but this engine was loaded "
                    f"with '{self.model_id}' which does not expose an image_processor.\n\n"
                    "Recommended multimodal models:\n"
                    "  - 'Qwen/Qwen2-VL-2B-Instruct'   (fast, 2B)\n"
                    "  - 'Qwen/Qwen2-VL-7B-Instruct'   (higher capacity, 7B)\n"
                    "  - 'google/paligemma2-3b-pt-224'  (PaliGemma 3B)\n"
                )

            prompt_text, self._branch_suffix = build_chat_prefill(
                self.processor,
                self._build_multimodal_message(context),
            )
            pil_images = loaded_images if loaded_images is not None else context.load_pil_images()
            inputs: dict[str, Any] = self.processor(
                text=prompt_text,
                images=pil_images if pil_images else None,
                return_tensors="pt",
            )
            return {k: v.to(self.device) for k, v in inputs.items()}

        text = self._extract_context_text(context)
        prefill_prompt, self._branch_suffix = build_chat_prefill(
            self.tokenizer,
            f"Document:\n{text}",
        )
        return dict(self.tokenizer(prefill_prompt, return_tensors="pt").to(self.device))

    def _build_multimodal_message(self, context: MultimodalContext) -> list[dict[str, Any]]:
        """
        Builds the ``content`` field for the user turn of a VLM chat message,
        interleaving image placeholders (as expected by the processor's chat
        template) and the document text.

        Returns a list of content parts compatible with HuggingFace VLM chat
        templates (Qwen2-VL, LLaVA, etc.).
        """
        parts: list[dict[str, Any]] = []
        # Prepend one image token per image (standard VLM convention)
        for _ in context.images:
            parts.append({"type": "image"})
        # Append the text document
        parts.append({"type": "text", "text": f"Document:\n{context.text}"})
        return parts

    def warmup(self) -> None:
        """Pre-run a minimal inference to initialize hardware execution graphs."""
        from jeeva.schema.primitives import Noul

        self._precache_letter_weights()
        dummy_schema = [
            Noul(
                "Determine whether the system is ready.",
                {
                    True: "The system is ready.",
                    False: "The system is not ready.",
                },
            ).bind("_warmup")
        ]
        if self.is_multimodal:
            try:
                from PIL import Image

                dummy_img = Image.new("RGB", (64, 64), color="gray")
                context = MultimodalContext(text="System warmup.", images=[dummy_img])
                self.evaluate(context=context, schema=dummy_schema)
                return
            except Exception:
                pass
        self.evaluate(context="System warmup.", schema=dummy_schema)

    def evaluate(
        self,
        context: ContextInput,
        schema: Any,
        cascade_threshold: float | None = None,
        arbiter: BaseArbiter | None = None,
        mode: DecisionMode | str | None = None,
        head: BaseDecisionHead | dict[str, BaseDecisionHead] | None = None,
        _score_offsets: Mapping[str, Sequence[float]] | None = None,
        _allow_empty_context: bool = False,
    ) -> EvaluationResult:
        """
        Executes Gatekeeper Parallel Context Slicing (PCD) structured decision evaluation.

        Parameters
        ----------
        context:
            The document context to evaluate against.  Accepts either:

            * ``str`` — plain text (original behaviour, unchanged).
            * :class:`~jeeva.schema.multimodal.MultimodalContext` — text + images
              for vision-language models (e.g. ``Qwen/Qwen2-VL-2B-Instruct``).
        """
        if isinstance(context, MultimodalContext):
            if not context.text.strip():
                raise ValueError("MultimodalContext.text must be a non-empty string.")
        elif not isinstance(context, str):
            raise ValueError("Context must be a string or MultimodalContext.")
        elif not context.strip() and not _allow_empty_context:
            raise ValueError("Context must be a non-empty string.")

        primitives = normalize_schema(schema)
        num_lanes = len(primitives)
        if num_lanes == 0:
            raise ValueError("Schema must contain at least one decision field.")

        threshold = (
            cascade_threshold if cascade_threshold is not None else self.default_cascade_threshold
        )
        lane_modes = self._resolve_readout_modes(primitives, mode)
        lane_temperatures = [
            self.temperature
            if lane_mode == self.mode
            else (1.0 if lane_mode == DecisionMode.LETTER.value else 0.03)
            for lane_mode in lane_modes
        ]
        loaded_images = (
            context.load_pil_images()
            if self.is_multimodal
            and isinstance(context, MultimodalContext)
            and context.has_images()
            else None
        )

        # Resolve active learned heads for each lane.
        lane_heads: list[BaseDecisionHead | None] = []
        for primitive in primitives:
            key = primitive.key or "field"
            active_head = head.get(key) if isinstance(head, dict) else head
            if active_head is None:
                active_head = self.head.get(key) if isinstance(self.head, dict) else self.head
            if isinstance(active_head, dict):
                raise TypeError(f"Field '{key}' must resolve to one decision head.")
            if active_head is not None and not isinstance(active_head, BaseDecisionHead):
                raise TypeError(
                    f"Field '{key}' resolved to unsupported head type {type(active_head).__name__}."
                )
            if active_head is not None:
                active_head = active_head.to(self.device)
            lane_heads.append(active_head)

        with self._lock, torch.inference_mode():
            t0 = time.perf_counter()

            # -------------------------------------------------------------
            # PASS 1: Base Prefill (Base context executed ONCE into KV cache)
            # -------------------------------------------------------------
            base_inputs = self._build_prefill_inputs(context, loaded_images=loaded_images)

            base_out = self.backbone(**base_inputs, use_cache=True)

            base_cache: DynamicCache = base_out.past_key_values
            past_len = base_cache.get_seq_length()
            t_prefill = time.perf_counter()

            # -------------------------------------------------------------
            # PASS 2: Position-Balanced Branch Expansion (Gatekeeper)
            # -------------------------------------------------------------
            branch_prompts: list[str] = []
            branch_orders: list[tuple[int, ...] | None] = []
            branch_cand_letters: list[list[str]] = []
            lane_branch_indices: list[list[int]] = [[] for _ in primitives]

            def append_branch(
                lane_index: int,
                prompt: str,
                *,
                order: tuple[int, ...] | None = None,
                candidate_letters: list[str] | None = None,
                assistant_continuation: bool = False,
                assistant_prefix: str = "",
            ) -> None:
                lane_branch_indices[lane_index].append(len(branch_prompts))
                branch_prompts.append(
                    f"{self._branch_suffix}{prompt}"
                    if assistant_continuation
                    else f"{prompt}{self._branch_suffix}{assistant_prefix}"
                )
                branch_orders.append(order)
                branch_cand_letters.append(candidate_letters or [])

            for idx, prim in enumerate(primitives):
                lane_mode = lane_modes[idx]
                active_head = lane_heads[idx]
                if (
                    isinstance(active_head, BilinearMetricHead)
                    and active_head.input_strategy == "contextual"
                ):
                    query_prompt, candidate_prompts = build_contextual_metric_prompts(
                        prim.prompt,
                        prim.options,
                    )
                    append_branch(idx, query_prompt, assistant_continuation=True)
                    for candidate_prompt in candidate_prompts:
                        append_branch(idx, candidate_prompt, assistant_continuation=True)
                elif active_head is not None:
                    append_branch(
                        idx,
                        prim.build_branch_prompt(),
                        assistant_continuation=True,
                    )
                elif lane_mode == DecisionMode.LETTER.value:
                    candidate_orders = (
                        balanced_candidate_orders(len(prim.options))
                        if self.candidate_position_balancing
                        else (tuple(range(len(prim.options))),)
                    )
                    for order in candidate_orders:
                        prompt, _, candidate_letters = prim.build_letter_branch_prompt(order)
                        append_branch(
                            idx,
                            prompt,
                            order=order,
                            candidate_letters=candidate_letters,
                            assistant_prefix="Answer: ",
                        )
                else:
                    append_branch(
                        idx,
                        prim.build_branch_prompt(),
                        assistant_continuation=True,
                    )

            num_branches = len(branch_prompts)
            expanded_cache = replicate_dynamic_cache(
                base_cache,
                num_lanes=num_branches,
                contiguous=self.contiguous_cache,
            )

            branch_inputs = self.tokenizer(branch_prompts, return_tensors="pt", padding=True).to(
                self.device
            )
            branch_input_ids = branch_inputs.input_ids
            branch_attention_mask = branch_inputs.attention_mask

            comb_mask, pos_ids = build_branch_attention_and_positions(
                past_len=past_len,
                branch_attention_mask=branch_attention_mask,
                device=torch.device(self.device),
            )

            branch_kwargs: dict[str, Any] = {
                "input_ids": branch_input_ids,
                "attention_mask": comb_mask,
                "position_ids": pos_ids,
                "past_key_values": expanded_cache,
                "use_cache": False,
            }
            if self.backbone is self.model:
                branch_kwargs["output_hidden_states"] = True

            branch_out = self.backbone(**branch_kwargs)

            t_dispatch = time.perf_counter()

            # -------------------------------------------------------------
            # PASS 3: Gatekeeper Decision Harvesting
            # -------------------------------------------------------------
            decisions: dict[str, DecisionResult] = {}
            candidate_scores: dict[str, tuple[float, ...]] = {}
            order_diagnostics: dict[str, dict[str, float | int | str]] = {}
            if hasattr(branch_out, "last_hidden_state"):
                branch_hidden_seq = branch_out.last_hidden_state
            else:
                branch_hidden_seq = branch_out.hidden_states[-1]

            branch_hiddens = branch_hidden_seq[:, -1, :]
            norm_branch_hiddens = torch.nn.functional.normalize(
                branch_hiddens,
                p=2,
                dim=-1,
            )

            for idx, prim in enumerate(primitives):
                active_head = lane_heads[idx]
                lane_mode = lane_modes[idx]
                branch_indices = lane_branch_indices[idx]
                branch_index = branch_indices[0]
                score_aggregate: CandidateScoreAggregate | None = None
                field_name = prim.key or prim.kind
                score_offset = (
                    _score_offsets.get(field_name) if _score_offsets is not None else None
                )

                if active_head is not None:
                    branch_h = branch_hidden_seq[branch_index, -1, :]
                    if isinstance(active_head, LinearDecisionHead):
                        sims = active_head(branch_h)
                    elif isinstance(active_head, BilinearMetricHead):
                        if active_head.input_strategy == "contextual":
                            expected_branches = len(prim.options) + 1
                            if len(branch_indices) != expected_branches:
                                raise RuntimeError(
                                    f"Contextual field {field_name!r} requires "
                                    f"{expected_branches} branches; received {len(branch_indices)}."
                                )
                            option_hiddens = branch_hidden_seq[
                                branch_indices[1:],
                                -1,
                                :,
                            ]
                            sims = active_head(branch_h, option_hiddens)
                        else:
                            opt_embeddings = self._get_or_compute_option_embeddings(prim)
                            sims = active_head(branch_h, opt_embeddings)
                    else:
                        sims = active_head(branch_h)
                    parse_temperature = 1.0
                elif lane_mode == DecisionMode.LETTER.value:
                    positional_scores: list[torch.Tensor] = []
                    orders: list[tuple[int, ...]] = []
                    for letter_branch_index in branch_indices:
                        candidate_letters = branch_cand_letters[letter_branch_index]
                        candidate_weights = self._get_candidate_weights(candidate_letters)
                        branch_h = branch_hidden_seq[
                            letter_branch_index,
                            -1,
                            :,
                        ].to(torch.float32)
                        positional_scores.append(torch.matmul(branch_h, candidate_weights.T))
                        order = branch_orders[letter_branch_index]
                        if order is None:
                            raise RuntimeError("Letter branch is missing its candidate order.")
                        orders.append(order)

                    stacked_scores = torch.stack(positional_scores)
                    parse_temperature = lane_temperatures[idx]
                    raw_aggregate = aggregate_candidate_scores(
                        stacked_scores,
                        orders,
                        temperature=parse_temperature,
                    )
                    sims = raw_aggregate.scores
                    if self.candidate_position_balancing:
                        if score_offset is None:
                            score_aggregate = raw_aggregate
                        else:
                            _subtract_score_offset(
                                sims,
                                score_offset,
                                field_name=field_name,
                            )
                            score_aggregate = aggregate_candidate_scores(
                                stacked_scores,
                                orders,
                                temperature=parse_temperature,
                                offset=score_offset,
                            )
                        calibrated_sims = score_aggregate.scores
                        order_diagnostics[field_name] = {
                            "strategy": "cyclic_position_balance",
                            "trials": score_aggregate.trials,
                            "sensitivity": round(score_aggregate.order_sensitivity, 6),
                            "robust_margin": round(score_aggregate.robust_margin, 6),
                        }
                else:
                    opt_embeddings = self._get_or_compute_option_embeddings(prim)
                    branch_vec = norm_branch_hiddens[branch_index]
                    sims = torch.matmul(branch_vec, opt_embeddings.T)
                    parse_temperature = lane_temperatures[idx]

                candidate_scores[field_name] = tuple(
                    sims.detach().to("cpu", torch.float32).tolist()
                )
                if score_aggregate is None:
                    calibrated_sims = _subtract_score_offset(
                        sims,
                        score_offset,
                        field_name=field_name,
                    )
                field_result = prim.parse_similarities(
                    similarities=calibrated_sims,
                    cascade_threshold=threshold,
                    temperature=parse_temperature,
                )
                if score_aggregate is not None:
                    field_result.certainty_score = score_aggregate.robust_margin
                    field_result.is_certain = bool(
                        score_aggregate.robust_margin >= field_result.tau
                    )
                decisions[field_result.field] = field_result

            t_harvest = time.perf_counter()

            # -------------------------------------------------------------
            # PASS 4: Optional Arbiter Resolution for Ambiguous Lanes
            # -------------------------------------------------------------
            arbiter_invoked: list[str] = []
            t_arbiter_start = time.perf_counter()

            if arbiter is not None:
                fields_to_resolve: dict[str, BasePrimitive] = {}
                for primitive in primitives:
                    field_key = primitive.key or primitive.kind
                    current = decisions.get(field_key)
                    if current is not None and (
                        arbiter.resolve_all_fields or not current.is_certain
                    ):
                        fields_to_resolve[field_key] = primitive

                if fields_to_resolve:
                    context_text = self._extract_context_text(context)
                    resolved_decisions = arbiter.resolve(
                        context=context_text,
                        ambiguous_fields=fields_to_resolve,
                        current_decisions=decisions,
                        engine=self,
                        base_cache=base_cache,
                        past_len=past_len,
                        threshold=threshold,
                    )
                    decisions.update(resolved_decisions)
                    arbiter_invoked = list(resolved_decisions.keys())

            t_end = time.perf_counter()
            arbiter_ms = round((t_end - t_arbiter_start) * 1000, 2) if arbiter_invoked else 0.0

            telemetry = {
                "prefill_ms": round((t_prefill - t0) * 1000, 2),
                "dispatch_ms": round((t_dispatch - t_prefill) * 1000, 2),
                "harvest_ms": round((t_harvest - t_dispatch) * 1000, 2),
                "arbiter_ms": arbiter_ms,
                "arbiter_invoked_fields": arbiter_invoked,
                "arbiter_name": arbiter.name if arbiter_invoked and arbiter else None,
                "total_ms": round((t_end - t0) * 1000, 2),
                "num_lanes": num_lanes,
                "num_branches": num_branches,
                "candidate_ordering": order_diagnostics,
                "candidate_position_balancing": self.candidate_position_balancing,
                "readout_modes": {
                    primitive.key or f"field_{idx}": (
                        "head" if lane_heads[idx] is not None else lane_modes[idx]
                    )
                    for idx, primitive in enumerate(primitives)
                },
                "device": str(self.device),
                "model_id": self.model_id,
                "context_free_calibrated": bool(_score_offsets),
            }

            return EvaluationResult(
                decisions=decisions,
                telemetry=telemetry,
                candidate_scores=candidate_scores,
            )

    def decide(
        self,
        context: ContextInput,
        schema: Any,
        cascade_threshold: float | None = None,
        arbiter: BaseArbiter | None = None,
        mode: DecisionMode | str | None = None,
        head: BaseDecisionHead | dict[str, BaseDecisionHead] | None = None,
    ) -> EvaluationResult:
        """Evaluate a DecisionSchema and return its field evidence."""
        return self.evaluate(
            context=context,
            schema=schema,
            cascade_threshold=cascade_threshold,
            arbiter=arbiter,
            mode=mode,
            head=head,
        )

    async def aevaluate(
        self,
        context: ContextInput,
        schema: Any,
        cascade_threshold: float | None = None,
        arbiter: BaseArbiter | None = None,
        mode: DecisionMode | str | None = None,
        head: BaseDecisionHead | dict[str, BaseDecisionHead] | None = None,
        _score_offsets: Mapping[str, Sequence[float]] | None = None,
        _allow_empty_context: bool = False,
    ) -> EvaluationResult:
        """Non-blocking async evaluation offloaded to thread pool."""
        return await asyncio.to_thread(
            self.evaluate,
            context=context,
            schema=schema,
            cascade_threshold=cascade_threshold,
            arbiter=arbiter,
            mode=mode,
            head=head,
            _score_offsets=_score_offsets,
            _allow_empty_context=_allow_empty_context,
        )

    async def adecide(
        self,
        context: ContextInput,
        schema: Any,
        cascade_threshold: float | None = None,
        arbiter: BaseArbiter | None = None,
        mode: DecisionMode | str | None = None,
        head: BaseDecisionHead | dict[str, BaseDecisionHead] | None = None,
    ) -> Any:
        """Non-blocking async decide offloaded to thread pool."""
        return await asyncio.to_thread(
            self.decide,
            context=context,
            schema=schema,
            cascade_threshold=cascade_threshold,
            arbiter=arbiter,
            mode=mode,
            head=head,
        )

    def check(
        self,
        context: ContextInput,
        condition: str,
        cascade_threshold: float | None = None,
        arbiter: BaseArbiter | None = None,
        **kwargs: Any,
    ) -> bool:
        """Evaluate one natural-language criterion against a context."""
        from jeeva.schema.primitives import Noul

        result = self.evaluate(
            context=context,
            schema=[
                Noul(
                    condition,
                    {
                        True: "The condition is satisfied.",
                        False: "The condition is not satisfied.",
                    },
                ).bind("_check")
            ],
            cascade_threshold=cascade_threshold,
            arbiter=arbiter,
            **kwargs,
        )
        return bool(result["_check"])

    def choose(
        self,
        context: ContextInput,
        instructions: str,
        criteria: Mapping[str, str],
        cascade_threshold: float | None = None,
        arbiter: BaseArbiter | None = None,
        **kwargs: Any,
    ) -> str:
        """Choose one stable value using its explicit semantic criteria."""
        result = self.evaluate(
            context=context,
            schema=[
                Choice(
                    instructions=instructions,
                    criteria=criteria,
                ).bind("_choice")
            ],
            cascade_threshold=cascade_threshold,
            arbiter=arbiter,
            **kwargs,
        )
        return str(result["_choice"])

    async def acheck(
        self,
        context: ContextInput,
        condition: str,
        cascade_threshold: float | None = None,
        arbiter: BaseArbiter | None = None,
        **kwargs: Any,
    ) -> bool:
        """Non-blocking async boolean condition evaluator offloaded to thread pool."""
        return await asyncio.to_thread(
            self.check,
            context=context,
            condition=condition,
            cascade_threshold=cascade_threshold,
            arbiter=arbiter,
            **kwargs,
        )

    async def achoose(
        self,
        context: ContextInput,
        instructions: str,
        criteria: Mapping[str, str],
        cascade_threshold: float | None = None,
        arbiter: BaseArbiter | None = None,
        **kwargs: Any,
    ) -> str:
        """Run ``choose`` without blocking the caller's event loop."""
        return await asyncio.to_thread(
            self.choose,
            context=context,
            instructions=instructions,
            criteria=criteria,
            cascade_threshold=cascade_threshold,
            arbiter=arbiter,
            **kwargs,
        )

    def evaluate_batch(
        self,
        contexts: Sequence[ContextInput],
        schema: Any,
        cascade_threshold: float | None = None,
        arbiter: BaseArbiter | None = None,
        mode: DecisionMode | str | None = None,
        head: BaseDecisionHead | dict[str, BaseDecisionHead] | None = None,
        _score_offsets: Mapping[str, Sequence[float]] | None = None,
    ) -> list[EvaluationResult]:
        """Evaluate contexts serially while reusing the loaded engine."""
        if not contexts:
            return []

        t_batch_start = time.perf_counter()
        results: list[EvaluationResult] = []
        ambiguous_count = 0
        arbiter_resolutions = 0

        for ctx in contexts:
            res = self.evaluate(
                context=ctx,
                schema=schema,
                cascade_threshold=cascade_threshold,
                arbiter=arbiter,
                mode=mode,
                head=head,
                _score_offsets=_score_offsets,
            )
            if res.is_ambiguous:
                ambiguous_count += 1
            if res.telemetry.get("arbiter_invoked_fields"):
                arbiter_resolutions += 1
            results.append(res)

        total_batch_ms = round((time.perf_counter() - t_batch_start) * 1000, 2)
        avg_ms = round(total_batch_ms / max(1, len(contexts)), 2)
        docs_per_sec = round(
            (len(contexts) / (total_batch_ms / 1000)) if total_batch_ms > 0 else 0.0, 1
        )

        for res in results:
            res.telemetry["batch_total_ms"] = total_batch_ms
            res.telemetry["batch_avg_ms"] = avg_ms
            res.telemetry["batch_throughput_docs_sec"] = docs_per_sec
            res.telemetry["batch_size"] = len(contexts)
            res.telemetry["batch_ambiguous_items"] = ambiguous_count
            res.telemetry["batch_arbiter_resolutions"] = arbiter_resolutions

        return results

    async def aevaluate_batch(
        self,
        contexts: Sequence[ContextInput],
        schema: Any,
        cascade_threshold: float | None = None,
        arbiter: BaseArbiter | None = None,
        mode: DecisionMode | str | None = None,
        head: BaseDecisionHead | dict[str, BaseDecisionHead] | None = None,
        _score_offsets: Mapping[str, Sequence[float]] | None = None,
    ) -> list[EvaluationResult]:
        """Run a batch without blocking the caller's asyncio event loop."""
        return await asyncio.to_thread(
            self.evaluate_batch,
            contexts=contexts,
            schema=schema,
            cascade_threshold=cascade_threshold,
            arbiter=arbiter,
            mode=mode,
            head=head,
            _score_offsets=_score_offsets,
        )
