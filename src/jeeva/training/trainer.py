import random
import secrets
import time
from dataclasses import dataclass, field
from importlib import metadata
from math import ceil
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)
from transformers.cache_utils import DynamicCache

from jeeva._terminal import terminal_text
from jeeva._weights import require_local_adapter_safetensors
from jeeva.core.cache import (
    build_branch_attention_and_positions,
    replicate_dynamic_cache,
)
from jeeva.core.candidate_scoring import (
    aggregate_candidate_scores,
    balanced_candidate_orders,
)
from jeeva.core.engine import (
    DecisionEngine,
    _load_causal_or_vision_model,
    config_declares_multimodal,
    get_default_device,
)
from jeeva.core.prompts import build_chat_prefill, build_contextual_metric_prompts
from jeeva.data.dataset import JeevaDataset
from jeeva.eval.evaluator import Evaluator
from jeeva.heads.base import BaseDecisionHead
from jeeva.heads.metric import DECODER_HEAD_DIR, BilinearMetricHead
from jeeva.schema.multimodal import MultimodalContext
from jeeva.schema.primitives import BasePrimitive, Score, normalize_schema
from jeeva.training.loss import (  # type: ignore[attr-defined]
    CompositeDecisionLoss,
    backward_finite_loss,
    clip_finite_gradients,
    require_finite_value,
)
from jeeva.types import resolve_dtype

_UNSET = object()


def _epoch_sample_order(sample_count: int, *, seed: int, epoch: int) -> list[int]:
    order = list(range(sample_count))
    random.Random(seed + epoch - 1).shuffle(order)
    return order


def _require_qlora_dependencies() -> str:
    try:
        return metadata.version("bitsandbytes")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "QLoRA requires bitsandbytes. Install it with `pip install 'jeeva-ai[qlora]'`."
        ) from exc


def _load_qlora_causal_model(
    *,
    model_name_or_path: str,
    device: str,
    dtype: torch.dtype,
    revision: str | None = None,
    local_files_only: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load and prepare a text causal model for NF4 QLoRA training."""
    device_type = torch.device(device).type
    if device_type not in {"mps", "cuda"}:
        raise ValueError("QLoRA training requires an MPS or CUDA device.")

    bitsandbytes_version = _require_qlora_dependencies()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    load_kwargs: dict[str, Any] = {
        "attn_implementation": "sdpa",
        "device_map": {"": device},
        "quantization_config": quantization_config,
        "revision": revision,
        "local_files_only": local_files_only,
        "use_safetensors": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            dtype=dtype,
            **load_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            torch_dtype=dtype,
            **load_kwargs,
        )

    prepared = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
    )
    return prepared, {
        "backend": "bitsandbytes",
        "backend_version": bitsandbytes_version,
        "bits": 4,
        "quant_type": "nf4",
        "double_quantization": True,
        "compute_dtype": str(dtype).removeprefix("torch."),
    }


@dataclass
class TrainingConfig:
    """
    Configuration hyperparameters for Jeeva RLCD fine-tuning.

    Attributes:
        epochs: Number of training epochs.
        lr: Learning rate for AdamW optimizer.
        weight_decay: Weight decay penalty.
        max_grad_norm: Maximum gradient norm for clipping.
        backprop_through_prefill:
            ===================================================================
            CRITICAL ARCHITECTURAL DECISION: backprop_through_prefill
            ===================================================================
            Controls whether gradients backpropagate through the shared
            document prefill KV cache or remain isolated to the branch lanes.

            WHEN SET TO TRUE (Default):
            - Mechanics: Gradients computed on sliced logits at the end of each
              branch lane flow back through the attention keys/values, past the
              replicated DynamicCache, and into the base document's attention heads
              and embedding representations.
            - Advantages: The model learns how to compress the base document
              representation specifically to make subsequent PCD logit extraction
              accurate and well-calibrated. Delivers optimal accuracy and ECE.
            - Disadvantages: Requires keeping the base document computation graph
              in VRAM during lane dispatch. Slightly higher memory consumption.
            - When to use: Recommended by default for standard documents (< 2048
              tokens) on Apple Silicon (MPS) or Nvidia GPUs.

            WHEN SET TO FALSE:
            - Mechanics: Detaches the base KV cache tensors before lane dispatch.
              Gradients only update weights during the branch query step.
            - Advantages: Significantly reduces peak VRAM. Memory is bounded by
              the branch query length rather than document length.
            - Disadvantages: The base document encoder remains frozen; only the
              query slot projections adapt to read the pre-existing representations.
            - When to use: Use when fine-tuning on very long documents (> 4096 tokens)
              or on low-VRAM hardware where out-of-memory (OOM) errors occur.
            ===================================================================
        lora_r: LoRA rank dimension.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout probability.
        lora_targets: List of target module names for LoRA injection.
        task_type: Optional PEFT task type override (defaults to None for VLMs, 'CAUSAL_LM' for text).
        output_dir: Directory path to save model checkpoints.
        device: Device to train on ('mps', 'cuda', 'cpu', or None for auto).
    """

    epochs: int = 5
    lr: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    backprop_through_prefill: bool = True
    contiguous_cache: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    adaptation: Literal["full", "lora", "qlora"] = "lora"
    decoder_head: Literal["letter", "bilinear", "contextual"] = "letter"
    head_lr: float = 1e-3
    candidate_position_balancing: bool = True
    task_type: str | None = None
    output_dir: str = "./checkpoints"
    lr_scheduler: str = "cosine"  # "cosine", "linear", or "constant"
    min_lr_ratio: float = 0.05
    save_best: bool = True
    brier_weight: float = 0.5
    ordinal_weight: float = 0.25
    progress_updates_per_epoch: int = 10
    seed: int = field(default_factory=lambda: secrets.randbelow(2**32))
    device: str | None = None
    dtype: torch.dtype | str | Any = None
    torch_dtype: torch.dtype | str | Any = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.progress_updates_per_epoch < 0:
            raise ValueError("progress_updates_per_epoch cannot be negative.")
        if self.brier_weight < 0.0 or self.ordinal_weight < 0.0:
            raise ValueError("Loss weights cannot be negative.")
        if self.adaptation not in ("full", "lora", "qlora"):
            raise ValueError("adaptation must be 'full', 'lora', or 'qlora'.")
        if self.decoder_head not in ("letter", "bilinear", "contextual"):
            raise ValueError("decoder_head must be 'letter', 'bilinear', or 'contextual'.")
        if self.head_lr <= 0:
            raise ValueError("head_lr must be positive.")


class Trainer:
    """Train a causal decoder with letter readout or a bilinear metric head."""

    def __init__(
        self,
        model_or_id: str | nn.Module,
        schema: Any,
        config: TrainingConfig | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        processor: Any = _UNSET,
        head: BilinearMetricHead | None = None,
    ):
        self.config = config or TrainingConfig()
        self.device = self.config.device or get_default_device()
        self.primitives = normalize_schema(schema)
        self.num_lanes = len(self.primitives)
        self.head: BilinearMetricHead | None = None
        self._option_embedding_cache: dict[str, torch.Tensor] = {}
        self.quantization_metadata: dict[str, Any] | None = None

        if self.config.adaptation == "qlora":
            if not isinstance(model_or_id, str):
                raise ValueError("QLoRA requires a model identifier or saved adapter path.")
            if torch.device(self.device).type not in {"mps", "cuda"}:
                raise ValueError("QLoRA training requires an MPS or CUDA device.")

        # Setup tokenizer and multimodal processor
        model_id = (
            model_or_id
            if isinstance(model_or_id, str)
            else getattr(model_or_id, "name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
        )
        self.model_id = str(model_id)
        self._branch_suffix = "\n\nAssistant:\n"
        self.processor: Any = None
        self.is_multimodal: bool = False
        if processor is not _UNSET:
            self.processor = processor
            if processor is not None and hasattr(processor, "image_processor"):
                self.is_multimodal = True
                self.tokenizer = cast(Any, processor).tokenizer
        else:
            config_source = str(model_id)
            config_revision = self.config.revision
            adapter_path = Path(config_source)
            if adapter_path.is_dir() and (adapter_path / "adapter_config.json").is_file():
                require_local_adapter_safetensors(adapter_path)
                source_adapter_config = PeftConfig.from_pretrained(
                    config_source,
                    local_files_only=self.config.local_files_only,
                )
                if not source_adapter_config.base_model_name_or_path:
                    raise ValueError(
                        f"Adapter at {config_source!r} does not declare base_model_name_or_path."
                    )
                config_source = source_adapter_config.base_model_name_or_path
                config_revision = source_adapter_config.revision or config_revision
            source_config = (
                getattr(model_or_id, "config", None)
                if isinstance(model_or_id, nn.Module)
                else AutoConfig.from_pretrained(
                    config_source,
                    revision=config_revision,
                    local_files_only=self.config.local_files_only,
                )
            )
            if config_declares_multimodal(source_config):
                from transformers import AutoProcessor

                loaded_processor = AutoProcessor.from_pretrained(
                    model_id,
                    revision=(
                        self.config.revision
                        if self.config.tokenizer_revision is None
                        else self.config.tokenizer_revision
                    ),
                    local_files_only=self.config.local_files_only,
                )
                if not hasattr(loaded_processor, "image_processor"):
                    raise ValueError(
                        f"Model {model_id!r} declares image inputs, but its processor "
                        "does not expose an image_processor."
                    )
                self.processor = loaded_processor
                self.is_multimodal = True
                self.tokenizer = loaded_processor.tokenizer

        if not self.is_multimodal:
            if tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    revision=(
                        self.config.revision
                        if self.config.tokenizer_revision is None
                        else self.config.tokenizer_revision
                    ),
                    local_files_only=self.config.local_files_only,
                )
            else:
                self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.config.adaptation == "qlora" and self.is_multimodal:
            raise ValueError("QLoRA currently supports causal text models only.")

        # Setup model and adapter.
        effective_dtype_spec = (
            self.config.dtype if self.config.dtype is not None else self.config.torch_dtype
        )
        if effective_dtype_spec is not None:
            self.torch_dtype = resolve_dtype(effective_dtype_spec, device=self.device)
        elif self.config.adaptation == "qlora":
            self.torch_dtype = torch.float16
        else:
            # Default to float32 for non-quantized training stability.
            self.torch_dtype = torch.float32

        def load_base_model(source: str, *, revision: str | None) -> nn.Module:
            if self.config.adaptation == "qlora":
                base_model, quantization_metadata = _load_qlora_causal_model(
                    model_name_or_path=source,
                    device=self.device,
                    dtype=self.torch_dtype,
                    revision=revision,
                    local_files_only=self.config.local_files_only,
                )
                self.quantization_metadata = quantization_metadata
                return base_model
            return _load_causal_or_vision_model(
                model_name_or_path=source,
                device=self.device,
                dtype=self.torch_dtype,
                revision=revision,
                local_files_only=self.config.local_files_only,
            )

        def attach_adapter(base_model: nn.Module) -> nn.Module:
            effective_task_type = (
                self.config.task_type
                if self.config.task_type is not None
                else (None if self.is_multimodal else "CAUSAL_LM")
            )
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=(
                    "all-linear" if self.config.adaptation == "qlora" else self.config.lora_targets
                ),
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=effective_task_type,
                revision=self.config.revision,
            )
            adapted = get_peft_model(base_model, lora_config)
            adapter_name = "QLoRA" if self.config.adaptation == "qlora" else "LoRA"
            print(f"{adapter_name} adapter attached:")
            adapted.print_trainable_parameters()
            return adapted

        if isinstance(model_or_id, str):
            adapter_path = Path(model_or_id)
            is_adapter = adapter_path.is_dir() and (adapter_path / "adapter_config.json").exists()
            if is_adapter and self.config.adaptation == "full":
                raise ValueError(
                    "Cannot continue a saved causal adapter with adaptation='full'; "
                    "load a merged model or keep adapter-based adaptation."
                )
            if is_adapter:
                require_local_adapter_safetensors(adapter_path)
                peft_config = PeftConfig.from_pretrained(
                    model_or_id,
                    local_files_only=self.config.local_files_only,
                )
                base_model_id = peft_config.base_model_name_or_path
                if not base_model_id:
                    raise ValueError(
                        f"Adapter at {model_or_id!r} does not declare base_model_name_or_path."
                    )
                print(
                    f"Loading trainable adapter '{model_or_id}' over "
                    f"base model '{base_model_id}' on {self.device}..."
                )
                base_model = load_base_model(
                    base_model_id,
                    revision=peft_config.revision or self.config.revision,
                )
                trainable_adapter = PeftModel.from_pretrained(
                    base_model,
                    model_or_id,
                    is_trainable=True,
                    local_files_only=self.config.local_files_only,
                )
                self.model = (
                    trainable_adapter
                    if self.config.adaptation == "qlora"
                    else trainable_adapter.to(self.device)
                )
            else:
                print(
                    f"Loading base model '{terminal_text(model_or_id)}' "
                    f"on {terminal_text(self.device)}..."
                )
                base_model = load_base_model(
                    model_or_id,
                    revision=self.config.revision,
                )
                if self.config.adaptation == "full":
                    self.model = base_model
                else:
                    self.model = attach_adapter(base_model)
        elif isinstance(model_or_id, nn.Module):
            self.model = model_or_id.to(self.device)
            if isinstance(self.model, PeftModel):
                if self.config.adaptation == "full":
                    raise ValueError(
                        "Cannot continue a saved causal adapter with adaptation='full'; "
                        "load a merged model or keep adapter-based adaptation."
                    )
                active_adapter = self.model.active_adapter
                if not isinstance(active_adapter, str):
                    raise ValueError(
                        "Continuing training requires exactly one active causal adapter."
                    )
                self.model.set_adapter(active_adapter)
            elif self.config.adaptation == "lora":
                self.model = attach_adapter(self.model)
        else:
            raise TypeError(f"Unsupported model_or_id type: {type(model_or_id)}")

        if self.config.decoder_head in ("bilinear", "contextual"):
            model_config: Any = self.model.config
            if hasattr(model_config, "get_text_config"):
                model_config = model_config.get_text_config(decoder=True)
            hidden_dim = int(model_config.hidden_size)
            expected_strategy = (
                "contextual" if self.config.decoder_head == "contextual" else "static"
            )
            metric_head_dtype = torch.float32
            head_path = (
                Path(model_or_id) / DECODER_HEAD_DIR if isinstance(model_or_id, str) else None
            )
            if head is not None:
                if head.input_strategy != expected_strategy:
                    raise ValueError(
                        f"Loaded decoder head uses {head.input_strategy!r} inputs; "
                        f"decoder_head={self.config.decoder_head!r} requires "
                        f"{expected_strategy!r} inputs."
                    )
                self.head = head.to(
                    device=self.device,
                    dtype=metric_head_dtype,
                )
            elif head_path is not None and head_path.is_dir():
                loaded_head = BaseDecisionHead.from_pretrained(
                    head_path,
                    map_location=self.device,
                    expected_hidden_dim=hidden_dim,
                )
                if not isinstance(loaded_head, BilinearMetricHead):
                    raise TypeError(f"{head_path} does not contain a BilinearMetricHead.")
                if loaded_head.input_strategy != expected_strategy:
                    raise ValueError(
                        f"{head_path} uses {loaded_head.input_strategy!r} inputs; "
                        f"decoder_head={self.config.decoder_head!r} requires "
                        f"{expected_strategy!r} inputs."
                    )
                self.head = loaded_head.to(
                    device=self.device,
                    dtype=metric_head_dtype,
                )
            else:
                self.head = BilinearMetricHead(
                    hidden_dim=hidden_dim,
                    input_strategy=expected_strategy,
                ).to(
                    device=self.device,
                    dtype=metric_head_dtype,
                )

        self.loss_fn = CompositeDecisionLoss(
            brier_weight=self.config.brier_weight,
            ordinal_weight=self.config.ordinal_weight,
        )

        model_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        parameter_groups: list[dict[str, Any]] = [
            {"params": model_parameters, "lr": self.config.lr}
        ]
        if self.head is not None:
            head_parameters = [
                parameter for parameter in self.head.parameters() if parameter.requires_grad
            ]
            parameter_groups.append({"params": head_parameters, "lr": self.config.head_lr})

        self._optimized_parameters = [
            parameter for group in parameter_groups for parameter in group["params"]
        ]
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=self.config.weight_decay,
        )

    def _get_or_compute_option_embeddings(self, primitive: BasePrimitive) -> torch.Tensor:
        candidate_texts = primitive.options
        cache_key = f"{primitive.kind}:{tuple(candidate_texts)!r}"

        if cache_key not in self._option_embedding_cache:
            opt_inputs = self.tokenizer(candidate_texts, return_tensors="pt", padding=True).to(
                self.device
            )
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad():
                    opt_out = self.model(**opt_inputs, output_hidden_states=True)
                    mask = opt_inputs.attention_mask.unsqueeze(-1)
                    emb = (opt_out.hidden_states[-1] * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                        min=1
                    )
                    self._option_embedding_cache[cache_key] = torch.nn.functional.normalize(
                        emb, p=2, dim=-1
                    ).detach()
            finally:
                self.model.train(was_training)
        return self._option_embedding_cache[cache_key]

    def _build_training_branches(
        self,
    ) -> tuple[
        list[str],
        list[list[int]],
        list[list[tuple[int, ...]]],
        list[tuple[int, ...] | None],
    ]:
        prompts: list[str] = []
        lane_branch_indices: list[list[int]] = []
        lane_orders: list[list[tuple[int, ...]]] = []
        lane_token_ids: list[tuple[int, ...] | None] = []

        for primitive in self.primitives:
            branch_indices: list[int] = []
            if self.config.decoder_head == "letter":
                orders = (
                    balanced_candidate_orders(len(primitive.options))
                    if self.config.candidate_position_balancing
                    else (tuple(range(len(primitive.options))),)
                )
                token_ids: tuple[int, ...] | None = None
                for order in orders:
                    prompt, _, letters = primitive.build_letter_branch_prompt(order)
                    if token_ids is None:
                        encoded_ids: list[int] = []
                        for letter in letters:
                            encoded = self.tokenizer.encode(letter, add_special_tokens=False)
                            if len(encoded) != 1:
                                raise ValueError(
                                    f"Letter scoring requires letter {letter!r} to be "
                                    "one token. Use decoder_head='bilinear' or 'contextual' "
                                    "for this model."
                                )
                            encoded_ids.append(encoded[0])
                        token_ids = tuple(encoded_ids)
                    branch_indices.append(len(prompts))
                    prompts.append(f"{prompt}{self._branch_suffix}Answer: ")
                lane_orders.append(list(orders))
                lane_token_ids.append(token_ids)
            elif self.config.decoder_head == "contextual":
                query_prompt, candidate_prompts = build_contextual_metric_prompts(
                    primitive.prompt,
                    primitive.options,
                )
                for prompt in (query_prompt, *candidate_prompts):
                    branch_indices.append(len(prompts))
                    prompts.append(f"{self._branch_suffix}{prompt}")
                lane_orders.append([])
                lane_token_ids.append(None)
            else:
                branch_indices.append(len(prompts))
                prompts.append(f"{self._branch_suffix}{primitive.build_branch_prompt()}")
                lane_orders.append([])
                lane_token_ids.append(None)
            lane_branch_indices.append(branch_indices)

        return prompts, lane_branch_indices, lane_orders, lane_token_ids

    def _extract_context_text(self, context: Any) -> str:
        """Returns the raw text portion of a context."""
        if isinstance(context, MultimodalContext):
            return context.text
        return str(context)

    def _build_multimodal_message(self, context: MultimodalContext) -> list[dict[str, Any]]:
        """Builds user chat turn with image placeholders and text."""
        parts: list[dict[str, Any]] = []
        for _ in context.images:
            parts.append({"type": "image"})
        parts.append({"type": "text", "text": f"Document:\n{context.text}"})
        return parts

    def _build_prefill_inputs(self, context: Any) -> dict[str, Any]:
        """Tokenize text or encode multimodal inputs for Pass 1 prefill."""
        if isinstance(context, MultimodalContext) and context.has_images():
            if not self.is_multimodal or self.processor is None:
                raise ValueError(
                    "MultimodalContext with images requires a vision-language model, "
                    "but this Trainer was loaded without an image_processor."
                )

            prompt_text, self._branch_suffix = build_chat_prefill(
                self.processor,
                self._build_multimodal_message(context),
            )
            pil_images = context.load_pil_images()
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

    def _report_training_progress(
        self,
        *,
        completed_samples: int,
        total_samples: int,
        progress_interval: int | None,
        total_loss: float,
        num_steps: int,
        started_at: float,
    ) -> None:
        if progress_interval is None or (
            completed_samples % progress_interval != 0 and completed_samples != total_samples
        ):
            return

        elapsed = time.perf_counter() - started_at
        eta = elapsed * (total_samples - completed_samples) / max(1, completed_samples)
        running_loss = total_loss / max(1, num_steps)
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"  Progress {completed_samples}/{total_samples} "
            f"({completed_samples / total_samples * 100:.1f}%) | "
            f"loss={running_loss:.6f} | "
            f"lr={current_lr:.2e} | "
            f"elapsed={elapsed:.1f}s | eta={eta:.1f}s",
            flush=True,
        )

    def train(
        self,
        train_dataset: JeevaDataset,
        eval_dataset: JeevaDataset | None = None,
    ) -> dict[str, Any]:
        """
        Runs the multi-lane PCD training loop.
        """
        validation_samples = len(eval_dataset) if eval_dataset is not None else 0
        print(
            f"\nTraining started | samples={len(train_dataset)} | "
            f"validation={validation_samples} | epochs={self.config.epochs}",
            flush=True,
        )

        total_steps = len(train_dataset) * self.config.epochs
        scheduler = None
        if self.config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_steps),
                eta_min=self.config.lr * self.config.min_lr_ratio,
            )
        elif self.config.lr_scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr_ratio,
                total_iters=max(1, total_steps),
            )

        epoch_history: list[dict[str, float]] = []
        best_metric = float("inf")
        saved_best = False

        for epoch in range(1, self.config.epochs + 1):
            t_epoch_start = time.perf_counter()
            self.model.train()
            total_loss = 0.0
            num_steps = 0
            total_samples = len(train_dataset)
            sample_order = _epoch_sample_order(
                total_samples,
                seed=self.config.seed,
                epoch=epoch,
            )
            progress_interval = (
                ceil(total_samples / self.config.progress_updates_per_epoch)
                if self.config.progress_updates_per_epoch > 0 and total_samples > 0
                else None
            )
            print(
                f"Epoch {epoch:02d}/{self.config.epochs:02d} started | samples={total_samples}",
                flush=True,
            )

            for completed_samples, sample_idx in enumerate(sample_order, start=1):
                sample = train_dataset[sample_idx]
                doc = sample["document"]
                targets = sample["targets"]
                lane_mask = sample["lane_mask"].to(self.device)

                # Skip samples with no active labels
                if lane_mask.sum() == 0:
                    self._report_training_progress(
                        completed_samples=completed_samples,
                        total_samples=total_samples,
                        progress_interval=progress_interval,
                        total_loss=total_loss,
                        num_steps=num_steps,
                        started_at=t_epoch_start,
                    )
                    continue

                self.optimizer.zero_grad()

                # ---------------------------------------------------------
                # PASS 1: Base Prefill
                # ---------------------------------------------------------
                base_inputs = self._build_prefill_inputs(doc)

                if self.config.backprop_through_prefill:
                    base_out = self.model(**base_inputs, use_cache=True)
                else:
                    with torch.no_grad():
                        base_out = self.model(**base_inputs, use_cache=True)

                base_cache: DynamicCache = base_out.past_key_values
                past_len = base_cache.get_seq_length()

                # ---------------------------------------------------------
                # PASS 2: Parallel branch expansion
                # ---------------------------------------------------------
                (
                    branch_prompts,
                    lane_branch_indices,
                    lane_orders,
                    lane_token_ids,
                ) = self._build_training_branches()
                expanded_cache = replicate_dynamic_cache(
                    base_cache,
                    num_lanes=len(branch_prompts),
                    contiguous=self.config.contiguous_cache,
                    autograd_safe=True,
                )
                branch_inputs = self.tokenizer(
                    branch_prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                comb_mask, pos_ids = build_branch_attention_and_positions(
                    past_len=past_len,
                    branch_attention_mask=branch_inputs.attention_mask,
                    device=torch.device(self.device),
                )
                branch_out = self.model(
                    input_ids=branch_inputs.input_ids,
                    attention_mask=comb_mask,
                    position_ids=pos_ids,
                    past_key_values=expanded_cache,
                    use_cache=False,
                    output_hidden_states=self.head is not None,
                )

                # ---------------------------------------------------------
                # PASS 3: Loss through the selected decoder head
                # ---------------------------------------------------------
                sample_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                active_lanes = 0
                for lane_idx, primitive in enumerate(self.primitives):
                    if lane_mask[lane_idx] == 0.0:
                        continue

                    branch_indices = lane_branch_indices[lane_idx]
                    if self.head is None:
                        letter_token_ids = lane_token_ids[lane_idx]
                        if letter_token_ids is None:
                            raise RuntimeError("Letter scoring is missing candidate token ids.")
                        positional_scores = torch.stack(
                            [
                                branch_out.logits[
                                    branch_index,
                                    -1,
                                    list(letter_token_ids),
                                ]
                                for branch_index in branch_indices
                            ]
                        )
                        scores = aggregate_candidate_scores(
                            positional_scores,
                            lane_orders[lane_idx],
                            temperature=1.0,
                        ).scores
                    else:
                        hidden_states = branch_out.hidden_states
                        if hidden_states is None:
                            raise RuntimeError("Bilinear decoder head requires hidden states.")
                        branch_hidden = hidden_states[-1][branch_indices[0], -1, :]
                        if self.head.input_strategy == "contextual":
                            expected_branches = len(primitive.options) + 1
                            if len(branch_indices) != expected_branches:
                                raise RuntimeError(
                                    f"Contextual field {primitive.key or primitive.kind!r} "
                                    f"requires {expected_branches} branches; "
                                    f"received {len(branch_indices)}."
                                )
                            option_embeddings = hidden_states[-1][
                                branch_indices[1:],
                                -1,
                                :,
                            ]
                        else:
                            option_embeddings = self._get_or_compute_option_embeddings(primitive)
                        scores = self.head(branch_hidden, option_embeddings)

                    target_tensor = targets[lane_idx].to(self.device)
                    level_values = (
                        torch.tensor(primitive.level_values, device=self.device)
                        if isinstance(primitive, Score)
                        else None
                    )
                    lane_loss = self.loss_fn(
                        scores,
                        target_tensor,
                        level_values=level_values,
                    )
                    sample_loss = sample_loss + lane_loss
                    active_lanes += 1

                if active_lanes > 0:
                    loss = sample_loss / active_lanes
                    step_context = (
                        f"epoch {epoch}, sample {completed_samples}/{total_samples}, "
                        f"scorer {self.config.decoder_head}, device {self.device}, "
                        f"dtype {self.torch_dtype}"
                    )
                    backward_finite_loss(loss, context=step_context)

                    clip_finite_gradients(
                        self._optimized_parameters,
                        max_norm=self.config.max_grad_norm,
                        context=step_context,
                    )

                    self.optimizer.step()
                    self._option_embedding_cache.clear()
                    if scheduler is not None:
                        scheduler.step()

                    total_loss += loss.item()
                    num_steps += 1
                self._report_training_progress(
                    completed_samples=completed_samples,
                    total_samples=total_samples,
                    progress_interval=progress_interval,
                    total_loss=total_loss,
                    num_steps=num_steps,
                    started_at=t_epoch_start,
                )

            mean_loss = total_loss / max(1, num_steps)
            epoch_duration = time.perf_counter() - t_epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:02d}/{self.config.epochs:02d} complete | "
                f"loss={mean_loss:.6f} | "
                f"lr={current_lr:.2e} | "
                f"duration={epoch_duration:.1f}s"
            )

            epoch_stat = {
                "epoch": epoch,
                "train_loss": mean_loss,
                "lr": current_lr,
                "duration_seconds": epoch_duration,
            }

            # Optional validation run
            if eval_dataset is not None:
                print(
                    f"  Validation started | samples={len(eval_dataset)}",
                    flush=True,
                )
                self.model.eval()
                temp_engine = DecisionEngine.from_components(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    model_id=self.model_id,
                    device=self.device,
                    dtype=self.torch_dtype,
                    mode="letter" if self.head is None else "dense",
                    head=self.head,
                    candidate_position_balancing=(
                        self.config.candidate_position_balancing if self.head is None else False
                    ),
                    contiguous_cache=self.config.contiguous_cache,
                )
                evaluator = Evaluator(model=temp_engine, schema=self.primitives)
                scorecard = evaluator.run(eval_dataset)
                categorical_metrics = [
                    metrics for metrics in scorecard.field_metrics.values() if "accuracy" in metrics
                ]
                score_metrics = [
                    metrics for metrics in scorecard.field_metrics.values() if "mae" in metrics
                ]
                if categorical_metrics:
                    epoch_stat["eval_accuracy"] = scorecard.overall_accuracy
                if score_metrics:
                    epoch_stat["eval_score_mae"] = sum(
                        metrics["mae"] for metrics in score_metrics
                    ) / len(score_metrics)
                epoch_stat["eval_brier_score"] = scorecard.overall_brier_score
                summary_parts = []
                if categorical_metrics:
                    summary_parts.append(
                        f"Categorical Accuracy: {scorecard.overall_accuracy * 100:.1f}%"
                    )
                if score_metrics:
                    summary_parts.append(f"Score MAE: {epoch_stat['eval_score_mae']:.4f}")
                summary_parts.append(f"Brier: {scorecard.overall_brier_score:.4f}")
                print(f"  Validation complete | {' | '.join(summary_parts)}")

            epoch_history.append(epoch_stat)

            # Best model tracking & auto-checkpointing
            current_metric = (
                epoch_stat["eval_brier_score"] if eval_dataset is not None else mean_loss
            )
            require_finite_value(
                float(current_metric),
                name="checkpoint metric",
                context=f"epoch {epoch}, scorer {self.config.decoder_head}",
            )
            if self.config.save_best and current_metric < best_metric:
                best_metric = current_metric
                self.save_checkpoint(self.config.output_dir)
                saved_best = True
                print(
                    f"  Checkpoint updated | path={self.config.output_dir} | "
                    f"metric={best_metric:.6f}"
                )

        if not saved_best:
            self.save_checkpoint(self.config.output_dir)
            print(f"Checkpoint saved | path={terminal_text(self.config.output_dir)}")

        report: dict[str, Any] = {
            "history": epoch_history,
            "output_dir": self.config.output_dir,
            "architecture": "causal",
            "scorer": self.config.decoder_head,
            "adaptation": self.config.adaptation,
        }
        if self.quantization_metadata is not None:
            report["quantization"] = dict(self.quantization_metadata)
        return report

    def save_checkpoint(self, path: str | Path | None = None) -> str:
        """Saves LoRA adapter weights and tokenizer/processor."""
        out_dir = Path(path or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_to_save: Any = self.model
        model_to_save.save_pretrained(str(out_dir), safe_serialization=True)

        if self.head is not None:
            self.head.save_pretrained(out_dir / DECODER_HEAD_DIR)

        self.tokenizer.save_pretrained(out_dir)
        if self.processor is not None:
            self.processor.save_pretrained(out_dir)
        return str(out_dir)

    def export_merged(self, output_path: str | Path) -> str:
        """
        Merges LoRA weights back into the base model weights via merge_and_unload()
        and exports a standalone HuggingFace model for zero-overhead inference.
        """
        if self.config.adaptation == "qlora":
            raise ValueError(
                "A live 4-bit QLoRA model cannot be merged safely. Save the adapter, "
                "then use DecisionModel.save(..., merge=True) to merge it over an "
                "unquantized base model."
            )
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.model, PeftModel):
            print("Merging LoRA weights into base model...")
            merged_model: Any = self.model.merge_and_unload()
            merged_model.save_pretrained(str(out_dir), safe_serialization=True)
            self.tokenizer.save_pretrained(str(out_dir))
            if self.processor is not None:
                self.processor.save_pretrained(str(out_dir))
            print(f"Merged model saved for zero-overhead deployment at: {terminal_text(out_dir)}")
        else:
            model_to_save: Any = self.model
            model_to_save.save_pretrained(str(out_dir), safe_serialization=True)
            self.tokenizer.save_pretrained(str(out_dir))
            if self.processor is not None:
                self.processor.save_pretrained(str(out_dir))

        if self.head is not None:
            self.head.save_pretrained(out_dir / DECODER_HEAD_DIR)

        return str(out_dir)
