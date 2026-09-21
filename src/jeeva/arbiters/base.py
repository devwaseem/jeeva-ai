from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from jeeva._weights import require_local_adapter_safetensors
from jeeva.core.prompts import build_chat_prefill
from jeeva.schema.primitives import BasePrimitive, DecisionResult


def get_default_device() -> str:
    """Detects the fastest available device with native Apple Silicon (MPS) priority."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_arbiter_model(
    model_or_id: str | Any,
    device: str | None = None,
    dtype: torch.dtype | str | Any = None,
    torch_dtype: torch.dtype | str | Any = None,
) -> tuple[Any, Any, str]:
    """
    Loads a model and tokenizer for an Arbiter, handling raw HuggingFace IDs,
    local paths, and PEFT adapters.
    """
    from jeeva.types import resolve_dtype

    target_device = device or get_default_device()
    effective_spec = dtype if dtype is not None else torch_dtype
    resolved_dtype = resolve_dtype(effective_spec, device=target_device)

    if not isinstance(model_or_id, str):
        # Already an instantiated model
        name_or_path = getattr(model_or_id, "name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model_or_id, tokenizer, target_device

    model_id = model_or_id
    is_peft = Path(model_id).is_dir() and (Path(model_id) / "adapter_config.json").exists()

    if is_peft:
        require_local_adapter_safetensors(model_id)
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise ImportError(
                "Loading a PEFT adapter requires the training extra. "
                "Install it with: pip install 'jeeva-ai[training]'"
            ) from exc
        peft_cfg = PeftConfig.from_pretrained(model_id)
        base_id = peft_cfg.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model: Any = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=base_id or model_id,
            torch_dtype=resolved_dtype,
            use_safetensors=True,
        )
        base_model = base_model.to(target_device)
        model = PeftModel.from_pretrained(base_model, model_id).to(target_device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if target_device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_id,
                torch_dtype=resolved_dtype,
                device_map="cuda",
                use_safetensors=True,
            )
        else:
            loaded_model: Any = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_id,
                torch_dtype=resolved_dtype,
                use_safetensors=True,
            )
            model = loaded_model.to(target_device)

    model.eval()
    return model, tokenizer, target_device


def prefill_context(
    model: Any,
    tokenizer: Any,
    context: str,
    device: str,
) -> tuple[DynamicCache, int, str]:
    """Prefill a document and return its cache plus model-specific branch suffix."""
    prefill_prompt, branch_suffix = build_chat_prefill(tokenizer, f"Document:\n{context}")
    base_inputs = tokenizer(prefill_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        base_out = model(**base_inputs, use_cache=True)
    base_cache: DynamicCache = base_out.past_key_values
    past_len = base_cache.get_seq_length()
    return base_cache, past_len, branch_suffix


class BaseArbiter(ABC):
    """Contract implemented by Jeeva decision arbiters.

    By default, ``DecisionModel`` invokes an arbiter only for fields whose
    initial decision is uncertain. Arbiters may opt into resolving every field
    when their strategy is intentionally authoritative rather than a fallback.

    Built-in implementations are ``SequenceArbiter``,
    ``SequenceThinkingArbiter``, and ``CloudArbiter``. Applications can
    subclass ``BaseArbiter`` when resolution belongs to another model or an
    application-specific adapter.
    """

    @property
    def resolve_all_fields(self) -> bool:
        """Whether this arbiter should replace certain as well as uncertain fields."""
        return False

    @property
    def uses_primary_model(self) -> bool:
        """Whether resolution requires the primary causal runtime and its KV cache."""
        return False

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name of this arbiter strategy."""
        pass

    @abstractmethod
    def resolve(
        self,
        context: str,
        ambiguous_fields: dict[str, BasePrimitive],
        current_decisions: dict[str, DecisionResult],
        engine: Any,
        base_cache: Any,
        past_len: int,
        threshold: float,
    ) -> dict[str, DecisionResult]:
        """
        Resolves ambiguous decisions and returns updated DecisionResults.

        Args:
            context: The raw input document or context string.
            ambiguous_fields: Map of field keys to their BasePrimitive schema specifications
                              for only the lanes that were flagged as ambiguous.
            current_decisions: The preliminary DecisionResults produced by the Gatekeeper.
            engine: Reference to the calling DecisionEngine (providing access to tokenizer,
                    model, and device).
            base_cache: The pre-computed DynamicCache from the document prefill pass.
            past_len: The sequence length of the prefilled document tokens.
            threshold: The certainty threshold tau.

        Returns:
            Dictionary mapping field keys to updated, resolved DecisionResult objects.
        """
        pass
