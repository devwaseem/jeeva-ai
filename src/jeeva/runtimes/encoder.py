from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer

from jeeva._json import MAX_CONFIG_JSON_BYTES, read_json
from jeeva._weights import require_local_adapter_safetensors
from jeeva.arbiters.base import BaseArbiter
from jeeva.core.engine import EvaluationResult
from jeeva.core.prompts import validate_context_budget
from jeeva.heads.base import MAX_HEAD_PARAMETERS
from jeeva.schema.multimodal import MultimodalContext
from jeeva.schema.primitives import BasePrimitive, DecisionResult, Noul
from jeeva.types import resolve_dtype

HEAD_CONFIG_NAME = "candidate_marker_head.json"
HEAD_WEIGHTS_NAME = "candidate_marker_head.safetensors"
HEAD_FORMAT_VERSION = 1


def _field_name(primitive: BasePrimitive) -> str:
    return primitive.key or str(cast(Any, primitive).kind)


def _candidate_orders(
    primitives: Sequence[BasePrimitive],
    option_orders: Sequence[Sequence[int]] | None,
) -> list[list[int]]:
    if option_orders is not None and len(option_orders) != len(primitives):
        raise ValueError("option_orders must contain one permutation per decision field.")
    resolved: list[list[int]] = []
    for field_index, primitive in enumerate(primitives):
        expected = list(range(len(primitive.candidates)))
        order = expected if option_orders is None else list(option_orders[field_index])
        if (
            any(isinstance(index, bool) or not isinstance(index, int) for index in order)
            or sorted(order) != expected
        ):
            raise ValueError(
                f"Candidate order for {_field_name(primitive)!r} must be a permutation "
                f"of {expected}; received {order}."
            )
        resolved.append(order)
    return resolved


class CandidateMarkerHead(nn.Module):
    """Shared nonlinear scorer for candidate-marker representations."""

    def __init__(self, hidden_size: int, scorer_size: int | None = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.scorer_size = scorer_size or max(64, hidden_size // 2)
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.scorer_size),
            nn.GELU(),
            nn.LayerNorm(self.scorer_size),
            nn.Dropout(0.1),
            nn.Linear(self.scorer_size, 1),
        )

    def forward(self, marker_hidden: torch.Tensor) -> torch.Tensor:
        return self.network(marker_hidden).squeeze(-1)


def save_candidate_marker_head(head: CandidateMarkerHead, directory: str | Path) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    config = {
        "format_version": HEAD_FORMAT_VERSION,
        "architecture": "candidate_marker_mlp",
        "hidden_size": head.hidden_size,
        "scorer_size": head.scorer_size,
    }
    (destination / HEAD_CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
    state = {name: tensor.detach().to("cpu") for name, tensor in head.state_dict().items()}
    save_file(state, destination / HEAD_WEIGHTS_NAME)


def load_candidate_marker_head(
    directory: str | Path,
    *,
    hidden_size: int,
    device: str,
    required: bool,
) -> CandidateMarkerHead:
    source = Path(directory)
    config_path = source / HEAD_CONFIG_NAME
    weights_path = source / HEAD_WEIGHTS_NAME
    scorer_size: int | None = None
    if required and (not config_path.is_file() or not weights_path.is_file()):
        raise ValueError(
            f"Encoder artifact at {source} is missing {HEAD_CONFIG_NAME} or {HEAD_WEIGHTS_NAME}."
        )
    if config_path.is_file():
        config = read_json(config_path, max_bytes=MAX_CONFIG_JSON_BYTES)
        if not isinstance(config, dict):
            raise ValueError("Candidate-marker head config must be a JSON object.")
        if int(config.get("format_version", -1)) != HEAD_FORMAT_VERSION:
            raise ValueError("Unsupported candidate-marker head format.")
        if config.get("architecture") != "candidate_marker_mlp":
            raise ValueError("Unsupported candidate-marker head architecture.")
        if int(config.get("hidden_size", -1)) != hidden_size:
            raise ValueError(
                "Candidate-marker head hidden size does not match the encoder backbone."
            )
        raw_scorer_size = config.get("scorer_size")
        if (
            isinstance(raw_scorer_size, bool)
            or not isinstance(raw_scorer_size, int)
            or raw_scorer_size <= 0
        ):
            raise ValueError("Candidate-marker scorer_size must be a positive integer.")
        scorer_size = raw_scorer_size

    resolved_scorer_size = scorer_size or max(64, hidden_size // 2)
    parameter_count = (
        hidden_size * resolved_scorer_size + 4 * resolved_scorer_size + 2 * hidden_size + 1
    )
    if parameter_count > MAX_HEAD_PARAMETERS:
        raise ValueError(
            f"Candidate-marker head requests {parameter_count:,} parameters; maximum is "
            f"{MAX_HEAD_PARAMETERS:,}."
        )
    if weights_path.is_file():
        expected_shapes = {
            "network.0.weight": (hidden_size,),
            "network.0.bias": (hidden_size,),
            "network.1.weight": (resolved_scorer_size, hidden_size),
            "network.1.bias": (resolved_scorer_size,),
            "network.3.weight": (resolved_scorer_size,),
            "network.3.bias": (resolved_scorer_size,),
            "network.5.weight": (1, resolved_scorer_size),
            "network.5.bias": (1,),
        }
        with safe_open(weights_path, framework="pt", device="cpu") as tensors:
            actual_shapes = {
                name: tuple(tensors.get_slice(name).get_shape()) for name in tensors.keys()
            }
        if actual_shapes != expected_shapes:
            raise ValueError(
                "Candidate-marker tensor shapes do not match config: "
                f"expected {expected_shapes}, received {actual_shapes}."
            )

    head = CandidateMarkerHead(hidden_size, resolved_scorer_size).to(device)
    if weights_path.is_file():
        head.load_state_dict(load_file(weights_path), strict=True)
    return head


def _candidate_isolation(
    attention_mask: torch.Tensor,
    marker_positions: Sequence[Sequence[int]],
    *,
    sliding_window: int | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Build isolated candidate attention and order-invariant positions."""

    batch_size, sequence_length = attention_mask.shape
    device = attention_mask.device
    candidate_ids = torch.full(
        (batch_size, sequence_length),
        -2,
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(sequence_length, device=device).repeat(batch_size, 1)
    last_content = attention_mask.long().sum(dim=1) - 1

    for row_index, positions in enumerate(marker_positions):
        if not positions:
            raise ValueError("Each packed encoder row must contain at least one candidate marker.")
        prefix_length = positions[0]
        candidate_ids[row_index, :prefix_length] = -1
        for candidate_index, start in enumerate(positions):
            stop = (
                positions[candidate_index + 1]
                if candidate_index + 1 < len(positions)
                else int(last_content[row_index].item())
            )
            candidate_ids[row_index, start:stop] = candidate_index
            position_ids[row_index, start:stop] = torch.arange(
                prefix_length,
                prefix_length + stop - start,
                device=device,
            )

    query_ids = candidate_ids.unsqueeze(2)
    key_ids = candidate_ids.unsqueeze(1)
    query_is_prefix = query_ids == -1
    key_is_prefix = key_ids == -1
    query_is_candidate = query_ids >= 0
    same_candidate = query_ids == key_ids
    allowed = (query_is_prefix & key_is_prefix) | (
        query_is_candidate & (key_is_prefix | same_candidate)
    )
    allowed &= attention_mask.bool().unsqueeze(1)
    identity = torch.eye(sequence_length, dtype=torch.bool, device=device).unsqueeze(0)
    allowed |= identity
    full_attention = allowed.unsqueeze(1)

    if sliding_window is None:
        sliding_attention = full_attention
    else:
        query_positions = position_ids.unsqueeze(2)
        key_positions = position_ids.unsqueeze(1)
        local = (query_positions - key_positions).abs() <= sliding_window
        sliding_attention = ((allowed & local) | identity).unsqueeze(1)

    return {
        "full_attention": full_attention,
        "sliding_attention": sliding_attention,
    }, position_ids


def _load_auto_model(
    source: str,
    *,
    device: str,
    dtype: str,
    trainable: bool,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Any:
    resolved_dtype = resolve_dtype(dtype, device=device)
    source_path = Path(source)
    if source_path.is_dir() and (source_path / "adapter_config.json").is_file():
        require_local_adapter_safetensors(source_path)
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise ImportError(
                "Loading an encoder adapter requires the training extra. "
                "Install it with: pip install 'jeeva-ai[training]'"
            ) from exc
        adapter_config = PeftConfig.from_pretrained(
            source,
            local_files_only=local_files_only,
        )
        base_model_name = adapter_config.base_model_name_or_path
        if not base_model_name:
            raise ValueError(f"Encoder adapter at {source!r} does not declare its base model.")
        base_model = AutoModel.from_pretrained(
            base_model_name,
            dtype=resolved_dtype,
            revision=adapter_config.revision or revision,
            local_files_only=local_files_only,
            use_safetensors=True,
        ).to(device)
        return PeftModel.from_pretrained(
            base_model,
            source,
            is_trainable=trainable,
            local_files_only=local_files_only,
        ).to(device)
    return AutoModel.from_pretrained(
        source,
        dtype=resolved_dtype,
        revision=revision,
        local_files_only=local_files_only,
        use_safetensors=True,
    ).to(device)


class EncoderRuntime:
    """Bidirectional candidate-marker runtime with dynamic schema candidates."""

    architecture: Literal["encoder"] = "encoder"
    scorer = "candidate_marker"
    processor = None
    is_multimodal = False

    def __init__(
        self,
        source: str,
        *,
        device: str,
        dtype: str,
        max_length: int,
        trained: bool,
        trainable: bool = False,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.source = source
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.trained = trained
        self.tokenizer_revision = revision if tokenizer_revision is None else tokenizer_revision
        self.local_files_only = local_files_only
        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=self.tokenizer_revision,
            local_files_only=local_files_only,
        )
        if self.tokenizer.mask_token_id is None or self.tokenizer.mask_token is None:
            raise ValueError(
                "Encoder tokenizers must define a mask token for candidate-marker scoring."
            )
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("Encoder candidate-marker scoring requires a fast tokenizer.")
        self.tokenizer.padding_side = "right"
        self.model = _load_auto_model(
            source,
            device=device,
            dtype=dtype,
            trainable=trainable,
            revision=revision,
            local_files_only=local_files_only,
        )
        model_config = getattr(self.model, "config", None)
        model_name = getattr(model_config, "_name_or_path", None)
        model_commit = getattr(model_config, "_commit_hash", None)
        tokenizer_name = getattr(self.tokenizer, "name_or_path", None)
        tokenizer_kwargs = getattr(self.tokenizer, "init_kwargs", None)
        tokenizer_commit = (
            tokenizer_kwargs.get("_commit_hash") if isinstance(tokenizer_kwargs, dict) else None
        )
        self.resolved_model_id = model_name if isinstance(model_name, str) else source
        self.resolved_model_revision = model_commit if isinstance(model_commit, str) else revision
        self.resolved_tokenizer_id = tokenizer_name if isinstance(tokenizer_name, str) else source
        self.resolved_tokenizer_revision = (
            tokenizer_commit if isinstance(tokenizer_commit, str) else self.tokenizer_revision
        )
        model_type = getattr(self.model.config, "model_type", None)
        if model_type not in (None, "modernbert"):
            raise ValueError(
                "Candidate-marker scoring currently requires a ModernBERT encoder; "
                f"received model_type={model_type!r}."
            )
        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(model_limit, int) and max_length > model_limit:
            raise ValueError(
                f"max_length={max_length} exceeds this encoder's "
                f"max_position_embeddings={model_limit}."
            )
        hidden_size = int(self.model.config.hidden_size)
        self.head = load_candidate_marker_head(
            source,
            hidden_size=hidden_size,
            device=device,
            required=trained,
        )
        model_dtype = next(self.model.parameters()).dtype
        self.head.to(device=device, dtype=model_dtype)
        self._lock = threading.Lock()
        if not trainable:
            self.model.eval()
            self.head.eval()

    @classmethod
    def from_components(
        cls,
        *,
        model: Any,
        tokenizer: Any,
        head: CandidateMarkerHead,
        device: str,
        max_length: int,
        resolved_model_id: str | None = None,
        resolved_model_revision: str | None = None,
        resolved_tokenizer_id: str | None = None,
        resolved_tokenizer_revision: str | None = None,
    ) -> EncoderRuntime:
        instance = cls.__new__(cls)
        instance.source = str(getattr(model, "name_or_path", "encoder"))
        instance.device = device
        instance.dtype = str(next(model.parameters()).dtype)
        instance.max_length = max_length
        instance.trained = True
        instance.tokenizer = tokenizer
        instance.model = model
        instance.head = head
        instance._lock = threading.Lock()
        instance.resolved_model_id = resolved_model_id or instance.source
        instance.resolved_model_revision = resolved_model_revision
        instance.resolved_tokenizer_id = resolved_tokenizer_id or instance.source
        instance.resolved_tokenizer_revision = resolved_tokenizer_revision
        instance.model.eval()
        instance.head.eval()
        return instance

    def warmup(self) -> None:
        if not self.trained:
            raise ValueError("Encoder models must be fitted before inference.")
        self.evaluate(
            "System warmup.",
            [
                Noul(
                    "Determine whether the system is ready.",
                    {
                        True: "The system is ready.",
                        False: "The system is not ready.",
                    },
                ).bind("ready")
            ],
        )

    def _escape_marker(self, text: str) -> str:
        marker = str(self.tokenizer.mask_token)
        replacement = str(getattr(self.tokenizer, "unk_token", None) or "mask")
        return text.replace(marker, replacement)

    def _field_prompt(self, primitive: BasePrimitive, context: str) -> str:
        return "\n".join(
            (
                f"Field: {self._escape_marker(_field_name(primitive))}",
                "Instructions:",
                self._escape_marker(str(cast(Any, primitive).instructions)),
                "Context:",
                self._escape_marker(context),
            )
        )

    def _candidate_block(self, primitive: BasePrimitive, candidate_index: int) -> str:
        candidate = primitive.candidates[candidate_index]
        lines = [
            str(self.tokenizer.mask_token),
            "Candidate value:",
            self._escape_marker(candidate.label),
            "Candidate criterion:",
            self._escape_marker(candidate.description),
        ]
        examples = getattr(primitive, "examples", None)
        if examples:
            candidate_examples = examples.get(candidate.value, ())
            if candidate_examples:
                lines.extend(
                    (
                        "Examples satisfying this candidate:",
                        *[f"- {self._escape_marker(example)}" for example in candidate_examples],
                    )
                )
        return "\n".join(lines)

    def _candidate_suffix(self, primitive: BasePrimitive, order: Sequence[int]) -> str:
        return "\n\n".join(
            self._candidate_block(primitive, candidate_index) for candidate_index in order
        )

    def _encode(
        self,
        contexts: Sequence[str],
        primitives: Sequence[BasePrimitive],
        option_orders: Sequence[Sequence[int]] | None = None,
    ) -> tuple[dict[str, Any], list[int], list[list[int]], list[list[int]], int]:
        orders = _candidate_orders(primitives, option_orders)
        prefixes: list[str] = []
        suffixes: list[str] = []
        row_fields: list[int] = []
        row_orders: list[list[int]] = []
        for context in contexts:
            for field_index, (primitive, order) in enumerate(zip(primitives, orders, strict=True)):
                prefixes.append(self._field_prompt(primitive, context))
                suffixes.append(self._candidate_suffix(primitive, order))
                row_fields.append(field_index)
                row_orders.append(order)

        suffix_lengths = [
            len(self.tokenizer("", suffix, add_special_tokens=True)["input_ids"])
            for suffix in suffixes[: len(primitives)]
        ]
        candidate_tokens = max(suffix_lengths)
        if candidate_tokens >= self.max_length:
            raise ValueError(
                "A schema candidate set does not fit in the encoder context window: "
                f"{candidate_tokens} tokens for max_length={self.max_length}."
            )

        encoded = self.tokenizer(
            prefixes,
            suffixes,
            padding=True,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        mask_token_id = int(self.tokenizer.mask_token_id)
        marker_positions: list[list[int]] = []
        for row_index, field_index in enumerate(row_fields):
            sequence_ids = encoded.sequence_ids(row_index)
            positions = [
                token_index
                for token_index, sequence_id in enumerate(sequence_ids)
                if sequence_id == 1
                and int(encoded["input_ids"][row_index, token_index]) == mask_token_id
            ]
            expected = len(primitives[field_index].candidates)
            if len(positions) != expected:
                raise ValueError(
                    f"Expected {expected} candidate markers for "
                    f"{_field_name(primitives[field_index])!r}, got {len(positions)}."
                )
            marker_positions.append(positions)

        input_ids = encoded["input_ids"].to(self.device)
        padding_mask = encoded["attention_mask"].to(self.device)
        sliding_window = getattr(self.model.config, "sliding_window", None)
        attention_masks, position_ids = _candidate_isolation(
            padding_mask,
            marker_positions,
            sliding_window=sliding_window,
        )
        return (
            {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "position_ids": position_ids,
            },
            row_fields,
            row_orders,
            marker_positions,
            candidate_tokens,
        )

    def score_batch(
        self,
        contexts: Sequence[str],
        primitives: Sequence[BasePrimitive],
        option_orders: Sequence[Sequence[int]] | None = None,
    ) -> tuple[list[torch.Tensor], int]:
        batch, row_fields, row_orders, marker_positions, candidate_tokens = self._encode(
            contexts,
            primitives,
            option_orders,
        )
        hidden = self.model(**batch).last_hidden_state
        logits_by_field: list[list[torch.Tensor]] = [[] for _ in primitives]
        for row_index, (field_index, order, positions) in enumerate(
            zip(row_fields, row_orders, marker_positions, strict=True)
        ):
            position_tensor = torch.tensor(positions, device=hidden.device)
            marker_hidden = hidden[row_index].index_select(0, position_tensor)
            permuted_logits = self.head(marker_hidden)
            inverse = torch.tensor(
                [order.index(stable_index) for stable_index in range(len(order))],
                device=hidden.device,
            )
            logits_by_field[field_index].append(permuted_logits.index_select(0, inverse))

        return [torch.stack(rows) for rows in logits_by_field], candidate_tokens

    def _apply_arbiter(
        self,
        *,
        context: str,
        primitives: Sequence[BasePrimitive],
        decisions: dict[str, DecisionResult],
        arbiter: BaseArbiter | None,
    ) -> tuple[list[str], float]:
        if arbiter is None:
            return [], 0.0
        fields = {
            _field_name(primitive): primitive
            for primitive in primitives
            if arbiter.resolve_all_fields or not decisions[_field_name(primitive)].is_certain
        }
        if not fields:
            return [], 0.0
        started = time.perf_counter()
        resolved = arbiter.resolve(
            context=context,
            ambiguous_fields=fields,
            current_decisions=decisions,
            engine=None,
            base_cache=None,
            past_len=0,
            threshold=0.30,
        )
        decisions.update(resolved)
        return list(resolved), round((time.perf_counter() - started) * 1000, 2)

    def evaluate(
        self,
        context: Any,
        schema: Sequence[BasePrimitive],
        *,
        arbiter: BaseArbiter | None = None,
        score_offsets: Mapping[str, Sequence[float]] | None = None,
        temperatures: Mapping[str, float] | None = None,
        allow_empty_context: bool = False,
    ) -> EvaluationResult:
        if isinstance(context, MultimodalContext):
            raise ValueError("Encoder runtimes currently support text contexts only.")
        if not isinstance(context, str) or (not context.strip() and not allow_empty_context):
            raise ValueError("Context must be a non-empty string.")
        validate_context_budget(context)
        return self.evaluate_batch(
            [context],
            schema,
            arbiter=arbiter,
            score_offsets=score_offsets,
            temperatures=temperatures,
        )[0]

    def evaluate_batch(
        self,
        contexts: Sequence[Any],
        schema: Sequence[BasePrimitive],
        *,
        arbiter: BaseArbiter | None = None,
        score_offsets: Mapping[str, Sequence[float]] | None = None,
        temperatures: Mapping[str, float] | None = None,
    ) -> list[EvaluationResult]:
        if not self.trained:
            raise ValueError("Encoder models must be fitted before inference.")
        if score_offsets:
            raise ValueError("Context-free score offsets are not supported by trained encoders.")
        if not contexts:
            return []
        if not schema:
            raise ValueError("Schema must contain at least one decision field.")
        text_contexts: list[str] = []
        for context in contexts:
            if isinstance(context, MultimodalContext):
                raise ValueError("Encoder runtimes currently support text contexts only.")
            if not isinstance(context, str) or not context.strip():
                raise ValueError("Context must be a non-empty string.")
            validate_context_budget(context)
            text_contexts.append(context)

        with self._lock, torch.inference_mode():
            started = time.perf_counter()
            field_logits, suffix_tokens = self.score_batch(text_contexts, schema)
            model_finished = time.perf_counter()

            results: list[EvaluationResult] = []
            for row_index, context in enumerate(text_contexts):
                decisions: dict[str, DecisionResult] = {}
                candidate_scores: dict[str, tuple[float, ...]] = {}
                for primitive, logits in zip(schema, field_logits):
                    field = _field_name(primitive)
                    row_logits = logits[row_index]
                    candidate_scores[field] = tuple(
                        row_logits.detach().to("cpu", torch.float32).tolist()
                    )
                    temperature = 1.0 if temperatures is None else temperatures.get(field, 1.0)
                    probabilities = torch.softmax(row_logits / max(1e-4, temperature), dim=-1)
                    decisions[field] = cast(Any, primitive).result_from_probabilities(probabilities)

                invoked, arbiter_ms = self._apply_arbiter(
                    context=context,
                    primitives=schema,
                    decisions=decisions,
                    arbiter=arbiter,
                )
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                results.append(
                    EvaluationResult(
                        decisions=decisions,
                        candidate_scores=candidate_scores,
                        telemetry={
                            "architecture": self.architecture,
                            "scorer": self.scorer,
                            "model_ms": round((model_finished - started) * 1000, 2),
                            "arbiter_ms": arbiter_ms,
                            "arbiter_invoked_fields": invoked,
                            "arbiter_name": arbiter.name if invoked and arbiter else None,
                            "total_ms": elapsed_ms,
                            "num_lanes": len(schema),
                            "batch_size": len(text_contexts),
                            "candidate_tokens": suffix_tokens,
                            "device": str(self.device),
                            "model_id": self.source,
                        },
                    )
                )

            total_ms = round((time.perf_counter() - started) * 1000, 2)
            throughput = round(len(results) / (total_ms / 1000), 1) if total_ms else 0.0
            ambiguous = sum(result.is_ambiguous for result in results)
            resolved = sum(bool(result.telemetry["arbiter_invoked_fields"]) for result in results)
            for result in results:
                result.telemetry.update(
                    {
                        "batch_total_ms": total_ms,
                        "batch_avg_ms": round(total_ms / len(results), 2),
                        "batch_throughput_docs_sec": throughput,
                        "batch_ambiguous_items": ambiguous,
                        "batch_arbiter_resolutions": resolved,
                    }
                )
            return results
