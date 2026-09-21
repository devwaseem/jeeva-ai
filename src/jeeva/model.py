from __future__ import annotations

import gc
import hashlib
import json
import platform
import random
import secrets
import shutil
import tempfile
import uuid
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import torch

from jeeva._json import MAX_JSON_BYTES, read_json, read_jsonl_record
from jeeva._version import __version__
from jeeva._weights import (
    require_local_adapter_safetensors,
    require_local_model_safetensors,
)
from jeeva.arbiters.base import BaseArbiter
from jeeva.calibration.threshold import QuantileNotchCalibrator
from jeeva.core.engine import EvaluationResult, get_default_device
from jeeva.core.prompts import validate_context_budget
from jeeva.data.dataset import JeevaDataset
from jeeva.heads.metric import DECODER_HEAD_DIR, BilinearMetricHead
from jeeva.runtimes import CausalRuntime, DecisionRuntime, EncoderRuntime
from jeeva.runtimes.encoder import save_candidate_marker_head
from jeeva.schema import CalibratedDecisionSchema  # type: ignore[attr-defined]
from jeeva.schema.decision import (
    SCHEMA_PROTOCOL,
    CompiledDecisionSchema,
    DecisionSchema,
    TrainingTargetValue,
    compile_decision_schema,
    validate_training_target,
)
from jeeva.schema.multimodal import MultimodalContext
from jeeva.schema.primitives import BasePrimitive
from jeeva.schema.targets import TargetDistribution
from jeeva.training.console import print_training_complete, print_training_header

Context = str | MultimodalContext
Scheduler = Literal["cosine", "linear", "constant"]
Architecture = Literal["causal", "encoder"]
Adaptation = Literal["auto", "head", "full", "lora", "qlora"]
DecoderHead = Literal["letter", "bilinear", "contextual"]

_MANIFEST_NAME = "jeeva.json"
_ARTIFACT_FORMAT = 11
_CONTEXT_FREE_POLICY = "context-free-v1"
_CONTEXT_FREE_NULL_INPUTS = ("", "N/A", "[MASK]")


def _distribution_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _artifact_provenance(*, weights: str) -> dict[str, Any]:
    versions = {
        "jeeva": __version__,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": _distribution_version("transformers"),
    }
    if weights == "adapter":
        versions["peft"] = _distribution_version("peft")
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
    }


_DecisionSchemaInput = type[DecisionSchema] | CalibratedDecisionSchema


class JeevaError(Exception):
    """Base exception for the public DecisionModel interface."""


class SchemaError(JeevaError):
    """Raised when a DecisionSchema is not a finite decision contract."""


class TrainingDataError(JeevaError):
    """Raised when training examples are empty, malformed, or incompatible."""


class ArtifactError(JeevaError):
    """Raised when a saved Jeeva artifact is incomplete or incompatible."""


@dataclass(frozen=True)
class Example:
    """One supervised context and its partial or complete decision target."""

    context: Context
    target: Mapping[str, TrainingTargetValue]
    group: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.context, str) and not self.context.strip():
            raise TrainingDataError("Example.context must be a non-empty string.")
        if not isinstance(self.context, (str, MultimodalContext)):
            raise TrainingDataError(
                "Example.context must be a string or MultimodalContext, "
                f"got {type(self.context).__name__}."
            )
        validate_context_budget(
            self.context.text if isinstance(self.context, MultimodalContext) else self.context
        )
        if not isinstance(self.target, Mapping) or not self.target:
            raise TrainingDataError("Example.target must be a non-empty decision mapping.")
        if self.group is not None and (not isinstance(self.group, str) or not self.group.strip()):
            raise TrainingDataError("Example.group must be a non-empty string when provided.")
        object.__setattr__(self, "target", MappingProxyType(dict(self.target)))
        if self.group is not None:
            object.__setattr__(self, "group", self.group.strip())


_CONTEXT_KEYS = ("context", "text", "document", "ticket", "input")


class _JsonlExampleSequence(Sequence[Example]):
    """Validated, random-access JSONL examples with bounded memory use."""

    def __init__(self, path: str | Path, schema: type[DecisionSchema]) -> None:
        self.path = Path(path)
        self.schema = schema
        if not self.path.is_file():
            raise TrainingDataError(f"JSONL dataset not found: {self.path}")

        self._offsets = array("Q")
        self._line_numbers = array("Q")
        with self.path.open("rb") as source:
            line_number = 0
            while True:
                offset = source.tell()
                try:
                    raw_line = read_jsonl_record(source)
                except ValueError as exc:
                    raise TrainingDataError(f"{self.path}:{line_number + 1}: {exc}") from exc
                if not raw_line:
                    break
                line_number += 1
                if not raw_line.strip():
                    continue
                self._decode(raw_line, line_number)
                self._offsets.append(offset)
                self._line_numbers.append(line_number)
        if not self._offsets:
            raise TrainingDataError(f"JSONL dataset contains no records: {self.path}")
        self._source = self.path.open("rb")

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> Example:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        self._source.seek(self._offsets[index])
        try:
            raw_line = read_jsonl_record(self._source)
        except ValueError as exc:
            raise TrainingDataError(f"{self.path}:{self._line_numbers[index]}: {exc}") from exc
        return self._decode(raw_line, self._line_numbers[index])

    def close(self) -> None:
        source = getattr(self, "_source", None)
        if source is not None:
            source.close()

    def __del__(self) -> None:
        self.close()

    def _decode(self, raw_line: bytes, line_number: int) -> Example:
        location = f"{self.path}:{line_number}"
        try:
            record = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TrainingDataError(f"{location}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise TrainingDataError(f"{location}: each JSONL record must be an object.")
        context = next(
            (
                record[key]
                for key in _CONTEXT_KEYS
                if isinstance(record.get(key), str) and record[key].strip()
            ),
            None,
        )
        if context is None:
            raise TrainingDataError(f"{location}: record requires a non-empty 'context' string.")
        if "target" in record:
            target_data = record["target"]
            if not isinstance(target_data, dict):
                raise TrainingDataError(f"{location}: 'target' must be a JSON object.")
        else:
            target_data = {
                field_name: record[field_name]
                for field_name in self.schema.fields()
                if field_name in record
            }
        try:
            return Example(
                context=context,
                target=validate_training_target(self.schema, target_data),
                group=record.get("group"),
            )
        except (TypeError, ValueError, TrainingDataError) as exc:
            raise TrainingDataError(
                f"{location}: target does not match {self.schema.__name__}: {exc}"
            ) from exc


class _IndexedExampleSequence(Sequence[Example]):
    def __init__(self, source: Sequence[Example], indices: array[int]) -> None:
        self.source = source
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Example:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.source[self.indices[index]]


class _TrainingRecordSequence(Sequence[dict[str, Any]]):
    def __init__(self, examples: Sequence[Example]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {"context": example.context, **example.target}


@dataclass(frozen=True)
class FitConfig:
    """Architecture-independent controls for fitting one decision schema."""

    epochs: int = 3
    progress_updates_per_epoch: int = 10
    learning_rate: float | None = None
    head_learning_rate: float | None = None
    decoder_head: DecoderHead = "letter"
    scheduler: Scheduler = "cosine"
    minimum_learning_rate_ratio: float = 0.05
    weight_decay: float = 0.01
    maximum_gradient_norm: float = 1.0
    adaptation: Adaptation = "auto"
    lora_rank: int = 8
    lora_alpha: int | None = None
    lora_dropout: float = 0.05
    batch_size: int = 8
    max_length: int = 512
    brier_weight: float = 0.5
    ordinal_weight: float = 0.25
    save_best: bool = True
    early_stopping_patience: int | None = 2
    validation_fraction: float = 0.2
    seed: int = field(default_factory=lambda: secrets.randbelow(2**32))
    target_precision: float = 0.95
    minimum_calibration_examples: int = 5

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.progress_updates_per_epoch < 0:
            raise ValueError("progress_updates_per_epoch cannot be negative.")
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive when provided.")
        if self.head_learning_rate is not None and self.head_learning_rate <= 0:
            raise ValueError("head_learning_rate must be positive when provided.")
        if self.decoder_head not in ("letter", "bilinear", "contextual"):
            raise ValueError("decoder_head must be 'letter', 'bilinear', or 'contextual'.")
        if self.scheduler not in ("cosine", "linear", "constant"):
            raise ValueError("scheduler must be 'cosine', 'linear', or 'constant'.")
        if not 0.0 <= self.minimum_learning_rate_ratio <= 1.0:
            raise ValueError("minimum_learning_rate_ratio must be in [0, 1].")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.maximum_gradient_norm < 0:
            raise ValueError("maximum_gradient_norm cannot be negative.")
        if self.adaptation not in ("auto", "head", "full", "lora", "qlora"):
            raise ValueError("adaptation must be 'auto', 'head', 'full', 'lora', or 'qlora'.")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be at least 1.")
        if self.lora_alpha is not None and self.lora_alpha < 1:
            raise ValueError("lora_alpha must be at least 1 when provided.")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1).")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.max_length < 32:
            raise ValueError("max_length must be at least 32.")
        if self.brier_weight < 0.0 or self.ordinal_weight < 0.0:
            raise ValueError("Loss weights cannot be negative.")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1 when provided.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        if not 0.0 < self.target_precision <= 1.0:
            raise ValueError("target_precision must be in (0, 1].")
        if self.minimum_calibration_examples < 1:
            raise ValueError("minimum_calibration_examples must be at least 1.")

    @property
    def effective_lora_alpha(self) -> int:
        return self.lora_alpha if self.lora_alpha is not None else 2 * self.lora_rank


@dataclass(frozen=True)
class FieldDecision:
    """One field's value and observable scoring evidence."""

    value: bool | str | float
    distribution: Mapping[str, float]
    margin: float
    certainty_score: float
    score: float
    entropy: float | None
    normalized_entropy: float | None
    winning_level: float | None
    standard_deviation: float | None
    is_certain: bool
    calibration: Literal["validated", "unvalidated"]
    tau: float
    threshold_source: Literal["manual", "calibrated", "default"]
    resolved_by: str
    reasoning: str | None


@dataclass(frozen=True)
class Decision:
    """A dictionary decision plus per-field scoring evidence."""

    value: Mapping[str, bool | str | float]
    fields: Mapping[str, FieldDecision]
    is_certain: bool
    calibration: Literal["validated", "partial", "unvalidated"]
    telemetry: Mapping[str, Any]

    @property
    def is_ambiguous(self) -> bool:
        return not self.is_certain


class DecisionModel:
    """One model owner for dictionary decisions, fitting, and artifact persistence."""

    def __init__(
        self,
        source: str | Path,
        *,
        architecture: Architecture | None = None,
        device: str = "auto",
        dtype: str = "auto",
        candidate_position_balancing: bool | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        if architecture not in (None, "causal", "encoder"):
            raise ValueError("architecture must be 'causal', 'encoder', or None.")
        for name, value in {
            "revision": revision,
            "tokenizer_revision": tokenizer_revision,
        }.items():
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when provided.")
        self.source = str(source)
        self.device = get_default_device() if device == "auto" else device
        self.dtype = dtype
        self._artifact_path: Path | None = None
        self._temporary_artifact: tempfile.TemporaryDirectory[str] | None = None
        self._training_report: Mapping[str, Any] | None = None
        self._context_free_calibration_owner = object()
        self._architecture: Architecture
        self.revision = revision
        self.tokenizer_revision = revision if tokenizer_revision is None else tokenizer_revision
        self.local_files_only = bool(local_files_only)
        self._source_metadata: dict[str, Any]

        source_path = Path(source)
        manifest_path = source_path / _MANIFEST_NAME
        if source_path.is_dir() and manifest_path.exists():
            self._artifact_path = source_path
            self._manifest = self._load_manifest(manifest_path)
            if revision is not None or tokenizer_revision is not None:
                raise ArtifactError(
                    "Saved artifacts own their resolved model and tokenizer revisions; "
                    "do not pass revision overrides when loading an artifact."
                )
            source_spec = cast(Mapping[str, Any], self._manifest["source"])
            model_source = cast(Mapping[str, Any], source_spec["model"])
            tokenizer_source = cast(Mapping[str, Any], source_spec["tokenizer"])
            self._source_metadata = {
                "model": dict(model_source),
                "tokenizer": dict(tokenizer_source),
                "local_files_only": bool(source_spec["local_files_only"]),
            }
            self.revision = cast(
                str | None,
                model_source["resolved_revision"] or model_source["requested_revision"],
            )
            self.tokenizer_revision = cast(
                str | None,
                tokenizer_source["resolved_revision"] or tokenizer_source["requested_revision"],
            )
            if local_files_only is None:
                self.local_files_only = bool(source_spec["local_files_only"])
            runtime_spec = cast(Mapping[str, Any], self._manifest["runtime"])
            declared_architecture = cast(Architecture, runtime_spec["architecture"])
            if architecture is not None and architecture != declared_architecture:
                raise ArtifactError(
                    f"Artifact architecture is {declared_architecture!r}, not {architecture!r}."
                )
            self._architecture = declared_architecture
            self._scorer = str(runtime_spec["scorer"])
            self._weights = str(runtime_spec["weights"])
            self._max_length = int(runtime_spec.get("max_length", 512))
            self._artifact_kind = str(self._manifest["kind"])
            self._base_model = str(self._manifest["base_model"])
            self._trained = self._artifact_kind in ("trained", "merged")
            self._calibration = dict(self._manifest.get("calibration", {}))
            self._training_report = self._manifest.get("training_report")

            if self._artifact_kind == "zero_shot":
                if self._architecture != "causal":
                    raise ArtifactError("Encoder artifacts cannot declare kind='zero_shot'.")
                runtime_source = self._base_model
            elif self._artifact_kind in ("trained", "merged"):
                self._validate_artifact_files(source_path, self._manifest)
                runtime_source = str(source_path)
            else:
                raise ArtifactError(f"Artifact declares unsupported kind {self._artifact_kind!r}.")
        else:
            self._architecture = cast(Architecture, architecture or "causal")
            self._base_model = self.source
            self._source_metadata = {
                "model": {
                    "requested_id": self.source,
                    "requested_revision": self.revision,
                    "resolved_id": self.source,
                    "resolved_revision": self.revision,
                },
                "tokenizer": {
                    "requested_id": self.source,
                    "requested_revision": self.tokenizer_revision,
                    "resolved_id": self.source,
                    "resolved_revision": self.tokenizer_revision,
                },
                "local_files_only": self.local_files_only,
            }
            self._calibration: dict[str, dict[str, dict[str, Any]]] = {}
            self._trained = False
            self._max_length = 512
            if self._architecture == "causal":
                self._artifact_kind = "zero_shot"
                self._scorer = "letter"
                self._weights = "base"
                self._manifest = self._new_manifest(
                    kind="zero_shot",
                    architecture="causal",
                    scorer="letter",
                    weights="base",
                )
            else:
                self._artifact_kind = "untrained"
                self._scorer = "candidate_marker"
                self._weights = "base"
                self._manifest = self._new_manifest(
                    kind="untrained",
                    architecture="encoder",
                    scorer="candidate_marker",
                    weights="base",
                    max_length=self._max_length,
                )
            runtime_source = self.source

        supports_position_balancing = self._architecture == "causal" and self._scorer == "letter"
        if candidate_position_balancing is True and not supports_position_balancing:
            raise ValueError("candidate_position_balancing requires decoder_head='letter'.")
        self.candidate_position_balancing = (
            supports_position_balancing
            if candidate_position_balancing is None
            else candidate_position_balancing
        )

        self._runtime_source = runtime_source
        self._runtime: DecisionRuntime | None = self._load_runtime()
        if self._artifact_path is None:
            self._capture_source_metadata()

    @property
    def training_report(self) -> Mapping[str, Any] | None:
        return self._training_report

    @property
    def architecture(self) -> Architecture:
        return self._architecture

    def _capture_source_metadata(self) -> None:
        runtime = self._runtime_instance()
        model_source = cast(dict[str, Any], self._source_metadata["model"])
        tokenizer_source = cast(dict[str, Any], self._source_metadata["tokenizer"])
        resolved_model_id = getattr(runtime, "resolved_model_id", None)
        resolved_model_revision = getattr(runtime, "resolved_model_revision", None)
        resolved_tokenizer_id = getattr(runtime, "resolved_tokenizer_id", None)
        resolved_tokenizer_revision = getattr(runtime, "resolved_tokenizer_revision", None)
        model_source["resolved_id"] = (
            resolved_model_id
            if isinstance(resolved_model_id, str) and resolved_model_id
            else self.source
        )
        model_source["resolved_revision"] = (
            resolved_model_revision
            if isinstance(resolved_model_revision, str) and resolved_model_revision
            else self.revision
        )
        tokenizer_source["resolved_id"] = (
            resolved_tokenizer_id
            if isinstance(resolved_tokenizer_id, str) and resolved_tokenizer_id
            else self.source
        )
        tokenizer_source["resolved_revision"] = (
            resolved_tokenizer_revision
            if isinstance(resolved_tokenizer_revision, str) and resolved_tokenizer_revision
            else self.tokenizer_revision
        )
        self._manifest["source"] = {
            "model": dict(model_source),
            "tokenizer": dict(tokenizer_source),
            "local_files_only": self.local_files_only,
        }

    def _load_runtime(self) -> DecisionRuntime:
        if self._architecture == "causal":
            return CausalRuntime(
                self._runtime_source,
                device=self.device,
                dtype=self.dtype,
                scorer=self._scorer,
                candidate_position_balancing=self.candidate_position_balancing,
                revision=self.revision,
                tokenizer_revision=self.tokenizer_revision,
                local_files_only=self.local_files_only,
            )
        try:
            return EncoderRuntime(
                self._runtime_source,
                device=self.device,
                dtype=self.dtype,
                max_length=self._max_length,
                trained=self._trained,
                revision=self.revision,
                tokenizer_revision=self.tokenizer_revision,
                local_files_only=self.local_files_only,
            )
        except (OSError, ValueError) as exc:
            if self._artifact_path is not None:
                raise ArtifactError(str(exc)) from exc
            raise

    def _runtime_instance(self) -> DecisionRuntime:
        if self._runtime is None:
            self._runtime = self._load_runtime()
        return self._runtime

    def _release_runtime(self) -> None:
        """Release the eager inference model before memory-constrained QLoRA fitting."""
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            del runtime
        gc.collect()
        device_type = torch.device(self.device).type
        if device_type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def warmup(self) -> None:
        """Initialize model execution paths before serving latency-sensitive decisions."""
        self._require_ready()
        self._runtime_instance().warmup()

    def decide(
        self,
        context: Context,
        schema: _DecisionSchemaInput,
        *,
        arbiter: BaseArbiter | None = None,
    ) -> Decision:
        self._require_ready()
        self._validate_arbiter(arbiter)
        compiled, score_offsets = self._resolve_decision_schema(schema)
        self._require_trained_schema(compiled)
        primitives = self._primitives_with_calibration(compiled)
        result = self._runtime_instance().evaluate(
            context,
            primitives,
            arbiter=arbiter,
            score_offsets=score_offsets,
            temperatures=self._field_temperatures(compiled),
        )
        return self._to_decision(compiled, result)

    def decide_many(
        self,
        contexts: Sequence[Context],
        schema: _DecisionSchemaInput,
        *,
        arbiter: BaseArbiter | None = None,
    ) -> list[Decision]:
        self._require_ready()
        self._validate_arbiter(arbiter)
        compiled, score_offsets = self._resolve_decision_schema(schema)
        self._require_trained_schema(compiled)
        primitives = self._primitives_with_calibration(compiled)
        results = self._runtime_instance().evaluate_batch(
            contexts,
            primitives,
            arbiter=arbiter,
            score_offsets=score_offsets,
            temperatures=self._field_temperatures(compiled),
        )
        return [self._to_decision(compiled, result) for result in results]

    def fit(
        self,
        examples: Iterable[Example] | str | Path,
        *,
        schema: type[DecisionSchema],
        validation: Iterable[Example] | str | Path | None = None,
        config: FitConfig | None = None,
    ) -> DecisionModel:
        """Fit one DecisionSchema in place and return this model."""
        active_config = config or FitConfig()
        if self._architecture == "encoder" and active_config.decoder_head != "letter":
            raise ValueError(
                "decoder_head applies only to causal models; encoders use candidate_marker."
            )
        adaptation: Literal["head", "full", "lora", "qlora"]
        if self._architecture == "causal":
            if active_config.adaptation == "head":
                raise ValueError("adaptation='head' is supported only by encoder models.")
            if active_config.adaptation == "auto":
                adaptation = "qlora" if self._artifact_adaptation() == "qlora" else "lora"
            else:
                adaptation = cast(
                    Literal["full", "lora", "qlora"],
                    active_config.adaptation,
                )
            scorer = active_config.decoder_head
            training_dtype = (
                "float16" if adaptation == "qlora" and self.dtype == "auto" else self.dtype
            )
            if adaptation == "qlora" and torch.device(self.device).type not in {"mps", "cuda"}:
                raise ValueError("QLoRA training requires an MPS or CUDA device.")
        else:
            if active_config.adaptation == "qlora":
                raise ValueError("adaptation='qlora' is supported only by causal text models.")
            adaptation = (
                "lora"
                if active_config.adaptation == "auto"
                else cast(
                    Literal["head", "full", "lora"],
                    active_config.adaptation,
                )
            )
            scorer = "candidate_marker"
            training_dtype = (
                "float32" if self.dtype == "auto" and self.device == "mps" else self.dtype
            )
        if isinstance(schema, CalibratedDecisionSchema):
            raise SchemaError(
                "fit() requires a raw DecisionSchema; context-free calibrated schemas "
                "are only for zero-shot inference."
            )
        compiled = self._compile_schema(schema)
        if isinstance(examples, (str, Path)):
            source_examples: Sequence[Example] = _JsonlExampleSequence(examples, schema)
            if validation is None:
                train_examples, validation_examples = self._split_indexed_examples(
                    source_examples,
                    fraction=active_config.validation_fraction,
                    seed=active_config.seed,
                )
            else:
                train_examples = source_examples
                validation_examples = self._load_validation(validation, schema)
        else:
            loaded_examples = list(examples)
            if not loaded_examples:
                raise TrainingDataError("fit() requires at least one training example.")
            self._validate_examples(loaded_examples, schema)
            if validation is None:
                train_examples, validation_examples = self._split_examples(
                    loaded_examples,
                    fraction=active_config.validation_fraction,
                    seed=active_config.seed,
                )
            else:
                train_examples = loaded_examples
                validation_examples = self._load_validation(validation, schema)

        primitives = list(compiled.primitives.values())
        train_records = _TrainingRecordSequence(train_examples)
        validation_records = _TrainingRecordSequence(validation_examples)
        train_dataset = JeevaDataset(train_records, primitives)
        validation_dataset = (
            JeevaDataset(validation_records, primitives) if len(validation_records) else None
        )

        if adaptation == "head":
            learning_rate = 0.0
        elif active_config.learning_rate is not None:
            learning_rate = active_config.learning_rate
        else:
            learning_rate = 2e-4 if adaptation in ("lora", "qlora") else 2e-5

        training_source = (
            str(examples) if isinstance(examples, (str, Path)) else "In-memory iterable"
        )
        if isinstance(validation, (str, Path)):
            validation_source = str(validation)
        elif validation is None:
            validation_source = f"Automatic split ({active_config.validation_fraction:.0%})"
        else:
            validation_source = "In-memory iterable"

        adaptation_name = {
            "head": "Frozen encoder + trainable head",
            "lora": "LoRA",
            "qlora": "QLoRA (4-bit NF4)",
            "full": "Full fine-tuning",
        }[adaptation]
        backbone_learning_rate: str | float = (
            "Frozen" if adaptation == "head" else f"{learning_rate:.2e}"
        )
        optimization_rows: list[tuple[str, object]] = [
            ("Adaptation", adaptation_name),
            ("Epochs", active_config.epochs),
            ("Backbone learning rate", backbone_learning_rate),
            (
                "Scheduler",
                f"{active_config.scheduler} "
                f"(minimum {active_config.minimum_learning_rate_ratio:.0%})",
            ),
            ("Weight decay", f"{active_config.weight_decay:g}"),
            ("Gradient clipping", f"{active_config.maximum_gradient_norm:g}"),
        ]
        if self._architecture == "encoder":
            optimization_rows.extend(
                [
                    ("Batch size", active_config.batch_size),
                    ("Maximum length", f"{active_config.max_length} tokens"),
                    (
                        "Head learning rate",
                        f"{(active_config.head_learning_rate or 1e-3):.2e}",
                    ),
                ]
            )
        else:
            optimization_rows.extend(
                [
                    ("Batch size", "1 sample per step"),
                    ("Prefill gradients", "Enabled"),
                ]
            )
        if adaptation in ("lora", "qlora"):
            optimization_rows.extend(
                [
                    ("LoRA rank", active_config.lora_rank),
                    ("LoRA alpha", active_config.effective_lora_alpha),
                    ("LoRA dropout", f"{active_config.lora_dropout:g}"),
                ]
            )
        if adaptation == "qlora":
            optimization_rows.extend(
                [
                    ("Base quantization", "4-bit NF4 with double quantization"),
                    ("Quantized compute", training_dtype),
                ]
            )
        optimization_rows.extend(
            [
                ("Seed", active_config.seed),
                (
                    "Progress updates",
                    f"up to {active_config.progress_updates_per_epoch} per epoch",
                ),
                ("Best checkpoint", "Enabled" if active_config.save_best else "Disabled"),
            ]
        )
        print_training_header(
            [
                (
                    "Model",
                    [
                        (
                            "Architecture",
                            "Causal decoder"
                            if self._architecture == "causal"
                            else "Bidirectional encoder",
                        ),
                        ("Backbone", self._training_source()),
                        ("Decision schema", schema.__name__),
                        ("Fields", ", ".join(compiled.primitives)),
                        (
                            "Scorer",
                            {
                                "letter": "Letter readout through model output head",
                                "bilinear": "Bilinear metric head",
                                "contextual": "Contextual bilinear metric head",
                                "candidate_marker": "Candidate option marker head",
                            }[scorer],
                        ),
                        ("Device", self.device),
                        ("Precision", training_dtype),
                    ],
                ),
                (
                    "Data",
                    [
                        ("Training source", training_source),
                        ("Training examples", len(train_examples)),
                        ("Validation source", validation_source),
                        ("Validation examples", len(validation_examples)),
                    ],
                ),
                ("Optimization", optimization_rows),
                (
                    "Decision objective",
                    [
                        ("Brier weight", f"{active_config.brier_weight:g}"),
                        ("Ordinal weight", f"{active_config.ordinal_weight:g}"),
                        ("Target precision", f"{active_config.target_precision:.0%}"),
                    ],
                ),
            ]
        )

        temporary_artifact = tempfile.TemporaryDirectory(prefix="jeeva-fit-")
        output_dir = temporary_artifact.name
        trained_runtime: DecisionRuntime | None = None
        if self._architecture == "causal":
            from jeeva.training.trainer import Trainer, TrainingConfig

            if adaptation == "head":
                raise RuntimeError("Internal error: head-only adaptation reached causal training.")

            trainer_source: Any = self._training_source()
            trainer_components: dict[str, Any] = {}
            causal_runtime: CausalRuntime | None = None
            if adaptation == "qlora":
                self._release_runtime()
            else:
                causal_runtime = cast(CausalRuntime, self._runtime_instance())
                trainer_source = causal_runtime.model
                trainer_components = {
                    "tokenizer": causal_runtime.tokenizer,
                    "processor": causal_runtime.processor,
                    "head": (
                        causal_runtime.engine.head
                        if isinstance(causal_runtime.engine.head, BilinearMetricHead)
                        else None
                    ),
                }

            trainer = Trainer(
                model_or_id=trainer_source,
                schema=primitives,
                config=TrainingConfig(
                    epochs=active_config.epochs,
                    progress_updates_per_epoch=active_config.progress_updates_per_epoch,
                    seed=active_config.seed,
                    lr=learning_rate,
                    weight_decay=active_config.weight_decay,
                    max_grad_norm=active_config.maximum_gradient_norm,
                    adaptation=adaptation,
                    lora_r=active_config.lora_rank,
                    lora_alpha=active_config.effective_lora_alpha,
                    lora_dropout=active_config.lora_dropout,
                    decoder_head=active_config.decoder_head,
                    head_lr=active_config.head_learning_rate or 1e-3,
                    candidate_position_balancing=active_config.decoder_head == "letter",
                    output_dir=output_dir,
                    lr_scheduler=active_config.scheduler,
                    min_lr_ratio=active_config.minimum_learning_rate_ratio,
                    save_best=active_config.save_best,
                    brier_weight=active_config.brier_weight,
                    ordinal_weight=active_config.ordinal_weight,
                    device=self.device,
                    dtype=training_dtype,
                    revision=cast(
                        str | None,
                        cast(Mapping[str, Any], self._source_metadata["model"])[
                            "resolved_revision"
                        ],
                    ),
                    tokenizer_revision=cast(
                        str | None,
                        cast(Mapping[str, Any], self._source_metadata["tokenizer"])[
                            "resolved_revision"
                        ],
                    ),
                    local_files_only=self.local_files_only,
                ),
                **trainer_components,
            )
            report = trainer.train(train_dataset, eval_dataset=validation_dataset)
            if adaptation == "qlora":
                del trainer
                self._release_runtime()
            else:
                if causal_runtime is None:
                    raise RuntimeError("Causal runtime was released unexpectedly.")
                trained_runtime = CausalRuntime.from_components(
                    model=trainer.model,
                    tokenizer=trainer.tokenizer,
                    processor=trainer.processor,
                    source=output_dir,
                    device=self.device,
                    dtype=trainer.torch_dtype,
                    scorer=scorer,
                    head=trainer.head,
                    candidate_position_balancing=trainer.config.candidate_position_balancing,
                    resolved_model_id=causal_runtime.resolved_model_id,
                    resolved_model_revision=causal_runtime.resolved_model_revision,
                    resolved_tokenizer_id=causal_runtime.resolved_tokenizer_id,
                    resolved_tokenizer_revision=causal_runtime.resolved_tokenizer_revision,
                )
        else:
            from jeeva.training.encoder import EncoderTrainer, EncoderTrainingConfig

            current_encoder_runtime = cast(EncoderRuntime, self._runtime_instance())
            encoder_trainer = EncoderTrainer(
                current_encoder_runtime,
                primitives,
                EncoderTrainingConfig(
                    epochs=active_config.epochs,
                    learning_rate=learning_rate,
                    head_learning_rate=active_config.head_learning_rate,
                    weight_decay=active_config.weight_decay,
                    maximum_gradient_norm=active_config.maximum_gradient_norm,
                    scheduler=active_config.scheduler,
                    minimum_learning_rate_ratio=active_config.minimum_learning_rate_ratio,
                    adaptation=cast(Literal["head", "full", "lora"], adaptation),
                    lora_rank=active_config.lora_rank,
                    lora_alpha=active_config.effective_lora_alpha,
                    lora_dropout=active_config.lora_dropout,
                    batch_size=active_config.batch_size,
                    max_length=active_config.max_length,
                    brier_weight=active_config.brier_weight,
                    ordinal_weight=active_config.ordinal_weight,
                    save_best=active_config.save_best,
                    early_stopping_patience=active_config.early_stopping_patience,
                    seed=active_config.seed,
                    progress_updates_per_epoch=active_config.progress_updates_per_epoch,
                    output_dir=output_dir,
                    device=self.device,
                    dtype=training_dtype,
                    revision=cast(
                        str | None,
                        cast(Mapping[str, Any], self._source_metadata["model"])[
                            "resolved_revision"
                        ],
                    ),
                    tokenizer_revision=cast(
                        str | None,
                        cast(Mapping[str, Any], self._source_metadata["tokenizer"])[
                            "resolved_revision"
                        ],
                    ),
                    local_files_only=self.local_files_only,
                ),
                trained_source=self._trained,
            )
            report = encoder_trainer.train(train_dataset, validation_dataset)
            trained_runtime = EncoderRuntime.from_components(
                model=encoder_trainer.runtime.model,
                tokenizer=encoder_trainer.runtime.tokenizer,
                head=encoder_trainer.runtime.head,
                device=self.device,
                max_length=active_config.max_length,
                resolved_model_id=current_encoder_runtime.resolved_model_id,
                resolved_model_revision=current_encoder_runtime.resolved_model_revision,
                resolved_tokenizer_id=current_encoder_runtime.resolved_tokenizer_id,
                resolved_tokenizer_revision=current_encoder_runtime.resolved_tokenizer_revision,
            )

        hard_target_fields, soft_target_fields = self._target_counts(train_examples)
        report = dict(report)
        report["hard_target_fields"] = hard_target_fields
        report["soft_target_fields"] = soft_target_fields
        report["run_id"] = str(uuid.uuid4())
        report["created_at"] = datetime.now(timezone.utc).isoformat()
        report["dataset_digest"] = self._dataset_digest(
            train_examples,
            validation_examples,
        )
        report["schema_fingerprint"] = compiled.fingerprint
        resolved_training_config = {
            **asdict(active_config),
            "architecture": self._architecture,
            "scorer": scorer,
            "adaptation": adaptation,
            "backbone_learning_rate": None if adaptation == "head" else learning_rate,
            "head_learning_rate": (
                active_config.head_learning_rate or 1e-3
                if self._architecture == "encoder" or scorer in ("bilinear", "contextual")
                else None
            ),
            "training_dtype": training_dtype,
            "training_examples": len(train_examples),
            "validation_examples": len(validation_examples),
        }
        if "quantization" in report:
            resolved_training_config["quantization"] = report["quantization"]
        report["resolved_training_config"] = resolved_training_config

        weights = "adapter" if report["adaptation"] in ("lora", "qlora") else "full"
        manifest = self._new_manifest(
            kind="trained",
            architecture=self._architecture,
            scorer=scorer,
            weights=weights,
            max_length=active_config.max_length if self._architecture == "encoder" else None,
        )
        manifest["fit_config"] = {
            **asdict(active_config),
            "adaptation": adaptation,
        }
        manifest["schemas"] = {compiled.fingerprint: dict(compiled.descriptor)}
        manifest["training_report"] = report
        manifest["calibration"] = {}
        if adaptation == "qlora":
            quantization = cast(Mapping[str, Any], report["quantization"])
            provenance = cast(dict[str, Any], manifest["provenance"])
            versions = cast(dict[str, str], provenance["versions"])
            versions["bitsandbytes"] = str(quantization["backend_version"])
        self._write_manifest(Path(output_dir), manifest)

        if trained_runtime is None:
            reloaded = DecisionModel(output_dir, device=self.device, dtype=self.dtype)
            trained_runtime = reloaded._runtime_instance()
            self._source_metadata = reloaded._source_metadata
            self.revision = reloaded.revision
            self.tokenizer_revision = reloaded.tokenizer_revision
            self.local_files_only = reloaded.local_files_only
            reloaded._runtime = None

        trained = self
        trained.source = output_dir
        trained._runtime_source = output_dir
        trained._runtime = trained_runtime
        trained._artifact_path = Path(output_dir)
        trained._artifact_kind = "trained"
        trained._trained = True
        trained._scorer = scorer
        trained._weights = weights
        trained._max_length = active_config.max_length if self._architecture == "encoder" else 512
        trained._manifest = manifest
        trained.candidate_position_balancing = self._architecture == "causal" and scorer == "letter"
        trained._temporary_artifact = temporary_artifact
        trained._training_report = report
        trained._calibration = (
            trained._calibrate_encoder(validation_examples, compiled, active_config)
            if self._architecture == "encoder"
            else trained._calibrate(validation_examples, compiled, active_config)
        )
        trained._manifest["calibration"] = trained._calibration
        trained._write_manifest(Path(output_dir), trained._manifest)
        print_training_complete(report)
        return trained

    def save(self, destination: str | Path, *, merge: bool = False) -> Path:
        """Persist a self-describing causal or encoder decision artifact."""
        if self._architecture == "encoder" and not self._trained:
            raise ArtifactError("Raw encoder models must be fitted before they can be saved.")
        if merge and (not self._trained or self._weights != "adapter"):
            raise ArtifactError("merge=True requires a fitted LoRA adapter artifact.")

        destination_path = Path(destination)
        if destination_path.exists():
            raise ArtifactError(f"Artifact destination already exists: {destination_path}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path = Path(
            tempfile.mkdtemp(
                prefix=f".{destination_path.name}.staging-",
                dir=destination_path.parent,
            )
        )
        committed = False
        try:
            self._save_artifact_contents(staging_path, merge=merge)
            staged_manifest = self._load_manifest(staging_path / _MANIFEST_NAME)
            self._validate_artifact_files(staging_path, staged_manifest)
            if destination_path.exists():
                raise ArtifactError(
                    f"Artifact destination appeared during save: {destination_path}"
                )
            staging_path.rename(destination_path)
            committed = True
        finally:
            if not committed:
                shutil.rmtree(staging_path, ignore_errors=True)
        return destination_path

    def _save_artifact_contents(self, destination: Path, *, merge: bool) -> None:
        if self._trained:
            runtime = self._runtime_instance()
            if merge:
                self._save_merged_weights(destination)
            else:
                runtime.model.save_pretrained(str(destination), safe_serialization=True)
            runtime.tokenizer.save_pretrained(str(destination))
            if runtime.processor is not None:
                runtime.processor.save_pretrained(str(destination))
            if self._architecture == "encoder":
                encoder_runtime = cast(EncoderRuntime, runtime)
                save_candidate_marker_head(encoder_runtime.head, destination)
            else:
                causal_runtime = cast(CausalRuntime, runtime)
                causal_runtime.save_decoder_head(destination)

        manifest = dict(self._manifest)
        runtime_spec = dict(cast(Mapping[str, Any], manifest["runtime"]))
        if merge:
            manifest["kind"] = "merged"
            runtime_spec["weights"] = "full"
        manifest["runtime"] = runtime_spec
        manifest["calibration"] = self._calibration
        manifest["training_report"] = self._training_report
        self._write_manifest(destination, manifest)

    def _save_merged_weights(self, destination: Path) -> None:
        try:
            from peft import PeftModel, get_base_model_state_dict
        except ImportError as exc:
            raise ArtifactError(
                "Merged export requires the training extra: pip install 'jeeva-ai[training]'"
            ) from exc

        model = self._runtime_instance().model
        if not isinstance(model, PeftModel):
            raise ArtifactError("The loaded trained artifact does not contain a mergeable adapter.")

        merged = False
        try:
            model.merge_adapter(safe_merge=True)
            merged = True
            state_dict = get_base_model_state_dict(model)
            base_model: Any = model.get_base_model()
            base_model.save_pretrained(
                str(destination),
                state_dict=state_dict,
                safe_serialization=True,
            )
        except Exception as exc:
            raise ArtifactError(f"Could not merge and save the LoRA adapter: {exc}") from exc
        finally:
            if merged:
                model.unmerge_adapter()

    def _validate_artifact_files(
        self,
        source_path: Path,
        manifest: Mapping[str, Any],
    ) -> None:
        if manifest["kind"] == "zero_shot":
            return
        runtime = cast(Mapping[str, Any], manifest["runtime"])
        weights = runtime["weights"]
        if weights == "adapter":
            if not (source_path / "adapter_config.json").is_file():
                raise ArtifactError(
                    f"Adapter artifact at {source_path} is missing adapter_config.json."
                )
            try:
                require_local_adapter_safetensors(source_path)
            except ValueError as exc:
                raise ArtifactError(str(exc)) from exc
        if weights == "full":
            if not (source_path / "config.json").is_file():
                raise ArtifactError(
                    f"Full-weight artifact at {source_path} is missing config.json."
                )
            try:
                require_local_model_safetensors(source_path)
            except ValueError as exc:
                raise ArtifactError(str(exc)) from exc
        if runtime["architecture"] == "causal" and runtime["scorer"] in (
            "bilinear",
            "contextual",
        ):
            head_path = source_path / DECODER_HEAD_DIR
            missing = [
                name
                for name in ("config.json", "model.safetensors")
                if not (head_path / name).is_file()
            ]
            if missing:
                raise ArtifactError(
                    f"Metric artifact at {source_path} is missing "
                    f"{DECODER_HEAD_DIR}/{', '.join(missing)}."
                )

    def _compile_schema(self, schema: type[DecisionSchema]) -> CompiledDecisionSchema:
        try:
            return compile_decision_schema(schema)
        except (TypeError, ValueError) as exc:
            raise SchemaError(str(exc)) from exc

    def _resolve_decision_schema(
        self,
        schema: _DecisionSchemaInput,
    ) -> tuple[CompiledDecisionSchema, Mapping[str, tuple[float, ...]] | None]:
        if not isinstance(schema, CalibratedDecisionSchema):
            return self._compile_schema(schema), None
        if self._trained:
            raise SchemaError(
                "Context-free calibrated schemas cannot be used with trained models. "
                "Use the raw training schema; fitted weights already learn the decision bias."
            )
        if schema._owner is not self._context_free_calibration_owner:
            raise SchemaError(
                "This calibrated schema belongs to a different DecisionModel instance. "
                "Call Schema.calibrate(model) on the model that will make the decision."
            )
        if schema.policy != _CONTEXT_FREE_POLICY:
            raise SchemaError(f"Unsupported context-free calibration policy {schema.policy!r}.")

        compiled = self._compile_schema(schema.schema)
        expected_fields = set(compiled.primitives)
        if set(schema.score_offsets) != expected_fields:
            raise SchemaError("Calibrated schema fields do not match the wrapped schema.")
        for path, primitive in compiled.primitives.items():
            if len(schema.score_offsets[path]) != len(primitive.options):
                raise SchemaError(
                    f"Calibration for field {path!r} does not match its candidate count."
                )
        return compiled, schema.score_offsets

    def _calibrate_schema(
        self,
        schema: type[DecisionSchema],
    ) -> CalibratedDecisionSchema:
        if self._architecture != "causal" or self._trained:
            raise SchemaError(
                "Context-free calibration is only available for raw causal models. "
                "Use validation calibration when fitting an encoder."
            )
        compiled = self._compile_schema(schema)
        primitives = list(compiled.primitives.values())
        totals = {
            path: [0.0] * len(primitive.options) for path, primitive in compiled.primitives.items()
        }
        for null_input in _CONTEXT_FREE_NULL_INPUTS:
            result = self._runtime_instance().evaluate(
                null_input,
                primitives,
                allow_empty_context=True,
            )
            for path, expected in totals.items():
                scores = result.candidate_scores.get(path)
                if scores is None or len(scores) != len(expected):
                    raise ArtifactError(
                        f"Model did not return the expected calibration scores for field {path!r}."
                    )
                for index, score in enumerate(scores):
                    expected[index] += float(score)

        count = float(len(_CONTEXT_FREE_NULL_INPUTS))
        offsets = {
            path: tuple(total / count for total in field_totals)
            for path, field_totals in totals.items()
        }
        return CalibratedDecisionSchema(
            schema=schema,
            score_offsets=offsets,
            policy=_CONTEXT_FREE_POLICY,
            _owner=self._context_free_calibration_owner,
        )

    def _require_ready(self) -> None:
        if self._architecture == "encoder" and not self._trained:
            raise ArtifactError(
                "Encoder models must be fitted before inference. Call fit(...) and use "
                "the returned DecisionModel or load a trained encoder artifact."
            )

    def _validate_arbiter(self, arbiter: BaseArbiter | None) -> None:
        if (
            arbiter is not None
            and self._architecture == "encoder"
            and bool(cast(Any, arbiter).uses_primary_model)
        ):
            raise ValueError(
                f"{arbiter.name} cannot reuse an encoder primary model. "
                "Configure the arbiter with a dedicated causal model."
            )

    def _field_temperatures(
        self,
        compiled: CompiledDecisionSchema,
    ) -> dict[str, float]:
        calibration = self._calibration.get(compiled.fingerprint, {})
        return {
            path: float(details.get("temperature", 1.0)) for path, details in calibration.items()
        }

    def _require_trained_schema(self, compiled: CompiledDecisionSchema) -> None:
        if not self._trained or self._architecture == "causal":
            return
        schemas = self._manifest.get("schemas")
        if not isinstance(schemas, Mapping) or not schemas:
            raise ArtifactError(
                "Trained encoder artifact does not declare its training schema. Refit the "
                "model to create an artifact with decision-schema metadata."
            )
        if compiled.fingerprint in schemas:
            return
        raise SchemaError(
            f"Schema {compiled.schema.__name__!r} was not used to fit this encoder model. "
            "Use the exact training fields, primitive kinds, criteria, candidate order, "
            "and examples, or refit the model. Runtime tau values may differ."
        )

    def _primitives_with_calibration(
        self,
        compiled: CompiledDecisionSchema,
    ) -> list[BasePrimitive]:
        field_calibration = self._calibration.get(compiled.fingerprint, {})
        primitives: list[BasePrimitive] = []
        for path, primitive in compiled.primitives.items():
            calibration = field_calibration.get(path)
            calibrated_tau = (
                float(calibration["threshold"])
                if calibration is not None and "threshold" in calibration
                else None
            )
            primitives.append(primitive.bind(path, calibrated_tau=calibrated_tau))
        return primitives

    def _to_decision(
        self,
        compiled: CompiledDecisionSchema,
        result: EvaluationResult,
    ) -> Decision:
        values = {path: result.decisions[path].value for path in compiled.primitives}
        field_calibration = self._calibration.get(compiled.fingerprint, {})
        fields: dict[str, FieldDecision] = {}
        calibrated_count = 0
        for path in compiled.primitives:
            raw = result.decisions[path]
            calibration = field_calibration.get(path)
            has_validated_threshold = (
                calibration is not None
                and "threshold" in calibration
                and calibration.get("status", "calibrated") == "calibrated"
            )
            if has_validated_threshold:
                calibrated_count += 1
                calibration_status: Literal["validated", "unvalidated"] = "validated"
            else:
                calibration_status = "unvalidated"
            fields[path] = FieldDecision(
                value=raw.value,
                distribution=dict(raw.distribution),
                margin=raw.margin,
                certainty_score=(
                    raw.margin if raw.certainty_score is None else raw.certainty_score
                ),
                score=raw.raw_score,
                entropy=raw.entropy,
                normalized_entropy=raw.normalized_entropy,
                winning_level=raw.winning_level,
                standard_deviation=raw.standard_deviation,
                is_certain=raw.is_certain,
                calibration=calibration_status,
                tau=raw.tau,
                threshold_source=raw.threshold_source,
                resolved_by=raw.resolved_by,
                reasoning=raw.reasoning,
            )
        if calibrated_count == len(fields):
            overall_calibration: Literal["validated", "partial", "unvalidated"] = "validated"
        elif calibrated_count:
            overall_calibration = "partial"
        else:
            overall_calibration = "unvalidated"
        telemetry = dict(result.telemetry)
        telemetry["scorer"] = self._scorer
        telemetry["architecture"] = self._architecture
        telemetry["schema_fingerprint"] = compiled.fingerprint
        return Decision(
            value=values,
            fields=fields,
            is_certain=all(field.is_certain for field in fields.values()),
            calibration=overall_calibration,
            telemetry=telemetry,
        )

    def _artifact_adaptation(self) -> str | None:
        training_report = self._manifest.get("training_report")
        if isinstance(training_report, Mapping):
            resolved = training_report.get("resolved_training_config")
            if isinstance(resolved, Mapping):
                adaptation = resolved.get("adaptation")
                if isinstance(adaptation, str):
                    return adaptation
        fit_config = self._manifest.get("fit_config")
        if isinstance(fit_config, Mapping):
            adaptation = fit_config.get("adaptation")
            if isinstance(adaptation, str):
                return adaptation
        return None

    def _training_source(self) -> str:
        if self._trained:
            if self._artifact_path is None:
                raise ArtifactError("Trained model has no reloadable artifact source.")
            return str(self._artifact_path)
        return self._base_model

    def _validate_examples(
        self,
        examples: Sequence[Example],
        schema: type[DecisionSchema],
    ) -> None:
        for index, example in enumerate(examples):
            if not isinstance(example, Example):
                raise TrainingDataError(
                    f"Training item {index} must be an Example, got {type(example).__name__}."
                )
            try:
                validate_training_target(schema, example.target)
            except (TypeError, ValueError) as exc:
                raise TrainingDataError(f"Training item {index}: {exc}") from exc

    def _load_validation(
        self,
        validation: Iterable[Example] | str | Path,
        schema: type[DecisionSchema],
    ) -> Sequence[Example]:
        if isinstance(validation, (str, Path)):
            return _JsonlExampleSequence(validation, schema)
        loaded = list(validation)
        self._validate_examples(loaded, schema)
        return loaded

    @staticmethod
    def _group_split_indices(
        examples: Sequence[Example],
        *,
        fraction: float,
        seed: int,
    ) -> tuple[list[int], list[int]] | None:
        if not any(example.group is not None for example in examples):
            return None
        groups: dict[str, list[int]] = {}
        for index, example in enumerate(examples):
            group = example.group or f"__ungrouped_{index}"
            groups.setdefault(group, []).append(index)
        if len(groups) < 2:
            raise TrainingDataError(
                "Automatic grouped validation requires at least two distinct groups."
            )
        grouped_indices = list(groups.values())
        random.Random(seed).shuffle(grouped_indices)
        validation_target = max(1, round(len(examples) * fraction))
        validation_indices: list[int] = []
        training_indices: list[int] = []
        for group_index, indices in enumerate(grouped_indices):
            groups_remaining = len(grouped_indices) - group_index - 1
            if len(validation_indices) < validation_target and groups_remaining >= 1:
                validation_indices.extend(indices)
            else:
                training_indices.extend(indices)
        if not validation_indices or not training_indices:
            raise TrainingDataError(
                "Grouped validation could not produce non-empty training and validation sets."
            )
        return training_indices, validation_indices

    def _split_examples(
        self,
        examples: list[Example],
        *,
        fraction: float,
        seed: int,
    ) -> tuple[list[Example], list[Example]]:
        if fraction == 0.0 or len(examples) < 2:
            return examples, []
        grouped = self._group_split_indices(examples, fraction=fraction, seed=seed)
        if grouped is not None:
            training_indices, validation_indices = grouped
            return (
                [examples[index] for index in training_indices],
                [examples[index] for index in validation_indices],
            )
        shuffled = list(examples)
        random.Random(seed).shuffle(shuffled)
        validation_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * fraction)))
        return shuffled[validation_count:], shuffled[:validation_count]

    def _split_indexed_examples(
        self,
        examples: Sequence[Example],
        *,
        fraction: float,
        seed: int,
    ) -> tuple[Sequence[Example], Sequence[Example]]:
        if fraction == 0.0 or len(examples) < 2:
            return examples, _IndexedExampleSequence(examples, array("Q"))
        grouped = self._group_split_indices(examples, fraction=fraction, seed=seed)
        if grouped is not None:
            training, validation = grouped
            return (
                _IndexedExampleSequence(examples, array("Q", training)),
                _IndexedExampleSequence(examples, array("Q", validation)),
            )
        rng = random.Random(seed)
        training_indices = array("Q")
        validation_indices = array("Q")
        for index in range(len(examples)):
            destination = validation_indices if rng.random() < fraction else training_indices
            destination.append(index)
        if not validation_indices:
            validation_indices.append(training_indices.pop())
        if not training_indices:
            training_indices.append(validation_indices.pop())
        return (
            _IndexedExampleSequence(examples, training_indices),
            _IndexedExampleSequence(examples, validation_indices),
        )

    @staticmethod
    def _dataset_digest(*datasets: Sequence[Example]) -> str:
        digest = hashlib.sha256()
        for dataset in datasets:
            for example in dataset:
                if isinstance(example.context, MultimodalContext):
                    context: Any = {
                        "text": example.context.text,
                        "images": [str(image) for image in example.context.images],
                    }
                else:
                    context = example.context
                target = {
                    key: (
                        {"probabilities": dict(value.probabilities)}
                        if isinstance(value, TargetDistribution)
                        else value
                    )
                    for key, value in example.target.items()
                }
                payload = json.dumps(
                    {
                        "context": context,
                        "target": target,
                        "group": example.group,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                digest.update(payload.encode("utf-8"))
                digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _target_counts(examples: Sequence[Example]) -> tuple[int, int]:
        hard = 0
        soft = 0
        for example in examples:
            for value in example.target.values():
                if isinstance(value, TargetDistribution):
                    soft += 1
                else:
                    hard += 1
        return hard, soft

    def _calibrate(
        self,
        examples: Sequence[Example],
        compiled: CompiledDecisionSchema,
        config: FitConfig,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        evidence: dict[str, tuple[array[float], bytearray]] = {}
        for example in examples:
            result = self._runtime_instance().evaluate(
                example.context,
                list(compiled.primitives.values()),
            )
            for path, expected in example.target.items():
                if isinstance(expected, TargetDistribution):
                    continue
                primitive = compiled.primitives[path]
                try:
                    correct = cast(Any, primitive).is_correct(
                        result.decisions[path].value,
                        expected,
                    )
                except ValueError:
                    continue
                certainty, correctness = evidence.setdefault(path, (array("d"), bytearray()))
                decision = result.decisions[path]
                certainty.append(
                    decision.margin
                    if decision.certainty_score is None
                    else decision.certainty_score
                )
                correctness.append(correct)
        field_calibration: dict[str, dict[str, Any]] = {}
        for path, (margins, correctness) in evidence.items():
            sample_count = len(margins)
            if sample_count < config.minimum_calibration_examples:
                field_calibration[path] = {
                    "status": "insufficient_data",
                    "samples": sample_count,
                    "minimum_samples": config.minimum_calibration_examples,
                    "target_precision": config.target_precision,
                }
                continue
            result = QuantileNotchCalibrator(target_accuracy=config.target_precision).fit(
                margins, correctness
            )
            entry: dict[str, Any] = {
                "status": result.status,
                "coverage": result.coverage,
                "precision": result.precision,
                "samples": result.samples,
                "target_precision": config.target_precision,
            }
            if result.threshold is not None:
                entry["threshold"] = result.threshold
            field_calibration[path] = entry
        return {compiled.fingerprint: field_calibration} if field_calibration else {}

    def _calibrate_encoder(
        self,
        examples: Sequence[Example],
        compiled: CompiledDecisionSchema,
        config: FitConfig,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        if not examples:
            return {}
        primitives = list(compiled.primitives.values())
        results: list[EvaluationResult] = []
        for start in range(0, len(examples), config.batch_size):
            contexts = [
                examples[index].context
                for index in range(start, min(start + config.batch_size, len(examples)))
            ]
            results.extend(
                self._runtime_instance().evaluate_batch(
                    contexts,
                    primitives,
                )
            )

        field_calibration: dict[str, dict[str, Any]] = {}
        for path, primitive in compiled.primitives.items():
            samples: list[tuple[torch.Tensor, torch.Tensor, Any]] = []
            for example, result in zip(examples, results):
                if path not in example.target:
                    continue
                scores = result.candidate_scores.get(path)
                if scores is None:
                    continue
                raw_target = example.target[path]
                target_value = (
                    raw_target.probabilities
                    if isinstance(raw_target, TargetDistribution)
                    else raw_target
                )
                target = cast(Any, primitive).target_distribution(target_value)
                samples.append((torch.tensor(scores), target, raw_target))
            if len(samples) < config.minimum_calibration_examples:
                continue

            baseline_brier = sum(
                float(torch.mean((torch.softmax(logits, dim=-1) - target) ** 2))
                for logits, target, _ in samples
            ) / len(samples)
            temperatures = torch.logspace(-2, 1, steps=80)
            best_temperature = 1.0
            best_brier = baseline_brier
            for temperature in temperatures:
                brier = 0.0
                for logits, target, _ in samples:
                    probability = torch.softmax(logits / float(temperature), dim=-1)
                    brier += float(torch.mean((probability - target) ** 2))
                brier /= len(samples)
                if brier < best_brier:
                    best_brier = brier
                    best_temperature = float(temperature)
            entry: dict[str, Any] = {
                "temperature": best_temperature,
                "brier_before": baseline_brier,
                "brier_after": best_brier,
                "samples": len(samples),
            }
            margins = array("d")
            correctness = bytearray()
            for logits, _, expected in samples:
                if isinstance(expected, TargetDistribution):
                    continue
                probability = torch.softmax(logits / best_temperature, dim=-1)
                decision = cast(Any, primitive).result_from_probabilities(probability)
                try:
                    correct = cast(Any, primitive).is_correct(decision.value, expected)
                except ValueError:
                    continue
                margins.append(
                    decision.margin
                    if decision.certainty_score is None
                    else decision.certainty_score
                )
                correctness.append(correct)
            threshold_samples = len(margins)
            entry["threshold_samples"] = threshold_samples
            entry["target_precision"] = config.target_precision
            if threshold_samples < config.minimum_calibration_examples:
                entry["status"] = "insufficient_data"
                entry["minimum_samples"] = config.minimum_calibration_examples
            else:
                result = QuantileNotchCalibrator(target_accuracy=config.target_precision).fit(
                    margins, correctness
                )
                entry.update(
                    {
                        "status": result.status,
                        "coverage": result.coverage,
                        "precision": result.precision,
                    }
                )
                if result.threshold is not None:
                    entry["threshold"] = result.threshold
            field_calibration[path] = entry
        return {compiled.fingerprint: field_calibration} if field_calibration else {}

    def _new_manifest(
        self,
        *,
        kind: str,
        architecture: Architecture,
        scorer: str,
        weights: str,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        runtime: dict[str, Any] = {
            "architecture": architecture,
            "scorer": scorer,
            "weights": weights,
        }
        if max_length is not None:
            runtime["max_length"] = max_length
        return {
            "format_version": _ARTIFACT_FORMAT,
            "kind": kind,
            "base_model": self._base_model,
            "runtime": runtime,
            "schema_protocol": SCHEMA_PROTOCOL,
            "calibration": {},
            "source": {
                "model": dict(cast(Mapping[str, Any], self._source_metadata["model"])),
                "tokenizer": dict(cast(Mapping[str, Any], self._source_metadata["tokenizer"])),
                "local_files_only": self.local_files_only,
            },
            "provenance": _artifact_provenance(weights=weights),
        }

    def _load_manifest(self, path: Path) -> dict[str, Any]:
        try:
            manifest = read_json(path, max_bytes=MAX_JSON_BYTES)
        except (OSError, ValueError) as exc:
            raise ArtifactError(f"Could not read Jeeva artifact manifest at {path}.") from exc
        if not isinstance(manifest, dict):
            raise ArtifactError("Jeeva artifact manifest must be a JSON object.")
        if manifest.get("format_version") != _ARTIFACT_FORMAT:
            raise ArtifactError(
                f"Unsupported Jeeva artifact format {manifest.get('format_version')!r}; "
                f"expected {_ARTIFACT_FORMAT}."
            )
        for required in (
            "kind",
            "base_model",
            "runtime",
            "schema_protocol",
            "provenance",
            "source",
        ):
            if required not in manifest:
                raise ArtifactError(f"Artifact manifest is missing required key {required!r}.")
        runtime = manifest["runtime"]
        if not isinstance(runtime, Mapping):
            raise ArtifactError("Artifact runtime specification must be an object.")
        for required in ("architecture", "scorer", "weights"):
            if required not in runtime:
                raise ArtifactError(
                    f"Artifact runtime specification is missing required key {required!r}."
                )
        architecture = runtime["architecture"]
        scorer = runtime["scorer"]
        weights = runtime["weights"]
        if (architecture, scorer) not in {
            ("causal", "letter"),
            ("causal", "bilinear"),
            ("causal", "contextual"),
            ("encoder", "candidate_marker"),
        }:
            raise ArtifactError(
                f"Unsupported artifact runtime architecture/scorer pair: "
                f"{architecture!r}/{scorer!r}."
            )
        if weights not in ("base", "adapter", "full"):
            raise ArtifactError(f"Unsupported artifact weight kind {weights!r}.")
        kind = manifest["kind"]
        if kind not in ("zero_shot", "trained", "merged"):
            raise ArtifactError(f"Unsupported artifact kind {kind!r}.")
        if kind == "zero_shot" and (architecture, scorer, weights) != (
            "causal",
            "letter",
            "base",
        ):
            raise ArtifactError(
                "Zero-shot artifacts must use the letter causal runtime and base weights."
            )
        if kind == "trained" and weights not in ("adapter", "full"):
            raise ArtifactError("Trained artifacts must contain adapter or full weights.")
        if kind == "merged" and weights != "full":
            raise ArtifactError("Merged artifacts must contain full weights.")
        if architecture == "encoder":
            max_length = runtime.get("max_length")
            if not isinstance(max_length, int) or max_length < 32:
                raise ArtifactError(
                    "Encoder artifacts require an integer runtime.max_length of at least 32."
                )
        if manifest["schema_protocol"] != SCHEMA_PROTOCOL:
            raise ArtifactError(
                f"Unsupported schema protocol {manifest['schema_protocol']!r}; "
                "refit the model with DecisionSchema."
            )
        provenance = manifest["provenance"]
        if not isinstance(provenance, Mapping):
            raise ArtifactError("Artifact provenance must be an object.")
        created_at = provenance.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            raise ArtifactError("Artifact provenance requires a creation timestamp.")
        try:
            created_datetime = datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise ArtifactError("Artifact provenance creation timestamp is invalid.") from exc
        if created_datetime.utcoffset() is None:
            raise ArtifactError("Artifact provenance creation timestamp must include a timezone.")
        versions = provenance.get("versions")
        if not isinstance(versions, Mapping):
            raise ArtifactError("Artifact provenance versions must be an object.")
        required_versions = {"jeeva", "python", "torch", "transformers"}
        if weights == "adapter":
            required_versions.add("peft")
        fit_config = manifest.get("fit_config")
        is_qlora = isinstance(fit_config, Mapping) and fit_config.get("adaptation") == "qlora"
        if is_qlora:
            required_versions.add("bitsandbytes")
            training_report = manifest.get("training_report")
            quantization = (
                training_report.get("quantization")
                if isinstance(training_report, Mapping)
                else None
            )
            if not isinstance(quantization, Mapping) or (
                quantization.get("backend") != "bitsandbytes"
                or quantization.get("bits") != 4
                or quantization.get("quant_type") != "nf4"
                or quantization.get("double_quantization") is not True
            ):
                raise ArtifactError(
                    "QLoRA artifacts require 4-bit bitsandbytes NF4 quantization metadata."
                )
        missing_versions = sorted(required_versions - set(versions))
        if missing_versions:
            raise ArtifactError(
                "Artifact provenance is missing required versions: "
                + ", ".join(missing_versions)
                + "."
            )
        for dependency in required_versions:
            version = versions[dependency]
            if not isinstance(version, str) or not version:
                raise ArtifactError(
                    f"Artifact provenance version for {dependency!r} must be a non-empty string."
                )
        source = manifest["source"]
        if not isinstance(source, Mapping):
            raise ArtifactError("Artifact source provenance must be an object.")
        if not isinstance(source.get("local_files_only"), bool):
            raise ArtifactError("Artifact source.local_files_only must be a boolean.")
        for component in ("model", "tokenizer"):
            details = source.get(component)
            if not isinstance(details, Mapping):
                raise ArtifactError(f"Artifact source.{component} must be an object.")
            for identifier in ("requested_id", "resolved_id"):
                value = details.get(identifier)
                if not isinstance(value, str) or not value:
                    raise ArtifactError(
                        f"Artifact source.{component}.{identifier} must be a non-empty string."
                    )
            for revision_key in ("requested_revision", "resolved_revision"):
                value = details.get(revision_key)
                if value is not None and (not isinstance(value, str) or not value):
                    raise ArtifactError(
                        f"Artifact source.{component}.{revision_key} must be null or "
                        "a non-empty string."
                    )
        return manifest

    def _write_manifest(self, directory: Path, manifest: Mapping[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dict(manifest), indent=2, sort_keys=True)
        (directory / _MANIFEST_NAME).write_text(payload + "\n", encoding="utf-8")
