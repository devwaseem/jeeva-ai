from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal

import torch

ThresholdSource = Literal["manual", "calibrated", "default"]


def compute_entropy(probs: torch.Tensor) -> tuple[float, float]:
    """Return Shannon entropy in bits and entropy normalized to ``[0, 1]``."""
    k = probs.shape[-1]
    p = probs.clamp(min=1e-12)
    entropy_bits = -float((p * torch.log2(p)).sum().item())
    if k > 1:
        max_entropy = math.log2(k)
        normalized = float(min(1.0, max(0.0, entropy_bits / max_entropy)))
    else:
        normalized = 0.0
    return entropy_bits, normalized


@dataclass
class DecisionResult:
    """The scored decision for one compiled schema field."""

    field: str
    primitive: str
    value: Any
    is_certain: bool
    margin: float
    raw_score: float
    resolved_by: str = "gatekeeper"
    reasoning: str | None = None
    distribution: dict[str, float] = dc_field(default_factory=dict)
    entropy: float | None = None
    normalized_entropy: float | None = None
    tau: float = 0.30
    threshold_source: ThresholdSource = "default"
    certainty_score: float | None = None
    winning_level: float | None = None
    standard_deviation: float | None = None

    def __post_init__(self) -> None:
        if self.certainty_score is None:
            self.certainty_score = self.margin

    @property
    def is_ambiguous(self) -> bool:
        return not self.is_certain

    def to_dict(self) -> dict[str, Any]:
        value = self.value
        if hasattr(value, "value"):
            value = value.value
        elif hasattr(value, "name"):
            value = value.name
        data: dict[str, Any] = {
            "field": self.field,
            "primitive": self.primitive,
            "value": value,
            "is_certain": self.is_certain,
            "resolved_by": self.resolved_by,
            "margin": round(self.margin, 4),
            "certainty_score": round(
                0.0 if self.certainty_score is None else self.certainty_score, 4
            ),
            "raw_score": round(self.raw_score, 4),
            "distribution": {key: round(score, 4) for key, score in self.distribution.items()},
            "tau": self.tau,
            "threshold_source": self.threshold_source,
        }
        if self.entropy is not None:
            data["entropy"] = round(self.entropy, 4)
        if self.normalized_entropy is not None:
            data["normalized_entropy"] = round(self.normalized_entropy, 4)
        if self.winning_level is not None:
            data["winning_level"] = self.winning_level
        if self.standard_deviation is not None:
            data["standard_deviation"] = round(self.standard_deviation, 4)
        if self.reasoning is not None:
            data["reasoning"] = self.reasoning
        return data


@dataclass(frozen=True)
class Candidate:
    """One stable decision value paired with its model-facing semantics."""

    value: bool | str | float
    description: str

    @property
    def label(self) -> str:
        if type(self.value) is bool:
            return "true" if self.value else "false"
        if isinstance(self.value, float):
            normalized = 0.0 if self.value == 0.0 else self.value
            return repr(normalized)
        return str(self.value)

    @property
    def text(self) -> str:
        return f"{self.label} — {self.description}"


def _validate_tau(tau: float | None) -> None:
    if tau is None:
        return
    if isinstance(tau, bool) or not isinstance(tau, (int, float)):
        raise TypeError("tau must be a number in [0, 1] or None.")
    if not math.isfinite(float(tau)) or not 0.0 <= float(tau) <= 1.0:
        raise ValueError("tau must be finite and in [0, 1].")


def _normalize_examples(
    examples: Mapping[Any, Sequence[str] | str] | None,
    *,
    valid_keys: set[Any],
    label: str,
) -> Mapping[Any, tuple[str, ...]]:
    if examples is None:
        return MappingProxyType({})
    if not isinstance(examples, Mapping):
        raise TypeError(f"{label} examples must be a mapping from result values to example texts.")

    normalized: dict[Any, tuple[str, ...]] = {}
    for key, values in examples.items():
        if key not in valid_keys or type(key) not in {type(candidate) for candidate in valid_keys}:
            expected = ", ".join(repr(candidate) for candidate in valid_keys)
            raise ValueError(f"{label} example key {key!r} is invalid; expected one of {expected}.")
        sequence = (values,) if isinstance(values, str) else tuple(values)
        if not sequence:
            raise ValueError(f"{label} examples for {key!r} must not be empty.")
        if any(not isinstance(item, str) or not item.strip() for item in sequence):
            raise ValueError(f"{label} examples for {key!r} must be non-empty strings.")
        normalized[key] = tuple(item.strip() for item in sequence)
    return MappingProxyType(normalized)


@dataclass(frozen=True, kw_only=True)
class BasePrimitive(ABC):
    """Immutable declaration that owns one decision field's complete semantics."""

    tau: float | None = None
    _key: str | None = dc_field(default=None, init=False, repr=False, compare=False)
    _calibrated_tau: float | None = dc_field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_tau(self.tau)

    @property
    def key(self) -> str | None:
        return self._key

    @property
    @abstractmethod
    def kind(self) -> str:
        """Stable schema kind used by artifacts and serving."""

    @property
    @abstractmethod
    def prompt(self) -> str:
        """Rendered semantic instructions used by gatekeepers and arbiters."""

    @property
    @abstractmethod
    def candidates(self) -> tuple[Candidate, ...]:
        """Ordered stable values and their model-facing semantic descriptions."""

    @property
    def options(self) -> list[str]:
        """Ordered semantic candidate texts used by scoring layers."""
        return [candidate.text for candidate in self.candidates]

    @property
    def distribution_labels(self) -> list[str]:
        """Stable labels used as keys in result distributions."""
        return [candidate.label for candidate in self.candidates]

    @property
    def threshold_source(self) -> ThresholdSource:
        if self.tau is not None:
            return "manual"
        if self._calibrated_tau is not None:
            return "calibrated"
        return "default"

    def bind(self, key: str, *, calibrated_tau: float | None = None) -> BasePrimitive:
        """Return an immutable keyed runtime clone without mutating the declaration."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Decision field names must be non-empty strings.")
        _validate_tau(calibrated_tau)
        clone = copy.copy(self)
        object.__setattr__(clone, "_key", key)
        object.__setattr__(clone, "_calibrated_tau", calibrated_tau)
        return clone

    def resolve_tau(self, default_tau: float) -> tuple[float, ThresholdSource]:
        _validate_tau(default_tau)
        if self.tau is not None:
            return float(self.tau), "manual"
        if self._calibrated_tau is not None:
            return float(self._calibrated_tau), "calibrated"
        return float(default_tau), "default"

    def build_branch_prompt(self) -> str:
        rendered = "\n".join(f"- {option}" for option in self.options)
        return f"{self.prompt}\n\nOptions:\n{rendered}\n\nAnswer: "

    def build_letter_branch_prompt(
        self,
        option_order: Sequence[int] | None = None,
    ) -> tuple[str, list[str], list[str]]:
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        options = self.options
        if len(options) > len(letters):
            raise ValueError(
                f"Decision fields support at most {len(letters)} options for letter readout; "
                f"{self.key!r} declares {len(options)}. Use mode='dense'."
            )
        order = tuple(range(len(options))) if option_order is None else tuple(option_order)
        expected = tuple(range(len(options)))
        if (
            any(isinstance(index, bool) or not isinstance(index, int) for index in order)
            or tuple(sorted(order)) != expected
        ):
            raise ValueError(f"option_order must be a permutation of {expected}; received {order}.")
        ordered_options = [options[index] for index in order]
        candidate_letters = letters[: len(options)]
        rendered = "\n".join(
            f"{letter}. {option}" for letter, option in zip(candidate_letters, ordered_options)
        )
        return (
            f"{self.prompt}\n\nOptions:\n{rendered}\n\nSelect the single best option letter.",
            ordered_options,
            candidate_letters,
        )

    @abstractmethod
    def descriptor(self, path: str) -> dict[str, Any]:
        """Return the semantic artifact descriptor for this field."""

    @abstractmethod
    def validate_target(self, value: Any, *, field: str) -> bool | str | float:
        """Validate and normalize one hard target."""

    @abstractmethod
    def target_distribution(self, value: Any) -> torch.Tensor:
        """Convert a hard or soft target into the candidate distribution."""

    @abstractmethod
    def selection_index(self, value: Any) -> int:
        """Return the candidate index represented by a decoded value."""

    @abstractmethod
    def is_correct(self, prediction: Any, target: Any) -> bool:
        """Return whether a prediction satisfies this primitive's correctness contract."""

    @abstractmethod
    def result_from_probabilities(
        self,
        probabilities: torch.Tensor,
        *,
        cascade_threshold: float | None = None,
        resolved_by: str = "gatekeeper",
        reasoning: str | None = None,
    ) -> DecisionResult:
        """Decode a normalized candidate distribution into a field result."""

    def parse_similarities(
        self,
        similarities: torch.Tensor,
        cascade_threshold: float | None = None,
        temperature: float = 0.03,
    ) -> DecisionResult:
        scale = 1.0 / max(1e-4, temperature)
        probabilities = torch.softmax(similarities * scale, dim=-1)
        return self.result_from_probabilities(
            probabilities,
            cascade_threshold=cascade_threshold,
        )


def _target_probabilities(values: Any, *, size: int, field: str) -> torch.Tensor:
    probs = torch.as_tensor(values, dtype=torch.float32)
    if probs.ndim != 1 or probs.numel() != size:
        raise ValueError(f"Target for {field!r} must contain exactly {size} probabilities.")
    if not torch.isfinite(probs).all():
        raise ValueError(f"Target distribution for {field!r} must contain only finite values.")
    if (probs < 0).any():
        raise ValueError(f"Target distribution for {field!r} cannot contain negative values.")
    total = probs.sum()
    if total <= 0:
        raise ValueError(f"Target distribution for {field!r} must have a positive sum.")
    return probs / total


def _ranked_probability_evidence(
    probabilities: torch.Tensor,
) -> tuple[list[float], int, float, float, float]:
    probs = probabilities.detach().to("cpu", torch.float32)
    values = probs.tolist()
    ranked = sorted(range(len(values)), key=values.__getitem__, reverse=True)
    best_index = ranked[0]
    runner_up = values[ranked[1]] if len(ranked) > 1 else 0.0
    margin = values[best_index] - runner_up
    entropy, normalized_entropy = compute_entropy(probs)
    return values, best_index, margin, entropy, normalized_entropy


@dataclass(frozen=True, init=False)
class Choice(BasePrimitive):
    """A finite categorical decision over described stable values."""

    instructions: str
    criteria: Mapping[str, str]
    examples: Mapping[str, Sequence[str] | str] | None = dc_field(default=None, kw_only=True)

    def __init__(
        self,
        instructions: str,
        criteria: Mapping[str, str],
        *,
        examples: Mapping[str, Sequence[str] | str] | None = None,
        tau: float | None = None,
    ) -> None:
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "criteria", criteria)
        object.__setattr__(self, "examples", examples)
        self.__post_init__()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("Choice instructions must be a non-empty string.")
        if not isinstance(self.criteria, Mapping):
            raise TypeError("Choice criteria must map stable string values to descriptions.")

        normalized: dict[str, str] = {}
        for raw_value, raw_description in self.criteria.items():
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError("Choice values must be non-empty strings.")
            if not isinstance(raw_description, str) or not raw_description.strip():
                raise ValueError("Choice descriptions must be non-empty strings.")
            value = raw_value.strip()
            if value in normalized:
                raise ValueError("Choice values must be unique after trimming.")
            normalized[value] = raw_description.strip()
        if len(normalized) < 2:
            raise ValueError("Choice requires at least two criteria.")
        if len(set(normalized.values())) != len(normalized):
            raise ValueError("Choice descriptions must be unique.")

        object.__setattr__(self, "instructions", self.instructions.strip())
        object.__setattr__(self, "criteria", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "examples",
            _normalize_examples(
                self.examples,
                valid_keys=set(normalized),
                label="Choice",
            ),
        )

    @property
    def kind(self) -> str:
        return "choice"

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(
            Candidate(value=value, description=description)
            for value, description in self.criteria.items()
        )

    @property
    def prompt(self) -> str:
        lines: list[str] = []
        if self.examples:
            lines.append("Examples:")
            for value, description in self.criteria.items():
                candidate = Candidate(value=value, description=description)
                for example in self.examples.get(value, ()):
                    lines.append(f"- {candidate.text}: {example}")
            lines.append("")
        lines.extend(("Instructions:", self.instructions))
        return "\n".join(lines)

    def descriptor(self, path: str) -> dict[str, Any]:
        descriptor: dict[str, Any] = {
            "type": self.kind,
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }
        primitive_examples = self.examples or {}
        examples = {
            value: list(primitive_examples[value])
            for value in self.criteria
            if value in primitive_examples
        }
        if examples:
            descriptor["examples"] = examples
        return descriptor

    def validate_target(self, value: Any, *, field: str) -> str:
        if not isinstance(value, str) or value not in self.criteria:
            options = ", ".join(repr(option) for option in self.criteria)
            raise ValueError(f"Choice target {field!r} must be one of {options}.")
        return value

    def target_distribution(self, value: Any) -> torch.Tensor:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned not in self.criteria:
                raise ValueError(
                    f"Target {value!r} does not match any Choice value: {list(self.criteria)}"
                )
            probs = torch.zeros(len(self.criteria), dtype=torch.float32)
            probs[list(self.criteria).index(cleaned)] = 1.0
            return probs
        if isinstance(value, Mapping):
            return _target_probabilities(
                [float(value.get(option, 0.0)) for option in self.criteria],
                size=len(self.criteria),
                field=self.key or self.kind,
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return _target_probabilities(
                value,
                size=len(self.criteria),
                field=self.key or self.kind,
            )
        raise TypeError(f"Unsupported target type {type(value)} for Choice field {self.key}")

    def selection_index(self, value: Any) -> int:
        return list(self.criteria).index(str(value))

    def is_correct(self, prediction: Any, target: Any) -> bool:
        return prediction == target

    def result_from_probabilities(
        self,
        probabilities: torch.Tensor,
        *,
        cascade_threshold: float | None = None,
        resolved_by: str = "gatekeeper",
        reasoning: str | None = None,
    ) -> DecisionResult:
        values, best_index, margin, entropy, normalized_entropy = _ranked_probability_evidence(
            probabilities
        )
        if len(values) != len(self.criteria):
            raise ValueError(
                f"Choice expected {len(self.criteria)} probabilities, got {len(values)}."
            )
        tau, source = self.resolve_tau(0.30 if cascade_threshold is None else cascade_threshold)
        choice_values = list(self.criteria)
        return DecisionResult(
            field=self.key or self.kind,
            primitive=self.kind,
            value=choice_values[best_index],
            is_certain=bool(margin >= tau),
            margin=margin,
            certainty_score=margin,
            raw_score=values[best_index],
            resolved_by=resolved_by,
            reasoning=reasoning,
            distribution=dict(zip(self.distribution_labels, values, strict=True)),
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            tau=tau,
            threshold_source=source,
        )


@dataclass(frozen=True, init=False)
class Noul(BasePrimitive):
    """A calibrated Boolean decision over described true and false outcomes."""

    instructions: str
    criteria: Mapping[bool, str]
    examples: Mapping[bool, Sequence[str] | str] | None = dc_field(default=None, kw_only=True)

    def __init__(
        self,
        instructions: str,
        criteria: Mapping[bool, str],
        *,
        examples: Mapping[bool, Sequence[str] | str] | None = None,
        tau: float | None = None,
    ) -> None:
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "criteria", criteria)
        object.__setattr__(self, "examples", examples)
        self.__post_init__()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("Noul instructions must be a non-empty string.")
        if not isinstance(self.criteria, Mapping):
            raise TypeError("Noul criteria must map True and False to descriptions.")
        if len(self.criteria) != 2 or {type(key) for key in self.criteria} != {bool}:
            raise ValueError("Noul criteria must contain exactly the Boolean keys True and False.")
        if set(self.criteria) != {True, False}:
            raise ValueError("Noul criteria must contain exactly the Boolean keys True and False.")

        normalized: dict[bool, str] = {}
        for value in (True, False):
            description = self.criteria[value]
            if not isinstance(description, str) or not description.strip():
                raise ValueError("Noul criteria descriptions must be non-empty strings.")
            normalized[value] = description.strip()
        if normalized[True] == normalized[False]:
            raise ValueError("Noul criteria descriptions must be distinct.")

        object.__setattr__(self, "instructions", self.instructions.strip())
        object.__setattr__(self, "criteria", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "examples",
            _normalize_examples(self.examples, valid_keys={True, False}, label="Noul"),
        )

    @property
    def kind(self) -> str:
        return "noul"

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(
            Candidate(value=value, description=self.criteria[value]) for value in (True, False)
        )

    @property
    def prompt(self) -> str:
        lines: list[str] = []
        if self.examples:
            lines.append("Examples:")
            for value in (True, False):
                candidate = Candidate(value=value, description=self.criteria[value])
                for example in self.examples.get(value, ()):
                    lines.append(f"- {candidate.text}: {example}")
            lines.append("")
        lines.extend(("Instructions:", self.instructions))
        return "\n".join(lines)

    def descriptor(self, path: str) -> dict[str, Any]:
        descriptor: dict[str, Any] = {
            "type": self.kind,
            "instructions": self.instructions,
            "criteria": {
                "true": self.criteria[True],
                "false": self.criteria[False],
            },
        }
        primitive_examples = self.examples or {}
        examples = {
            ("true" if verdict else "false"): list(primitive_examples[verdict])
            for verdict in (True, False)
            if verdict in primitive_examples
        }
        if examples:
            descriptor["examples"] = examples
        return descriptor

    def validate_target(self, value: Any, *, field: str) -> bool:
        if type(value) is not bool:
            raise TypeError(f"Noul target {field!r} must be a boolean.")
        return value

    def target_distribution(self, value: Any) -> torch.Tensor:
        if type(value) is bool:
            return torch.tensor([1.0, 0.0] if value else [0.0, 1.0])
        if isinstance(value, Real):
            probability = float(value)
            return _target_probabilities(
                [probability, 1.0 - probability],
                size=2,
                field=self.key or self.kind,
            )
        if isinstance(value, Mapping):
            true_prob = value.get(True, value.get("true", 0.0))
            false_prob = value.get(False, value.get("false", 0.0))
            return _target_probabilities(
                [float(true_prob), float(false_prob)],
                size=2,
                field=self.key or self.kind,
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return _target_probabilities(value, size=2, field=self.key or self.kind)
        raise TypeError(f"Unsupported target type {type(value)} for Noul field {self.key}")

    def selection_index(self, value: Any) -> int:
        return 0 if bool(value) else 1

    def is_correct(self, prediction: Any, target: Any) -> bool:
        return prediction is target

    def result_from_probabilities(
        self,
        probabilities: torch.Tensor,
        *,
        cascade_threshold: float | None = None,
        resolved_by: str = "gatekeeper",
        reasoning: str | None = None,
    ) -> DecisionResult:
        values, best_index, margin, entropy, normalized_entropy = _ranked_probability_evidence(
            probabilities
        )
        if len(values) != 2:
            raise ValueError(f"Noul expected 2 probabilities, got {len(values)}.")
        tau, source = self.resolve_tau(0.30 if cascade_threshold is None else cascade_threshold)
        return DecisionResult(
            field=self.key or self.kind,
            primitive=self.kind,
            value=best_index == 0,
            is_certain=bool(margin >= tau),
            margin=margin,
            certainty_score=margin,
            raw_score=values[best_index],
            resolved_by=resolved_by,
            reasoning=reasoning,
            distribution=dict(zip(self.distribution_labels, values, strict=True)),
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            tau=tau,
            threshold_source=source,
        )


def _canonical_level(value: float) -> str:
    normalized = 0.0 if value == 0.0 else value
    return repr(normalized)


@dataclass(frozen=True, init=False)
class Score(BasePrimitive):
    """An ordinal decision decoded as the expected value over semantic numeric anchors."""

    instructions: str
    criteria: Mapping[float, str]
    examples: Mapping[float, Sequence[str] | str] | None = dc_field(default=None, kw_only=True)
    tolerance: float | None = dc_field(default=None, kw_only=True)

    def __init__(
        self,
        instructions: str,
        criteria: Mapping[float, str],
        *,
        examples: Mapping[float, Sequence[str] | str] | None = None,
        tolerance: float | None = None,
        tau: float | None = None,
    ) -> None:
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "criteria", criteria)
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "tolerance", tolerance)
        self.__post_init__()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("Score instructions must be a non-empty string.")
        if not isinstance(self.criteria, Mapping):
            raise TypeError("Score criteria must be a mapping from finite numbers to descriptions.")
        normalized_criteria: list[tuple[float, str]] = []
        for raw_value, description in self.criteria.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError("Score criterion values must be finite numbers, not booleans.")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("Score criterion values must be finite.")
            if not isinstance(description, str) or not description.strip():
                raise ValueError("Score criterion descriptions must be non-empty strings.")
            normalized_criteria.append((value, description.strip()))
        if len(normalized_criteria) < 2:
            raise ValueError("Score requires at least two criteria.")
        normalized_criteria.sort(key=lambda item: item[0])
        values = [value for value, _ in normalized_criteria]
        if len(set(values)) != len(values):
            raise ValueError("Score criterion values must be unique.")
        descriptions = [description for _, description in normalized_criteria]
        if len(set(descriptions)) != len(descriptions):
            raise ValueError("Score criterion descriptions must be unique.")
        if self.tolerance is not None:
            if (
                isinstance(self.tolerance, bool)
                or not isinstance(self.tolerance, Real)
                or not math.isfinite(float(self.tolerance))
                or float(self.tolerance) <= 0.0
            ):
                raise ValueError("Score tolerance must be a finite positive number or None.")

        normalized_examples: dict[float, tuple[str, ...]] = {}
        if self.examples is not None:
            if not isinstance(self.examples, Mapping):
                raise TypeError("Score examples must map criterion values to example texts.")
            valid_values = set(values)
            for raw_value, raw_examples in self.examples.items():
                if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                    raise ValueError("Score example keys must be configured numeric values.")
                value = float(raw_value)
                if value not in valid_values:
                    raise ValueError(
                        f"Score example key {raw_value!r} is invalid; expected one of {values}."
                    )
                sequence = (raw_examples,) if isinstance(raw_examples, str) else tuple(raw_examples)
                if not sequence or any(
                    not isinstance(item, str) or not item.strip() for item in sequence
                ):
                    raise ValueError("Score examples must be non-empty strings.")
                normalized_examples[value] = tuple(item.strip() for item in sequence)

        object.__setattr__(self, "instructions", self.instructions.strip())
        object.__setattr__(self, "criteria", MappingProxyType(dict(normalized_criteria)))
        object.__setattr__(self, "examples", MappingProxyType(normalized_examples))
        if self.tolerance is not None:
            object.__setattr__(self, "tolerance", float(self.tolerance))

    @property
    def kind(self) -> str:
        return "score"

    @property
    def level_values(self) -> tuple[float, ...]:
        return tuple(self.criteria)

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(
            Candidate(value=value, description=description)
            for value, description in self.criteria.items()
        )

    @property
    def prompt(self) -> str:
        lines: list[str] = []
        if self.examples:
            lines.append("Examples:")
            for value, description in self.criteria.items():
                candidate = Candidate(value=value, description=description)
                for example in self.examples.get(value, ()):
                    lines.append(f"- {candidate.text}: {example}")
            lines.append("")
        lines.extend(("Instructions:", self.instructions))
        return "\n".join(lines)

    def descriptor(self, path: str) -> dict[str, Any]:
        descriptor: dict[str, Any] = {
            "type": self.kind,
            "instructions": self.instructions,
            "criteria": [
                {"value": value, "description": description}
                for value, description in self.criteria.items()
            ],
            "tolerance": self.tolerance,
        }
        primitive_examples = self.examples or {}
        examples = {
            _canonical_level(value): list(primitive_examples[value])
            for value in self.level_values
            if value in primitive_examples
        }
        if examples:
            descriptor["examples"] = examples
        return descriptor

    def validate_target(self, value: Any, *, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Score target {field!r} must be a finite number.")
        target = float(value)
        if not math.isfinite(target):
            raise ValueError(f"Score target {field!r} must be finite.")
        if not self.level_values[0] <= target <= self.level_values[-1]:
            raise ValueError(
                f"Score target {field!r} must be in "
                f"[{self.level_values[0]}, {self.level_values[-1]}]."
            )
        return target

    def target_distribution(self, value: Any) -> torch.Tensor:
        size = len(self.level_values)
        if isinstance(value, Mapping):
            return _target_probabilities(
                [
                    float(value.get(level, value.get(_canonical_level(level), 0.0)))
                    for level in self.level_values
                ],
                size=size,
                field=self.key or self.kind,
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return _target_probabilities(value, size=size, field=self.key or self.kind)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Unsupported target type {type(value)} for Score field {self.key}")
        target = float(value)
        levels = self.level_values
        if not levels[0] <= target <= levels[-1]:
            raise ValueError(f"Score target {target} is outside [{levels[0]}, {levels[-1]}].")
        exact_index = next(
            (index for index, level in enumerate(levels) if target == level),
            None,
        )
        if exact_index is not None:
            probs = torch.zeros(size, dtype=torch.float32)
            probs[exact_index] = 1.0
            return probs
        upper_index = next(index for index, level in enumerate(levels) if level > target)
        lower_index = upper_index - 1
        lower, upper = levels[lower_index], levels[upper_index]
        upper_weight = (target - lower) / (upper - lower)
        probs = torch.zeros(size, dtype=torch.float32)
        probs[lower_index] = 1.0 - upper_weight
        probs[upper_index] = upper_weight
        return probs

    def selection_index(self, value: Any) -> int:
        target = float(value)
        return min(
            range(len(self.level_values)),
            key=lambda index: abs(self.level_values[index] - target),
        )

    def is_correct(self, prediction: Any, target: Any) -> bool:
        tolerance = self.tolerance
        if tolerance is None:
            values = self.level_values
            tolerance = (
                min(right - left for left, right in zip(values, values[1:], strict=False)) / 2.0
            )
        return abs(float(prediction) - float(target)) <= tolerance

    def result_from_probabilities(
        self,
        probabilities: torch.Tensor,
        *,
        cascade_threshold: float | None = None,
        resolved_by: str = "gatekeeper",
        reasoning: str | None = None,
    ) -> DecisionResult:
        probs = probabilities.detach().to("cpu", torch.float32)
        if probs.ndim != 1 or probs.numel() != len(self.level_values):
            raise ValueError(
                f"Score expected {len(self.level_values)} probabilities, got {probs.numel()}."
            )
        values = list(self.level_values)
        expectation = float(
            sum(probability * value for probability, value in zip(probs.tolist(), values))
        )
        variance = float(
            sum(
                probability * ((value - expectation) ** 2)
                for probability, value in zip(probs.tolist(), values)
            )
        )
        standard_deviation = math.sqrt(max(0.0, variance))
        span = values[-1] - values[0]
        certainty = 1.0 if span <= 0.0 else 1.0 - min(1.0, 2.0 * standard_deviation / span)
        winning_index = int(torch.argmax(probs).item())
        winning_level = values[winning_index]
        entropy, normalized_entropy = compute_entropy(probs)
        tau, source = self.resolve_tau(0.30 if cascade_threshold is None else cascade_threshold)
        return DecisionResult(
            field=self.key or self.kind,
            primitive=self.kind,
            value=expectation,
            is_certain=bool(certainty >= tau),
            margin=certainty,
            certainty_score=certainty,
            raw_score=float(probs[winning_index].item()),
            resolved_by=resolved_by,
            reasoning=reasoning,
            distribution=dict(zip(self.distribution_labels, probs.tolist(), strict=True)),
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            tau=tau,
            threshold_source=source,
            winning_level=winning_level,
            standard_deviation=standard_deviation,
        )


def normalize_schema(schema: Any) -> list[BasePrimitive]:
    """Compile a DecisionSchema class or validate an internal sequence of bound lanes."""
    from jeeva.schema.decision import DecisionSchema, compile_decision_schema

    if isinstance(schema, type) and issubclass(schema, DecisionSchema):
        return list(compile_decision_schema(schema).primitives.values())
    if isinstance(schema, Sequence) and not isinstance(schema, (str, bytes)):
        primitives = list(schema)
        if not all(isinstance(item, BasePrimitive) and item.key for item in primitives):
            raise TypeError("Internal primitive sequences must contain only compiled, keyed lanes.")
        return primitives
    raise TypeError("Schema must be a DecisionSchema class.")
