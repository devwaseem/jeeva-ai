from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, cast

from jeeva.schema.primitives import BasePrimitive, Choice, Noul, Score
from jeeva.schema.targets import DISTRIBUTION_TAG, TargetDistribution, TargetKey

SCHEMA_PROTOCOL = "decision-schema-v4"
MAX_SCHEMA_FIELDS = 16
MAX_CANDIDATES_PER_FIELD = 64
MAX_TOTAL_CANDIDATES = 128
MAX_FIELD_NAME_CHARACTERS = 128
MAX_PROMPT_CHARACTERS = 16_384
MAX_OPTION_CHARACTERS = 4_096
MAX_SCHEMA_CHARACTERS = 131_072


class DecisionSchema:
    """Declarative collection of finite decision fields.

    Subclasses are schema definitions, not value containers. Pass the subclass itself to
    ``DecisionModel.decide`` and receive a plain dictionary as the result value.
    """

    __decision_fields__: ClassVar[Mapping[str, BasePrimitive]] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        fields: dict[str, BasePrimitive] = {}
        for base in reversed(cls.__mro__[1:]):
            inherited = getattr(base, "__decision_fields__", None)
            if inherited:
                fields.update(inherited)
        for name, value in cls.__dict__.items():
            if isinstance(value, BasePrimitive):
                if name.startswith("_"):
                    raise TypeError("Decision field names cannot start with an underscore.")
                fields[name] = value
        if not fields:
            raise TypeError(
                "DecisionSchema subclasses must declare at least one Noul, Choice, or Score field."
            )
        cls.__decision_fields__ = MappingProxyType(fields)

    def __new__(cls, *_: Any, **__: Any) -> DecisionSchema:
        raise TypeError(f"{cls.__name__} is a schema definition and cannot be instantiated.")

    @classmethod
    def fields(cls) -> Mapping[str, BasePrimitive]:
        return cls.__decision_fields__

    @classmethod
    def calibrate(cls, model: Any) -> CalibratedDecisionSchema:
        """Bind a data-free prior correction for one zero-shot model.

        This correction can narrow margins or reduce accuracy when the model's
        context-free prior is useful for the target distribution. Evaluate it on
        representative labeled examples before deployment.
        """
        calibrate_schema = getattr(model, "_calibrate_schema", None)
        if calibrate_schema is None:
            raise TypeError("model must be a DecisionModel.")
        return calibrate_schema(cls)


@dataclass(frozen=True)
class CalibratedDecisionSchema:
    """A schema paired with context-free offsets from one zero-shot model instance."""

    schema: type[DecisionSchema]
    score_offsets: Mapping[str, tuple[float, ...]]
    policy: str
    _owner: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.schema, type) or not issubclass(self.schema, DecisionSchema):
            raise TypeError("Calibrated schemas must wrap a DecisionSchema subclass.")
        if not isinstance(self.policy, str) or not self.policy:
            raise ValueError("Calibration policy must be a non-empty string.")
        normalized: dict[str, tuple[float, ...]] = {}
        for path, offsets in self.score_offsets.items():
            values = tuple(float(value) for value in offsets)
            if not isinstance(path, str) or not path or not values:
                raise ValueError("Calibration offsets require non-empty field names and values.")
            if any(not math.isfinite(value) for value in values):
                raise ValueError("Calibration offsets must be finite.")
            normalized[path] = values
        object.__setattr__(self, "score_offsets", MappingProxyType(normalized))


@dataclass(frozen=True)
class CompiledDecisionSchema:
    """Canonical structural representation of one DecisionSchema class."""

    schema: type[DecisionSchema]
    primitives: Mapping[str, BasePrimitive]
    fingerprint: str
    descriptor: Mapping[str, Any]


def _field_descriptor(primitive: BasePrimitive) -> dict[str, Any]:
    return primitive.descriptor(primitive.key or primitive.kind)


def _noul_criteria_from_definition(
    raw_criteria: Any,
    *,
    field_name: str,
) -> dict[bool, str]:
    if not isinstance(raw_criteria, Mapping):
        raise ValueError(
            f"Noul field {field_name!r} criteria must map 'true' and 'false' to descriptions."
        )
    if set(raw_criteria) != {"true", "false"}:
        raise ValueError(
            f"Noul field {field_name!r} criteria must contain exactly 'true' and 'false'."
        )
    return {
        True: raw_criteria["true"],
        False: raw_criteria["false"],
    }


def _noul_examples_from_definition(
    raw_examples: Any,
    *,
    field_name: str,
) -> dict[bool, Any] | None:
    if raw_examples is None:
        return None
    if not isinstance(raw_examples, Mapping):
        raise ValueError(f"Noul field {field_name!r} examples must map 'true' or 'false' to texts.")
    unknown = set(raw_examples) - {"true", "false"}
    if unknown:
        raise ValueError(
            f"Noul field {field_name!r} examples contain unknown values: "
            f"{sorted(str(value) for value in unknown)!r}."
        )
    return {verdict == "true": texts for verdict, texts in raw_examples.items()}


def _score_criteria_from_definition(raw_criteria: Any, *, field_name: str) -> dict[float, str]:
    if isinstance(raw_criteria, Mapping):
        criteria: dict[float, str] = {}
        for raw_value, description in raw_criteria.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Score field {field_name!r} has non-numeric value {raw_value!r}."
                ) from exc
            criteria[value] = description
        return criteria
    if isinstance(raw_criteria, Sequence) and not isinstance(raw_criteria, (str, bytes)):
        criteria = {}
        for item in raw_criteria:
            if not isinstance(item, Mapping) or "value" not in item or "description" not in item:
                raise ValueError(
                    f"Score field {field_name!r} criteria must contain value/description objects."
                )
            criteria[float(item["value"])] = item["description"]
        return criteria
    raise ValueError(
        f"Score field {field_name!r} criteria must be an object or a list of "
        "value/description objects."
    )


def _score_examples_from_definition(raw_examples: Any) -> Any:
    if not isinstance(raw_examples, Mapping):
        return raw_examples
    try:
        return {float(value): texts for value, texts in raw_examples.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Score example keys must be numeric levels.") from exc


def decision_schema_from_dict(
    definition: Mapping[str, Any],
    *,
    name: str = "DynamicDecisionSchema",
) -> type[DecisionSchema]:
    """Build a DecisionSchema class from the CLI/HTTP JSON representation."""
    if not isinstance(definition, Mapping) or not definition:
        raise ValueError("Schema definition must be a non-empty object.")
    fields: dict[str, BasePrimitive] = {}
    for field_name, field_definition in definition.items():
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError("Schema field names must be non-empty strings.")
        if not isinstance(field_definition, Mapping):
            raise ValueError(f"Schema field {field_name!r} must be an object.")
        primitive_type = str(field_definition.get("type", "")).lower()
        instructions: Any = field_definition.get("instructions")
        criteria: Any = field_definition.get("criteria")
        examples: Any = field_definition.get("examples")
        tau: Any = field_definition.get("tau")
        if primitive_type == "noul":
            fields[field_name] = Noul(
                instructions=instructions,
                criteria=_noul_criteria_from_definition(criteria, field_name=field_name),
                examples=_noul_examples_from_definition(examples, field_name=field_name),
                tau=tau,
            )
        elif primitive_type == "choice":
            fields[field_name] = Choice(
                instructions=instructions,
                criteria=criteria,
                examples=examples,
                tau=tau,
            )
        elif primitive_type == "score":
            fields[field_name] = Score(
                instructions=instructions,
                criteria=_score_criteria_from_definition(
                    criteria,
                    field_name=field_name,
                ),
                examples=_score_examples_from_definition(examples),
                tolerance=field_definition.get("tolerance"),
                tau=tau,
            )
        else:
            raise ValueError(
                f"Schema field {field_name!r} has unsupported type {primitive_type!r}; "
                "expected 'noul', 'choice', or 'score'."
            )
    return cast(type[DecisionSchema], type(name, (DecisionSchema,), fields))


def _validate_schema_budget(declarations: Mapping[str, BasePrimitive]) -> None:
    if len(declarations) > MAX_SCHEMA_FIELDS:
        raise ValueError(
            f"Decision schemas support at most {MAX_SCHEMA_FIELDS} fields; "
            f"received {len(declarations)}."
        )

    total_candidates = 0
    total_characters = 0
    for name, declaration in declarations.items():
        if len(name) > MAX_FIELD_NAME_CHARACTERS:
            raise ValueError(
                f"Decision field {name!r} exceeds {MAX_FIELD_NAME_CHARACTERS} characters."
            )
        prompt = declaration.prompt
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise ValueError(
                f"Decision field {name!r} prompt exceeds {MAX_PROMPT_CHARACTERS} characters."
            )
        candidates = declaration.candidates
        if len(candidates) > MAX_CANDIDATES_PER_FIELD:
            raise ValueError(
                f"Decision field {name!r} has {len(candidates)} candidates; maximum is "
                f"{MAX_CANDIDATES_PER_FIELD}."
            )
        total_candidates += len(candidates)
        total_characters += len(name) + len(prompt)
        for candidate in candidates:
            option_text = candidate.text
            if len(option_text) > MAX_OPTION_CHARACTERS:
                raise ValueError(
                    f"Decision field {name!r} has an option exceeding "
                    f"{MAX_OPTION_CHARACTERS} characters."
                )
            total_characters += len(option_text)

    if total_candidates > MAX_TOTAL_CANDIDATES:
        raise ValueError(
            f"Decision schema has {total_candidates} total candidates; maximum is "
            f"{MAX_TOTAL_CANDIDATES}."
        )
    if total_characters > MAX_SCHEMA_CHARACTERS:
        raise ValueError(
            f"Decision schema text uses {total_characters} characters; maximum is "
            f"{MAX_SCHEMA_CHARACTERS}."
        )


def compile_decision_schema(schema: type[DecisionSchema]) -> CompiledDecisionSchema:
    """Compile a schema into immutable keyed lanes and a structural fingerprint."""
    if not isinstance(schema, type) or not issubclass(schema, DecisionSchema):
        raise TypeError("Decision schemas must be DecisionSchema subclasses.")

    declarations = schema.fields()
    if not declarations:
        raise TypeError("Decision schemas must contain at least one finite decision field.")
    _validate_schema_budget(declarations)
    primitives = MappingProxyType(
        {name: declaration.bind(name) for name, declaration in declarations.items()}
    )
    descriptor: dict[str, Any] = {
        "protocol": SCHEMA_PROTOCOL,
        "fields": {
            name: _field_descriptor(declaration) for name, declaration in declarations.items()
        },
    }
    canonical = json.dumps(descriptor, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return CompiledDecisionSchema(
        schema=schema,
        primitives=primitives,
        fingerprint=fingerprint,
        descriptor=MappingProxyType(descriptor),
    )


def validate_target(
    schema: type[DecisionSchema],
    target: Mapping[str, Any],
    *,
    partial: bool = True,
) -> dict[str, bool | str | float]:
    """Validate and copy one hard-label target against a decision schema."""
    if not isinstance(target, Mapping):
        raise TypeError("Decision targets must be mappings.")
    fields = schema.fields()
    unknown = sorted(set(target) - set(fields))
    if unknown:
        raise ValueError(f"Unknown decision target fields: {', '.join(unknown)}.")
    if not partial:
        missing = [name for name in fields if name not in target]
        if missing:
            raise ValueError(f"Missing decision target fields: {', '.join(missing)}.")

    validated: dict[str, bool | str | float] = {}
    for name, value in target.items():
        validated[name] = fields[name].validate_target(value, field=name)
    if not validated:
        raise ValueError("Decision targets must label at least one schema field.")
    return validated


TrainingTargetValue = bool | str | float | TargetDistribution


def _validated_distribution(
    primitive: BasePrimitive,
    value: TargetDistribution,
    *,
    field: str,
) -> TargetDistribution:
    probabilities = value.probabilities
    normalized: dict[TargetKey, float]
    if isinstance(primitive, Choice):
        expected = list(primitive.criteria)
        if any(not isinstance(candidate, str) for candidate in probabilities):
            raise TypeError(f"Choice distribution keys for {field!r} must be strings.")
        normalized = dict(probabilities)
    elif isinstance(primitive, Noul):
        expected = [True, False]
        if any(type(candidate) is not bool for candidate in probabilities):
            raise TypeError(f"Noul distribution keys for {field!r} must be True and False.")
        normalized = dict(probabilities)
    elif isinstance(primitive, Score):
        expected = list(primitive.level_values)
        if any(
            isinstance(candidate, bool) or not isinstance(candidate, (int, float))
            for candidate in probabilities
        ):
            raise TypeError(f"Score distribution keys for {field!r} must be numeric levels.")
        normalized = {
            float(candidate): probability for candidate, probability in probabilities.items()
        }
    else:
        raise TypeError(f"Unsupported decision primitive for field {field!r}.")

    missing = [candidate for candidate in expected if candidate not in normalized]
    unknown = [candidate for candidate in normalized if candidate not in expected]
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing!r}")
        if unknown:
            details.append(f"unknown {unknown!r}")
        raise ValueError(
            f"TargetDistribution keys for {field!r} must exactly match its candidates "
            f"({', '.join(details)})."
        )
    return TargetDistribution(normalized)


def _tagged_distribution(
    primitive: BasePrimitive,
    value: Mapping[str, Any],
    *,
    field: str,
) -> TargetDistribution:
    if set(value) != {DISTRIBUTION_TAG}:
        raise ValueError(
            f"Tagged distribution for {field!r} must contain only {DISTRIBUTION_TAG!r}."
        )
    raw = value[DISTRIBUTION_TAG]
    if not isinstance(raw, Mapping):
        raise TypeError(f"{DISTRIBUTION_TAG!r} for {field!r} must be a mapping.")

    labels = primitive.distribution_labels
    unknown = [str(label) for label in raw if str(label) not in labels]
    missing = [label for label in labels if label not in raw]
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing!r}")
        if unknown:
            details.append(f"unknown {unknown!r}")
        raise ValueError(
            f"Tagged distribution keys for {field!r} must exactly match its labels "
            f"({', '.join(details)})."
        )

    if isinstance(primitive, Choice):
        candidates: Sequence[bool | str | float] = list(primitive.criteria)
    elif isinstance(primitive, Noul):
        candidates = [True, False]
    elif isinstance(primitive, Score):
        candidates = list(primitive.level_values)
    else:
        raise TypeError(f"Unsupported decision primitive for field {field!r}.")
    return _validated_distribution(
        primitive,
        TargetDistribution(
            {candidate: raw[label] for candidate, label in zip(candidates, labels, strict=True)}
        ),
        field=field,
    )


def validate_training_target(
    schema: type[DecisionSchema],
    target: Mapping[str, Any],
    *,
    partial: bool = True,
) -> dict[str, TrainingTargetValue]:
    """Validate hard labels and explicit probability distributions for fitting."""
    if not isinstance(target, Mapping):
        raise TypeError("Decision targets must be mappings.")
    fields = schema.fields()
    unknown = sorted(set(target) - set(fields))
    if unknown:
        raise ValueError(f"Unknown decision target fields: {', '.join(unknown)}.")
    if not partial:
        missing = [name for name in fields if name not in target]
        if missing:
            raise ValueError(f"Missing decision target fields: {', '.join(missing)}.")

    validated: dict[str, TrainingTargetValue] = {}
    for name, value in target.items():
        primitive = fields[name]
        if isinstance(value, TargetDistribution):
            validated[name] = _validated_distribution(primitive, value, field=name)
        elif isinstance(value, Mapping) and DISTRIBUTION_TAG in value:
            validated[name] = _tagged_distribution(primitive, value, field=name)
        else:
            validated[name] = primitive.validate_target(value, field=name)
    if not validated:
        raise ValueError("Decision targets must label at least one schema field.")
    return validated
