from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import TypeAlias

TargetKey: TypeAlias = bool | str | int | float
DISTRIBUTION_TAG = "$distribution"


@dataclass(frozen=True)
class TargetDistribution:
    """Explicit probability target for one finite decision field."""

    probabilities: Mapping[TargetKey, float]

    def __post_init__(self) -> None:
        if not isinstance(self.probabilities, Mapping) or not self.probabilities:
            raise ValueError("TargetDistribution probabilities must be a non-empty mapping.")

        validated: dict[TargetKey, float] = {}
        for candidate, probability in self.probabilities.items():
            if not isinstance(candidate, (bool, str, int, float)):
                raise TypeError(
                    "TargetDistribution candidate keys must be booleans, strings, or numbers."
                )
            if isinstance(probability, bool) or not isinstance(probability, Real):
                raise TypeError("TargetDistribution probabilities must be numbers.")
            value = float(probability)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("TargetDistribution probabilities must be finite and nonnegative.")
            validated[candidate] = value

        total = math.fsum(validated.values())
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"TargetDistribution probabilities must sum to 1.0; got {total:.12g}.")
        object.__setattr__(self, "probabilities", MappingProxyType(validated))
