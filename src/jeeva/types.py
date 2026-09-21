"""
Strongly typed enums and type definitions for Jeeva.
"""

from enum import Enum
from typing import Any, TypeVar

import torch

T = TypeVar("T")


class DecisionMode(str, Enum):
    """Execution mode for structured decision harvesting."""

    LETTER = "letter"
    DENSE = "dense"


def normalize_decision_mode(mode: DecisionMode | str) -> str:
    """Normalize and validate a decision readout mode."""
    raw = mode.value if isinstance(mode, DecisionMode) else str(mode).lower().strip()
    supported = {candidate.value for candidate in DecisionMode}
    if raw not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported decision mode {mode!r}. Supported modes: {choices}.")
    return raw


class AmbiguityCriterion(str, Enum):
    """Metric criterion used to classify predictions as certain vs ambiguous."""

    MARGIN = "margin"
    ENTROPY = "entropy"
    BOTH = "both"


class ArbiterStrategy(str, Enum):
    """Strategy used by the cascade arbiter to resolve ambiguous decisions."""

    NONE = "none"
    SEQUENCE = "sequence"
    THINKING = "thinking"
    CLOUD = "cloud"


class DeviceType(str, Enum):
    """Hardware acceleration device target."""

    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"


class DtypeType(str, Enum):
    """Supported tensor precision datatypes."""

    FLOAT16 = "float16"
    FP16 = "fp16"
    BFLOAT16 = "bfloat16"
    BF16 = "bf16"
    FLOAT32 = "float32"
    FP32 = "fp32"
    AUTO = "auto"


def resolve_dtype(
    dtype_spec: Any,
    device: str | None = None,
) -> torch.dtype:
    """
    Resolves a dtype specification (str, torch.dtype, or None) to a concrete torch.dtype.

    Accepted inputs:
    - torch.dtype instances (e.g. torch.float16, torch.bfloat16, torch.float32)
    - Strings: 'float16', 'fp16', 'bfloat16', 'bf16', 'float32', 'fp32', 'auto'
    - None: Automatically picks torch.float16 for cuda/mps, torch.float32 for cpu.
    """
    import torch

    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec

    if dtype_spec is None or dtype_spec in ("auto", DtypeType.AUTO):
        if device in ("cuda", "mps"):
            return torch.float16
        return torch.float32

    raw = dtype_spec.value if isinstance(dtype_spec, Enum) else str(dtype_spec).lower().strip()

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }

    if raw in mapping:
        return mapping[raw]

    raise ValueError(
        f"Unsupported dtype '{dtype_spec}'. "
        f"Supported options: 'float16' ('fp16'), 'bfloat16' ('bf16'), 'float32' ('fp32'), 'auto', "
        f"or torch.dtype objects (torch.float16, torch.bfloat16, torch.float32)."
    )
