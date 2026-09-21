from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from jeeva.arbiters.base import BaseArbiter
from jeeva.core.engine import EvaluationResult
from jeeva.schema.primitives import BasePrimitive

RuntimeArchitecture = Literal["causal", "encoder"]


class DecisionRuntime(Protocol):
    """Internal execution seam shared by causal and encoder model adapters."""

    @property
    def architecture(self) -> RuntimeArchitecture: ...

    @property
    def scorer(self) -> str: ...

    @property
    def model(self) -> Any: ...

    @property
    def tokenizer(self) -> Any: ...

    @property
    def processor(self) -> Any: ...

    @property
    def is_multimodal(self) -> bool: ...

    @property
    def resolved_model_id(self) -> str: ...

    @property
    def resolved_model_revision(self) -> str | None: ...

    @property
    def resolved_tokenizer_id(self) -> str: ...

    @property
    def resolved_tokenizer_revision(self) -> str | None: ...

    def warmup(self) -> None: ...

    def evaluate(
        self,
        context: Any,
        schema: Sequence[BasePrimitive],
        *,
        arbiter: BaseArbiter | None = None,
        score_offsets: Mapping[str, Sequence[float]] | None = None,
        temperatures: Mapping[str, float] | None = None,
        allow_empty_context: bool = False,
    ) -> EvaluationResult: ...

    def evaluate_batch(
        self,
        contexts: Sequence[Any],
        schema: Sequence[BasePrimitive],
        *,
        arbiter: BaseArbiter | None = None,
        score_offsets: Mapping[str, Sequence[float]] | None = None,
        temperatures: Mapping[str, float] | None = None,
    ) -> list[EvaluationResult]: ...
