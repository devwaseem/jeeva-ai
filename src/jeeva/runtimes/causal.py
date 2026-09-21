from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from jeeva.arbiters.base import BaseArbiter
from jeeva.core.engine import DecisionEngine, EvaluationResult
from jeeva.heads.base import BaseDecisionHead
from jeeva.heads.metric import DECODER_HEAD_DIR, BilinearMetricHead
from jeeva.schema.primitives import BasePrimitive


class CausalRuntime:
    """Adapter preserving the existing shared-KV causal execution behavior."""

    architecture: Literal["causal"] = "causal"

    def __init__(
        self,
        source: str,
        *,
        device: str,
        dtype: str,
        scorer: str,
        candidate_position_balancing: bool = True,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        if scorer not in {"letter", "bilinear", "contextual"}:
            raise ValueError(f"Unsupported causal scorer: {scorer!r}.")
        self.scorer = scorer
        self.engine = DecisionEngine(
            model_id=source,
            device=device,
            dtype=dtype,
            mode="letter" if scorer == "letter" else "dense",
            candidate_position_balancing=(
                candidate_position_balancing if scorer == "letter" else False
            ),
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            local_files_only=local_files_only,
        )
        if scorer in {"bilinear", "contextual"}:
            model_config: Any = self.engine.model.config
            if hasattr(model_config, "get_text_config"):
                model_config = model_config.get_text_config(decoder=True)
            hidden_dim = int(model_config.hidden_size)
            loaded_head = BaseDecisionHead.from_pretrained(
                Path(source) / DECODER_HEAD_DIR,
                map_location=self.engine.device,
                expected_hidden_dim=hidden_dim,
            )
            if not isinstance(loaded_head, BilinearMetricHead):
                raise ValueError("Decoder artifact does not contain a bilinear metric head.")
            expected_strategy = "contextual" if scorer == "contextual" else "static"
            if loaded_head.input_strategy != expected_strategy:
                raise ValueError(
                    f"Decoder artifact uses {loaded_head.input_strategy!r} inputs; "
                    f"scorer {scorer!r} requires {expected_strategy!r} inputs."
                )
            self.engine.head = loaded_head.to(
                device=self.engine.device,
                dtype=self.engine.torch_dtype,
            )
            self.engine.head.eval()

    @classmethod
    def from_components(
        cls,
        *,
        model: Any,
        tokenizer: Any,
        processor: Any,
        source: str,
        device: str,
        dtype: Any,
        scorer: str,
        head: BilinearMetricHead | None,
        candidate_position_balancing: bool,
        resolved_model_id: str,
        resolved_model_revision: str | None,
        resolved_tokenizer_id: str,
        resolved_tokenizer_revision: str | None,
    ) -> CausalRuntime:
        if scorer not in {"letter", "bilinear", "contextual"}:
            raise ValueError(f"Unsupported causal scorer: {scorer!r}.")
        instance = cls.__new__(cls)
        instance.scorer = scorer
        instance.engine = DecisionEngine.from_components(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            model_id=source,
            device=device,
            dtype=dtype,
            mode="letter" if scorer == "letter" else "dense",
            head=head,
            candidate_position_balancing=(
                candidate_position_balancing if scorer == "letter" else False
            ),
        )
        instance.engine.head = head
        instance.engine.resolved_model_id = resolved_model_id
        instance.engine.resolved_model_revision = resolved_model_revision
        instance.engine.resolved_tokenizer_id = resolved_tokenizer_id
        instance.engine.resolved_tokenizer_revision = resolved_tokenizer_revision
        instance.engine.model.eval()
        if isinstance(instance.engine.head, BaseDecisionHead):
            instance.engine.head.eval()
        return instance

    @property
    def model(self) -> Any:
        return self.engine.model

    @property
    def tokenizer(self) -> Any:
        return self.engine.tokenizer

    @property
    def processor(self) -> Any:
        return self.engine.processor

    @property
    def is_multimodal(self) -> bool:
        return self.engine.is_multimodal

    @property
    def resolved_model_id(self) -> str:
        return self.engine.resolved_model_id

    @property
    def resolved_model_revision(self) -> str | None:
        return self.engine.resolved_model_revision

    @property
    def resolved_tokenizer_id(self) -> str:
        return self.engine.resolved_tokenizer_id

    @property
    def resolved_tokenizer_revision(self) -> str | None:
        return self.engine.resolved_tokenizer_revision

    def save_decoder_head(self, destination: str | Path) -> None:
        if self.scorer not in {"bilinear", "contextual"}:
            return
        head = self.engine.head
        if not isinstance(head, BilinearMetricHead):
            raise RuntimeError("Metric causal runtime is missing its decoder head.")
        head.save_pretrained(Path(destination) / DECODER_HEAD_DIR)

    def warmup(self) -> None:
        self.engine.warmup()

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
        del temperatures
        result = self.engine.evaluate(
            context=context,
            schema=schema,
            arbiter=arbiter,
            _score_offsets=score_offsets,
            _allow_empty_context=allow_empty_context,
        )
        result.telemetry["architecture"] = self.architecture
        return result

    def evaluate_batch(
        self,
        contexts: Sequence[Any],
        schema: Sequence[BasePrimitive],
        *,
        arbiter: BaseArbiter | None = None,
        score_offsets: Mapping[str, Sequence[float]] | None = None,
        temperatures: Mapping[str, float] | None = None,
    ) -> list[EvaluationResult]:
        del temperatures
        results = self.engine.evaluate_batch(
            contexts=contexts,
            schema=schema,
            arbiter=arbiter,
            _score_offsets=score_offsets,
        )
        for result in results:
            result.telemetry["architecture"] = self.architecture
        return results
