"""Authenticated HTTP serving for one registered Jeeva decision schema."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from jeeva import __version__
from jeeva.arbiters import BaseArbiter
from jeeva.model import Decision
from jeeva.schema import DecisionSchema

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover - exercised only without the serve extra
    raise ImportError(
        "Serving requires the optional dependencies. Install Jeeva with `pip install 'jeeva[serve]'`."
    ) from exc

logger = logging.getLogger(__name__)


class _DecisionProvider(Protocol):
    def decide(
        self,
        context: str,
        schema: type[DecisionSchema],
        *,
        arbiter: BaseArbiter | None = None,
    ) -> Decision: ...

    def decide_many(
        self,
        contexts: Sequence[str],
        schema: type[DecisionSchema],
        *,
        arbiter: BaseArbiter | None = None,
    ) -> list[Decision]: ...


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: str = Field(min_length=1)


class BatchDecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contexts: list[str] = Field(min_length=1)


class FieldResponse(BaseModel):
    value: bool | str | float
    is_certain: bool
    calibration: str
    tau: float
    threshold_source: str
    resolved_by: str
    distribution: dict[str, float] | None = None
    margin: float | None = None
    certainty_score: float | None = None
    score: float | None = None
    entropy: float | None = None
    normalized_entropy: float | None = None
    winning_level: float | None = None
    standard_deviation: float | None = None
    reasoning: str | None = None


class DecideResponse(BaseModel):
    value: dict[str, bool | str | float]
    fields: dict[str, FieldResponse]
    is_certain: bool
    is_ambiguous: bool
    calibration: str
    telemetry: dict[str, Any] | None = None


class BatchDecideResponse(BaseModel):
    results: list[DecideResponse]


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


def _serialize_decision(decision: Decision, *, include_evidence: bool) -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    for path, field in decision.fields.items():
        payload: dict[str, Any] = {
            "value": field.value,
            "is_certain": field.is_certain,
            "calibration": field.calibration,
            "tau": field.tau,
            "threshold_source": field.threshold_source,
            "resolved_by": field.resolved_by,
        }
        if include_evidence:
            payload.update(
                {
                    "distribution": dict(field.distribution),
                    "margin": field.margin,
                    "certainty_score": field.certainty_score,
                    "score": field.score,
                    "entropy": field.entropy,
                    "normalized_entropy": field.normalized_entropy,
                    "winning_level": field.winning_level,
                    "standard_deviation": field.standard_deviation,
                    "reasoning": field.reasoning,
                }
            )
        fields[path] = payload

    result: dict[str, Any] = {
        "value": dict(decision.value),
        "fields": fields,
        "is_certain": decision.is_certain,
        "is_ambiguous": decision.is_ambiguous,
        "calibration": decision.calibration,
    }
    if include_evidence:
        result["telemetry"] = dict(decision.telemetry)
    return result


def create_app(
    *,
    model: _DecisionProvider,
    schema: type[DecisionSchema],
    arbiter: BaseArbiter | None = None,
    api_key: str | None = None,
    allow_unauthenticated: bool = False,
    max_batch_size: int = 32,
    max_context_chars: int = 100_000,
    max_pending_requests: int = 64,
    max_body_bytes: int = 1_000_000,
    enable_docs: bool = False,
) -> FastAPI:
    """Create a bounded server for one model and one startup-registered schema."""
    if not isinstance(schema, type) or not issubclass(schema, DecisionSchema):
        raise TypeError("schema must be a DecisionSchema subclass.")
    schema.fields()
    for name, value in {
        "max_batch_size": max_batch_size,
        "max_context_chars": max_context_chars,
        "max_pending_requests": max_pending_requests,
        "max_body_bytes": max_body_bytes,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1.")

    configured_key = api_key or os.environ.get("JEEVA_API_KEY")
    if not allow_unauthenticated and not configured_key:
        raise ValueError(
            "Set JEEVA_API_KEY (or pass api_key) before serving. "
            "For deliberate local-only use, pass allow_unauthenticated=True."
        )

    started_at = time.monotonic()
    inference_slots = asyncio.Semaphore(max_pending_requests)
    docs_url = "/docs" if enable_docs else None
    app = FastAPI(
        title="Jeeva Decision API",
        version=__version__,
        docs_url=docs_url,
        redoc_url=None,
        openapi_url="/openapi.json" if enable_docs else None,
    )

    @app.middleware("http")
    async def enforce_body_limit(request: Request, call_next: Callable[..., Any]):
        if request.method in {"POST", "PUT", "PATCH"}:
            raw_length = request.headers.get("content-length")
            if raw_length is None:
                return JSONResponse(
                    status_code=411,
                    content={"detail": "Content-Length is required."},
                )
            try:
                content_length = int(raw_length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Content-Length must be an integer."},
                )
            if content_length < 0:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Content-Length must not be negative."},
                )
            if content_length > max_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body is too large."},
                )
        return await call_next(request)

    async def authenticate(authorization: str | None = Header(default=None)) -> None:
        if allow_unauthenticated:
            return
        prefix = "Bearer "
        if (
            authorization is None
            or not authorization.startswith(prefix)
            or configured_key is None
            or not secrets.compare_digest(authorization[len(prefix) :], configured_key)
        ):
            raise HTTPException(
                status_code=401,
                detail="Valid bearer authentication is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def validate_context(context: str) -> None:
        if len(context) > max_context_chars:
            raise HTTPException(
                status_code=400,
                detail=f"Context exceeds the {max_context_chars}-character limit.",
            )

    async def run_inference(operation: Callable[[], Any]) -> Any:
        try:
            await asyncio.wait_for(inference_slots.acquire(), timeout=0.01)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=503,
                detail="Inference capacity is full; retry later.",
            ) from exc
        try:
            return await asyncio.to_thread(operation)
        except HTTPException:
            raise
        except Exception as exc:
            reference = uuid.uuid4().hex
            logger.exception("Inference failed (reference=%s)", reference)
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed. Reference: {reference}",
            ) from exc
        finally:
            inference_slots.release()

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            uptime_seconds=round(time.monotonic() - started_at, 3),
        )

    @app.post(
        "/v1/decide",
        response_model=DecideResponse,
        response_model_exclude_none=True,
        dependencies=[Depends(authenticate)],
    )
    async def decide(
        request: DecideRequest,
        include_evidence: bool = Query(default=False),
    ) -> Mapping[str, Any]:
        validate_context(request.context)
        decision = await run_inference(
            lambda: model.decide(request.context, schema, arbiter=arbiter)
        )
        return _serialize_decision(decision, include_evidence=include_evidence)

    @app.post(
        "/v1/batch",
        response_model=BatchDecideResponse,
        response_model_exclude_none=True,
        dependencies=[Depends(authenticate)],
    )
    async def decide_batch(
        request: BatchDecideRequest,
        include_evidence: bool = Query(default=False),
    ) -> Mapping[str, Any]:
        if len(request.contexts) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch exceeds the {max_batch_size}-context limit.",
            )
        for context in request.contexts:
            validate_context(context)
        decisions = await run_inference(
            lambda: model.decide_many(request.contexts, schema, arbiter=arbiter)
        )
        return {
            "results": [
                _serialize_decision(decision, include_evidence=include_evidence)
                for decision in decisions
            ]
        }

    return app


def run_server(
    *,
    model: _DecisionProvider,
    schema: type[DecisionSchema],
    host: str = "127.0.0.1",
    port: int = 8000,
    arbiter: BaseArbiter | None = None,
    api_key: str | None = None,
    allow_unauthenticated: bool = False,
    max_batch_size: int = 32,
    max_context_chars: int = 100_000,
    max_pending_requests: int = 64,
    max_body_bytes: int = 1_000_000,
    enable_docs: bool = False,
) -> None:
    """Run the built-in HTTP server with Uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised only without the serve extra
        raise ImportError(
            "Serving requires uvicorn. Install Jeeva with `pip install 'jeeva[serve]'`."
        ) from exc

    app = create_app(
        model=model,
        schema=schema,
        arbiter=arbiter,
        api_key=api_key,
        allow_unauthenticated=allow_unauthenticated,
        max_batch_size=max_batch_size,
        max_context_chars=max_context_chars,
        max_pending_requests=max_pending_requests,
        max_body_bytes=max_body_bytes,
        enable_docs=enable_docs,
    )
    uvicorn.run(app, host=host, port=port)
