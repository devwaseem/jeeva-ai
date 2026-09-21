import asyncio
import threading
from collections.abc import Sequence

import httpx
import pytest
from fastapi.testclient import TestClient

from jeeva.arbiters import BaseArbiter
from jeeva.model import Decision, FieldDecision
from jeeva.schema import DecisionSchema, decision_schema_from_dict
from jeeva.serve import create_app

SCHEMA_DEFINITION = {
    "kind": {
        "type": "choice",
        "instructions": "Choose a kind.",
        "criteria": {"a": "Kind A.", "b": "Kind B."},
    }
}
SCHEMA = decision_schema_from_dict(SCHEMA_DEFINITION, name="ServerSchema")
AUTHORIZATION = {"Authorization": "Bearer test-secret"}


def _decision(context: str) -> Decision:
    field = FieldDecision(
        value="a",
        distribution={"a": 0.8, "b": 0.2},
        margin=0.6,
        certainty_score=0.6,
        score=0.8,
        entropy=0.5,
        normalized_entropy=0.5,
        winning_level=None,
        standard_deviation=None,
        is_certain=True,
        calibration="validated",
        tau=0.3,
        threshold_source="calibrated",
        resolved_by="gatekeeper",
        reasoning=f"Resolved {context}",
    )
    return Decision(
        value={"kind": "a"},
        fields={"kind": field},
        is_certain=True,
        calibration="validated",
        telemetry={"total_ms": 1.0, "device": "secret-device"},
    )


class FakeModel:
    def __init__(self) -> None:
        self.batch_calls = 0
        self.failure: Exception | None = None

    def decide(
        self,
        context: str,
        schema: type[DecisionSchema],
        *,
        arbiter: BaseArbiter | None = None,
    ) -> Decision:
        if self.failure is not None:
            raise self.failure
        assert schema is SCHEMA
        return _decision(context)

    def decide_many(
        self,
        contexts: Sequence[str],
        schema: type[DecisionSchema],
        *,
        arbiter: BaseArbiter | None = None,
    ) -> list[Decision]:
        self.batch_calls += 1
        if self.failure is not None:
            raise self.failure
        assert schema is SCHEMA
        return [_decision(context) for context in contexts]


class BlockingModel(FakeModel):
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.started = started
        self.release = release

    def decide(
        self,
        context: str,
        schema: type[DecisionSchema],
        *,
        arbiter: BaseArbiter | None = None,
    ) -> Decision:
        self.started.set()
        self.release.wait(timeout=5)
        return super().decide(context, schema, arbiter=arbiter)


def test_server_requires_authentication_configuration():
    with pytest.raises(ValueError, match="JEEVA_API_KEY"):
        create_app(model=FakeModel(), schema=SCHEMA)


def test_server_requires_valid_bearer_token():
    client = TestClient(create_app(model=FakeModel(), schema=SCHEMA, api_key="test-secret"))

    missing = client.post("/v1/decide", json={"context": "one"})
    incorrect = client.post(
        "/v1/decide",
        json={"context": "one"},
        headers={"Authorization": "Bearer wrong"},
    )
    accepted = client.post(
        "/v1/decide",
        json={"context": "one"},
        headers=AUTHORIZATION,
    )

    assert missing.status_code == 401
    assert incorrect.status_code == 401
    assert accepted.status_code == 200


def test_server_uses_fixed_schema_and_minimal_response_by_default():
    client = TestClient(create_app(model=FakeModel(), schema=SCHEMA, api_key="test-secret"))

    response = client.post(
        "/v1/decide",
        json={"context": "one"},
        headers=AUTHORIZATION,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["value"] == {"kind": "a"}
    assert payload["fields"]["kind"] == {
        "value": "a",
        "is_certain": True,
        "calibration": "validated",
        "tau": 0.3,
        "threshold_source": "calibrated",
        "resolved_by": "gatekeeper",
    }
    assert "telemetry" not in payload
    assert "distribution" not in payload["fields"]["kind"]

    dynamic_schema = client.post(
        "/v1/decide",
        json={"context": "one", "schema": SCHEMA_DEFINITION},
        headers=AUTHORIZATION,
    )
    assert dynamic_schema.status_code == 422


def test_server_returns_full_evidence_only_when_requested():
    client = TestClient(create_app(model=FakeModel(), schema=SCHEMA, api_key="test-secret"))

    response = client.post(
        "/v1/decide?include_evidence=true",
        json={"context": "one"},
        headers=AUTHORIZATION,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fields"]["kind"]["distribution"] == {"a": 0.8, "b": 0.2}
    assert payload["fields"]["kind"]["reasoning"] == "Resolved one"
    assert payload["telemetry"]["total_ms"] == 1.0


def test_health_does_not_disclose_model_or_device():
    client = TestClient(create_app(model=FakeModel(), schema=SCHEMA, api_key="test-secret"))

    response = client.get("/health")

    assert response.status_code == 200
    assert set(response.json()) == {"status", "version", "uptime_seconds"}


def test_server_enforces_batch_and_context_limits_before_inference():
    model = FakeModel()
    client = TestClient(
        create_app(
            model=model,
            schema=SCHEMA,
            api_key="test-secret",
            max_batch_size=1,
            max_context_chars=3,
        )
    )

    batch = client.post(
        "/v1/batch",
        json={"contexts": ["one", "two"]},
        headers=AUTHORIZATION,
    )
    context = client.post(
        "/v1/decide",
        json={"context": "long"},
        headers=AUTHORIZATION,
    )

    assert batch.status_code == 400
    assert context.status_code == 400
    assert model.batch_calls == 0


def test_server_rejects_oversized_or_lengthless_request_bodies():
    client = TestClient(
        create_app(
            model=FakeModel(),
            schema=SCHEMA,
            api_key="test-secret",
            max_body_bytes=20,
        )
    )

    oversized = client.post(
        "/v1/decide",
        content=b'{"context":"this body is too large"}',
        headers={**AUTHORIZATION, "Content-Type": "application/json"},
    )

    assert oversized.status_code == 413


def test_server_batch_uses_public_decide_many():
    model = FakeModel()
    client = TestClient(create_app(model=model, schema=SCHEMA, api_key="test-secret"))

    response = client.post(
        "/v1/batch",
        json={"contexts": ["one", "two"]},
        headers=AUTHORIZATION,
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
    assert model.batch_calls == 1


def test_server_rejects_requests_when_inference_queue_is_full():
    async def exercise() -> None:
        started = threading.Event()
        release = threading.Event()
        app = create_app(
            model=BlockingModel(started, release),
            schema=SCHEMA,
            api_key="test-secret",
            max_pending_requests=1,
        )
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(
                client.post("/v1/decide", json={"context": "one"}, headers=AUTHORIZATION)
            )
            await asyncio.to_thread(started.wait, 5)

            overloaded = await client.post(
                "/v1/decide",
                json={"context": "two"},
                headers=AUTHORIZATION,
            )
            assert overloaded.status_code == 503

            release.set()
            completed = await first
            assert completed.status_code == 200

    asyncio.run(exercise())


def test_server_hides_internal_error_details():
    model = FakeModel()
    model.failure = RuntimeError("provider-secret")
    client = TestClient(create_app(model=model, schema=SCHEMA, api_key="test-secret"))

    response = client.post(
        "/v1/decide",
        json={"context": "one"},
        headers=AUTHORIZATION,
    )

    assert response.status_code == 500
    assert "provider-secret" not in response.text
    assert "Reference:" in response.text
