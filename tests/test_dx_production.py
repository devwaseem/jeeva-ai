"""Production-facing behavior for DecisionSchema, engine, async, and HTTP APIs."""

import pytest
import torch

from jeeva import Choice, DecisionSchema, Noul
from jeeva.core.engine import EvaluationResult
from jeeva.schema import compile_decision_schema
from jeeva.schema.primitives import DecisionResult, normalize_schema


class TicketTriage(DecisionSchema):
    department = Choice(
        "Choose the department that should handle the ticket.",
        {
            "Billing & Refunds": "Billing, invoice, charge, and refund requests.",
            "Technical Support": "Product failures and troubleshooting requests.",
            "Account Security": "Authentication and account-security requests.",
        },
    )
    is_urgent = Noul(
        "Does the ticket require emergency intervention today?",
        {
            True: "Emergency intervention is required today.",
            False: "The ticket can follow the normal response process.",
        },
    )
    threat_level = Choice(
        "Classify the severity of the legal, fraud, or churn threat.",
        {
            "low": "No material legal, fraud, or churn threat.",
            "medium": "A credible but non-immediate threat is present.",
            "critical": "An immediate severe legal, fraud, or churn threat is present.",
        },
    )


def test_schema_primitives_are_explicit_and_compiled_immutably():
    compiled = compile_decision_schema(TicketTriage)
    assert list(compiled.primitives) == ["department", "is_urgent", "threat_level"]
    assert compiled.primitives["department"].distribution_labels == [
        "Billing & Refunds",
        "Technical Support",
        "Account Security",
    ]
    assert TicketTriage.department.key is None
    assert compiled.primitives["department"].key == "department"
    assert normalize_schema(TicketTriage) == list(compiled.primitives.values())


def test_choice_returns_plain_string_values():
    primitive = TicketTriage.department.bind("department")
    result = primitive.parse_similarities(torch.tensor([5.0, 1.0, 0.5]))
    assert result.value == "Billing & Refunds"
    assert result.to_dict()["value"] == "Billing & Refunds"


def test_evaluation_result_dx():
    """Verify dictionary indexing and dot-attribute access on EvaluationResult."""
    decisions = {
        "category": DecisionResult(
            field="category",
            primitive="choice",
            value="Billing",
            is_certain=True,
            margin=0.8,
            raw_score=0.9,
        ),
        "is_urgent": DecisionResult(
            field="is_urgent",
            primitive="noul",
            value=True,
            is_certain=True,
            margin=0.95,
            raw_score=0.98,
        ),
    }
    res = EvaluationResult(decisions=decisions, telemetry={"total_ms": 15.2, "device": "cpu"})

    # 1. Attribute access
    assert res.category == "Billing"
    assert res.is_urgent is True
    assert res.is_certain is True
    assert res.is_ambiguous is False

    # 2. Dictionary indexing
    assert res["category"] == "Billing"
    assert res["is_urgent"] is True
    assert "category" in res
    assert res.get("category") == "Billing"
    assert res.get("nonexistent", "default") == "default"

    # 3. Iteration
    items = dict(res)
    assert items == {"category": "Billing", "is_urgent": True}

    # 5. Margins and distributions convenience properties
    assert res.margins == {"category": 0.8, "is_urgent": 0.95}
    assert "category" in res.distributions
    assert res.distributions["category"] == {}


@pytest.mark.model_integration
def test_input_validation(qwen_engine):
    with pytest.raises(ValueError, match="Context must be a non-empty string"):
        qwen_engine.evaluate(context="", schema=TicketTriage)
    with pytest.raises(ValueError, match="Context must be a non-empty string"):
        qwen_engine.evaluate(context="   ", schema=TicketTriage)


@pytest.mark.model_integration
def test_engine_warmup(qwen_engine):
    """Verify engine.warmup runs without errors."""
    qwen_engine.warmup()


@pytest.mark.model_integration
def test_decide_with_decision_schema_e2e(qwen_engine):
    context = "Please reverse the duplicate $49.99 subscription fee billed this morning."
    result = qwen_engine.decide(context=context, schema=TicketTriage)
    assert isinstance(result, EvaluationResult)
    assert result.department in TicketTriage.department.criteria
    assert isinstance(result.is_urgent, bool)
    assert result.threat_level in TicketTriage.threat_level.criteria
    assert result.telemetry["total_ms"] > 0


@pytest.mark.model_integration
def test_async_evaluate_and_decide(qwen_engine):
    """Verify engine async methods (adecide, aevaluate, acheck, achoose) execute without blocking."""
    import asyncio

    async def _run():
        context = "The server returns 500 error on checkout."

        class ServiceStatus(DecisionSchema):
            is_down = Noul(
                "The service is experiencing an outage.",
                {True: "The condition is satisfied.", False: "The condition is not satisfied."},
            )

        eval_res = await qwen_engine.aevaluate(context=context, schema=ServiceStatus)
        assert isinstance(eval_res, EvaluationResult)
        assert isinstance(eval_res.is_down, bool)

        is_down = await qwen_engine.acheck(context, "The service is down.")
        assert isinstance(is_down, bool)

        choice = await qwen_engine.achoose(
            context,
            "Classify the request.",
            {
                "Outage": "The service is unavailable or severely degraded.",
                "Feature Request": "The user is asking for new functionality.",
            },
        )
        assert choice in ["Outage", "Feature Request"]

        model_res = await qwen_engine.adecide(context=context, schema=TicketTriage)
        assert isinstance(model_res, EvaluationResult)

    asyncio.run(_run())


def test_dtype_resolution():
    """Verify dtype aliases, enum values, automatic selection, and validation."""
    from jeeva.types import DtypeType, resolve_dtype

    # resolve_dtype unit tests
    assert resolve_dtype("float16") == torch.float16
    assert resolve_dtype("fp16") == torch.float16
    assert resolve_dtype("bfloat16") == torch.bfloat16
    assert resolve_dtype("bf16") == torch.bfloat16
    assert resolve_dtype("float32") == torch.float32
    assert resolve_dtype("fp32") == torch.float32
    assert resolve_dtype(torch.bfloat16) == torch.bfloat16
    assert resolve_dtype(DtypeType.BFLOAT16) == torch.bfloat16
    assert resolve_dtype("auto", device="cuda") == torch.float16
    assert resolve_dtype("auto", device="cpu") == torch.float32

    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_dtype("int8")
