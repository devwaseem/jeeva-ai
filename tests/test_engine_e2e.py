import pytest

from tests.schema_helpers import choice, noul, schema

pytestmark = pytest.mark.model_integration


def test_engine_e2e_evaluation(qwen_engine):
    ticket = (
        "I was charged $100 twice on my credit card for the yearly renewal. "
        "Please issue a refund immediately or I will contact my credit card issuer."
    )

    decision_schema = schema(
        category=choice(
            ["Billing Inquiries", "Technical Troubleshooting", "Account Recovery"],
            criterion="Select ticket category.",
        ),
        is_urgent=noul("The request demands urgent remediation."),
    )

    result = qwen_engine.evaluate(context=ticket, schema=decision_schema)

    assert "category" in result.decisions
    assert "is_urgent" in result.decisions

    cat_res = result.decisions["category"]
    assert cat_res.primitive == "choice"
    assert cat_res.value in ["Billing Inquiries", "Technical Troubleshooting", "Account Recovery"]
    assert isinstance(cat_res.is_certain, bool)
    assert cat_res.margin >= 0.0
    assert cat_res.raw_score >= 0.0
    assert cat_res.resolved_by in ["gatekeeper", "SequenceArbiter"]

    urg_res = result.decisions["is_urgent"]
    assert urg_res.primitive == "noul"
    assert isinstance(urg_res.value, bool)
    assert urg_res.raw_score >= 0.0

    # Test values dictionary convenience property
    assert result.values["category"] == cat_res.value
    assert result.values["is_urgent"] == urg_res.value

    # Test telemetry presence
    assert "prefill_ms" in result.telemetry
    assert "dispatch_ms" in result.telemetry
    assert "total_ms" in result.telemetry
    assert result.telemetry["num_lanes"] == 2
