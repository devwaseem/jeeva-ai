import pytest

from jeeva.schema.primitives import DecisionResult
from tests.schema_helpers import choice, noul, schema

pytestmark = pytest.mark.model_integration


def test_default_engine_mode_is_letter(qwen_engine):
    assert qwen_engine.mode == "letter"


def test_global_readout_mode_applies_to_all_fields(qwen_engine, qwen_single_order_engine):
    context = "The customer was charged twice and demands an immediate refund."
    options = ["Technical Support", "Billing & Invoicing", "Account Security"]
    decision_schema = schema(
        category=choice(options, criterion="Select ticket category."),
        is_dissatisfied=noul("The customer expresses dissatisfaction."),
    )
    balanced_result = qwen_engine.evaluate(
        context=context,
        schema=decision_schema,
        mode="letter",
    )
    single_order_result = qwen_single_order_engine.evaluate(
        context=context,
        schema=decision_schema,
        mode="letter",
    )
    dense_result = qwen_engine.evaluate(context=context, schema=decision_schema, mode="dense")
    assert balanced_result.telemetry["readout_modes"] == {
        "category": "letter",
        "is_dissatisfied": "letter",
    }
    assert dense_result.telemetry["readout_modes"] == {
        "category": "dense",
        "is_dissatisfied": "dense",
    }
    assert balanced_result.telemetry["num_lanes"] == 2
    assert balanced_result.telemetry["num_branches"] == 5
    assert balanced_result.telemetry["candidate_position_balancing"] is True
    assert balanced_result.telemetry["candidate_ordering"]["category"]["trials"] == 3
    assert balanced_result.telemetry["candidate_ordering"]["is_dissatisfied"]["trials"] == 2
    assert single_order_result.telemetry["num_lanes"] == 2
    assert single_order_result.telemetry["num_branches"] == 2
    assert single_order_result.telemetry["candidate_position_balancing"] is False
    assert single_order_result.telemetry["candidate_ordering"] == {}


def test_letter_readout_choice_and_noul(qwen_engine):
    ticket = "I was double-billed for my subscription last month. Please refund invoice #49281 immediately."
    decision_schema = schema(
        category=choice(
            ["Technical Support", "Billing & Invoicing", "Account Security"],
            criterion="Select ticket category.",
        ),
        is_urgent=noul("The request requires urgent attention."),
    )

    result = qwen_engine.evaluate(context=ticket, schema=decision_schema)

    assert "category" in result.decisions
    assert "is_urgent" in result.decisions

    cat = result.decisions["category"]
    assert isinstance(cat, DecisionResult)
    assert cat.primitive == "choice"
    # Result value must be the domain option string, not the raw letter
    assert cat.value in ["Technical Support", "Billing & Invoicing", "Account Security"]
    assert cat.value == "Billing & Invoicing"
    assert "Billing & Invoicing" in cat.distribution
    assert isinstance(cat.is_certain, bool)
    assert cat.margin >= 0.0

    urg = result.decisions["is_urgent"]
    assert isinstance(urg, DecisionResult)
    assert urg.primitive == "noul"
    # Result value must be a typed boolean
    assert isinstance(urg.value, bool)
    assert urg.value is True
    assert "true" in urg.distribution
    assert "false" in urg.distribution


def test_letter_readout_multi_option_choice(qwen_engine):
    context = "The system latency spikes to 800ms during peak database write operations."
    decision_schema = schema(
        component=choice(
            [
                "Frontend UI",
                "Database Layer",
                "Network Gateway",
                "Auth Service",
                "Worker Queue",
            ],
            criterion="Identify the bottleneck subsystem.",
        )
    )

    result = qwen_engine.evaluate(context=context, schema=decision_schema)
    comp = result.decisions["component"]
    assert comp.value in [
        "Frontend UI",
        "Database Layer",
        "Network Gateway",
        "Auth Service",
        "Worker Queue",
    ]
    assert len(comp.distribution) == 5


def test_letter_readout_mode_override(qwen_engine):
    ticket = "I need help resetting my account password."
    decision_schema = schema(
        category=choice(["Billing", "Security", "Support"], criterion="Categorize.")
    )

    # Evaluate with explicit mode override
    res_dense = qwen_engine.evaluate(context=ticket, schema=decision_schema, mode="dense")
    res_letter = qwen_engine.evaluate(context=ticket, schema=decision_schema, mode="letter")

    assert res_dense.decisions["category"].value in ["Billing", "Security", "Support"]
    assert res_letter.decisions["category"].value in ["Billing", "Security", "Support"]
