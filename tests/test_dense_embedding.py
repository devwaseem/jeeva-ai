import pytest
import torch

from tests.schema_helpers import choice, noul, schema

pytestmark = pytest.mark.model_integration


def test_dense_embedding_choice_and_noul(qwen_dense_engine):
    decision_schema = schema(
        category=choice(
            ["Technical Support", "Billing & Invoicing", "Account Security"],
            criterion="Select ticket category.",
        ),
        escalate=noul("The request requires urgent human escalation."),
    )

    doc = "I was double-billed for my subscription last month. Please refund invoice #49281 immediately."

    res = qwen_dense_engine.evaluate(
        context=doc,
        schema=decision_schema,
    )

    assert "category" in res.decisions
    assert "escalate" in res.decisions
    cat_res = res.decisions["category"]

    assert cat_res.value == "Billing & Invoicing"
    assert cat_res.raw_score > 0.0
    assert cat_res.margin >= 0.0
    assert isinstance(cat_res.is_certain, bool)


def test_dense_embedding_option_caching(qwen_dense_engine):
    choice_primitive = choice(["Option Alpha", "Option Beta", "Option Gamma"])
    emb1 = qwen_dense_engine._get_or_compute_option_embeddings(choice_primitive)
    emb2 = qwen_dense_engine._get_or_compute_option_embeddings(choice_primitive)

    # Must return exact same cached tensor object
    assert emb1 is emb2
    assert emb1.shape == (3, 896)
    # Must be unit-normalized
    norms = torch.norm(emb1, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_dense_embedding_continuous_classification(qwen_dense_engine):
    decision_schema = schema(topic=choice(["Refund", "Bug Report", "Feature Request"]))
    doc = "Found a NullPointerException in the checkout API endpoint when payload is empty."

    # Dense embedding understands the continuous semantic meaning of 'Bug Report' directly
    res = qwen_dense_engine.evaluate(
        context=doc,
        schema=decision_schema,
    )
    assert res.decisions["topic"].value in ["Refund", "Bug Report", "Feature Request"]
    assert res.decisions["topic"].resolved_by == "gatekeeper"
    assert res.decisions["topic"].raw_score > 0.0


def test_dense_embedding_supports_memory_efficient_cache(qwen_dense_engine):
    decision_schema = schema(topic=choice(["Refund", "Bug Report"]))
    context = "The checkout API raises an exception for empty payloads."
    previous = qwen_dense_engine.contiguous_cache
    try:
        qwen_dense_engine.contiguous_cache = True
        contiguous = qwen_dense_engine.evaluate(
            context=context,
            schema=decision_schema,
        )
        qwen_dense_engine.contiguous_cache = False
        memory_efficient = qwen_dense_engine.evaluate(
            context=context,
            schema=decision_schema,
        )
    finally:
        qwen_dense_engine.contiguous_cache = previous

    assert memory_efficient.decisions["topic"].value == contiguous.decisions["topic"].value
    assert memory_efficient.decisions["topic"].distribution == pytest.approx(
        contiguous.decisions["topic"].distribution,
        abs=1e-5,
    )
