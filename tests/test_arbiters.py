from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from jeeva.arbiters import BaseArbiter, SequenceArbiter, SequenceThinkingArbiter
from jeeva.arbiters.thinking import (
    _build_thinking_prompt,
    _ends_with_token_sequence,
    _sample_next_token,
)
from jeeva.schema.primitives import DecisionResult
from tests.schema_helpers import choice, schema


@pytest.fixture(scope="module")
def engine(qwen_engine):
    return qwen_engine


def category_schema(options):
    return schema(category=choice(options, criterion="Select ticket category."))


@pytest.mark.model_integration
def test_gatekeeper_without_arbiter(engine):
    ticket = "I was double-billed for my subscription last month. Please refund invoice #49281 immediately."
    decision_schema = category_schema(
        ["Billing & Invoicing", "Technical Support", "Account Security"]
    )

    result = engine.evaluate(
        context=ticket, schema=decision_schema, cascade_threshold=0.20, arbiter=None
    )
    res = result.decisions["category"]

    assert res.is_certain is True
    assert res.is_ambiguous is False
    assert res.resolved_by == "gatekeeper"
    assert result.telemetry["arbiter_invoked_fields"] == []


@pytest.mark.model_integration
def test_sequence_arbiter_dormant_on_certainty(engine):
    ticket = "I was double-billed for my subscription last month. Please refund invoice #49281 immediately."
    decision_schema = category_schema(
        ["Billing & Invoicing", "Technical Support", "Account Security"]
    )

    # Standard low threshold: Gatekeeper is certain, Arbiter remains dormant
    arbiter = SequenceArbiter()
    result = engine.evaluate(
        context=ticket, schema=decision_schema, cascade_threshold=0.20, arbiter=arbiter
    )
    res = result.decisions["category"]

    assert res.is_certain is True
    assert res.resolved_by == "gatekeeper"
    assert result.telemetry["arbiter_invoked_fields"] == []


@pytest.mark.model_integration
def test_sequence_arbiter_resolution_on_ambiguity(engine):
    ticket = "My account was locked right after I authorized the payment renewal."
    decision_schema = category_schema(
        ["Billing & Refunds", "Account Security", "Technical Support"]
    )

    # Force a very strict threshold (0.99) so Gatekeeper is flagged ambiguous
    # and SequenceArbiter Cross-Encoder is triggered
    arbiter = SequenceArbiter()
    result = engine.evaluate(
        context=ticket, schema=decision_schema, cascade_threshold=0.99, arbiter=arbiter
    )
    res = result.decisions["category"]

    assert "category" in result.telemetry["arbiter_invoked_fields"]
    assert res.resolved_by == "SequenceArbiter"
    assert res.value in ["Billing & Refunds", "Account Security", "Technical Support"]
    assert result.telemetry["arbiter_ms"] > 0
    assert isinstance(res.is_certain, bool)
    assert 0.0 <= res.raw_score <= 1.0
    assert 0.0 <= res.margin <= 1.0


@pytest.mark.model_integration
def test_custom_pluggable_arbiter(engine):
    class MockCloudArbiter(BaseArbiter):
        @property
        def name(self) -> str:
            return "MockCloudArbiter"

        def resolve(
            self,
            context,
            ambiguous_fields,
            current_decisions,
            engine,
            base_cache,
            past_len,
            threshold,
        ):
            res = {}
            for k, _prim in ambiguous_fields.items():
                res[k] = DecisionResult(
                    field=k,
                    primitive="choice",
                    value="Account Security",
                    is_certain=True,
                    margin=0.99,
                    raw_score=0.995,
                    resolved_by=self.name,
                )
            return res

    ticket = "Unknown state with vague terms."
    decision_schema = category_schema(["Option A", "Option B"])

    custom_arbiter = MockCloudArbiter()
    result = engine.evaluate(
        context=ticket,
        schema=decision_schema,
        cascade_threshold=0.99,
        arbiter=custom_arbiter,
    )
    res = result.decisions["category"]

    assert res.resolved_by == "MockCloudArbiter"
    assert res.value == "Account Security"
    assert res.is_certain is True
    assert "category" in result.telemetry["arbiter_invoked_fields"]


@pytest.mark.model_integration
def test_sequence_thinking_arbiter(engine):
    ticket = "My account was locked right after I authorized the payment renewal."
    decision_schema = category_schema(
        ["Billing & Refunds", "Account Security", "Technical Support"]
    )

    # Strict threshold triggers thinking arbiter
    thinking_arbiter = SequenceThinkingArbiter(
        max_thinking_tokens=32,
        temperature=0.0,
        readout="sequence",
    )
    result = engine.evaluate(
        context=ticket,
        schema=decision_schema,
        cascade_threshold=0.99,
        arbiter=thinking_arbiter,
    )
    res = result.decisions["category"]

    assert "category" in result.telemetry["arbiter_invoked_fields"]
    assert res.resolved_by == "SequenceThinkingArbiter"
    assert res.value in ["Billing & Refunds", "Account Security", "Technical Support"]
    assert res.reasoning is not None
    assert len(res.reasoning) > 0
    assert result.telemetry["arbiter_ms"] > 0

    # Verify to_dict contains reasoning
    d = res.to_dict()
    assert "reasoning" in d
    assert d["reasoning"] == res.reasoning
    assert d["resolved_by"] == "SequenceThinkingArbiter"


@pytest.mark.model_integration
def test_thinking_arbiter_can_run_on_certain_fields(engine):
    decision_schema = category_schema(
        ["Billing & Refunds", "Account Security", "Technical Support"]
    )
    arbiter = SequenceThinkingArbiter(
        always_think=True,
        max_thinking_tokens=8,
    )

    result = engine.evaluate(
        context="The account was locked after a password reset.",
        schema=decision_schema,
        cascade_threshold=0.0,
        arbiter=arbiter,
    )

    assert arbiter.resolve_all_fields is True
    assert result.decisions["category"].resolved_by == "SequenceThinkingArbiter"
    assert result.telemetry["arbiter_invoked_fields"] == ["category"]


def test_thinking_sampling_is_deterministic_by_default_and_honors_nucleus_filter():
    arbiter = SequenceThinkingArbiter()
    logits = torch.tensor([[5.0, 4.0, 3.0]])

    assert arbiter.temperature == 0.0
    assert arbiter.resolve_all_fields is False
    assert arbiter.readout == "letter"
    assert arbiter.candidate_position_balancing is True
    assert (
        SequenceThinkingArbiter(candidate_position_balancing=False).candidate_position_balancing
        is False
    )
    with pytest.raises(ValueError, match="readout"):
        SequenceThinkingArbiter(readout="freeform")  # type: ignore[arg-type]
    assert _sample_next_token(logits, temperature=0.7, top_p=0.01).item() == 0


def test_thinking_letter_readout_balances_candidate_positions():
    class Inputs:
        input_ids = torch.tensor([[1], [1]])
        attention_mask = torch.ones((2, 1), dtype=torch.long)

        def to(self, _device):
            return self

    class Tokenizer:
        def __init__(self):
            self.prompts = []

        def __call__(self, prompts, **_kwargs):
            self.prompts = prompts
            return Inputs()

        def encode(self, letter, add_special_tokens=False):
            assert add_special_tokens is False
            return {"A": [0], "B": [1]}[letter]

    class Model:
        def __call__(self, **_kwargs):
            logits = torch.tensor(
                [
                    [[0.0, 2.0, -1.0]],
                    [[0.0, 2.0, -1.0]],
                ]
            )
            return SimpleNamespace(logits=logits)

    arbiter = SequenceThinkingArbiter()
    tokenizer = Tokenizer()
    cache = SimpleNamespace(get_seq_length=lambda: 4)
    with patch(
        "jeeva.arbiters.thinking.replicate_dynamic_cache",
        return_value=object(),
    ):
        aggregate = arbiter._score_letter_candidates(
            model=Model(),
            tokenizer=tokenizer,
            target_device="cpu",
            think_cache=cache,
            options=["first", "second"],
            use_contiguous_cache=False,
        )

    assert tokenizer.prompts == [
        "\nOptions:\nA. first\nB. second\nAnswer: ",
        "\nOptions:\nA. second\nB. first\nAnswer: ",
    ]
    assert torch.equal(aggregate.scores, torch.tensor([1.0, 1.0]))
    assert aggregate.robust_margin == 0.0


def test_thinking_prompt_requests_evidence_without_a_verdict():
    prompt = _build_thinking_prompt(
        branch_suffix="\nAssistant:",
        prompt_label=(
            "Examples:\n"
            "- Yes: The document includes a passport number.\n"
            "\n"
            "Question:\n"
            "Whether the document contains PII"
        ),
        options=["Yes", "No"],
    )

    assert prompt.index("Examples:") < prompt.index("Question:")
    assert "Field: '" not in prompt
    assert "Candidate choices: 'Yes', 'No'" in prompt
    assert "Do not state a final verdict" in prompt
    assert "End after the evidence analysis" in prompt
    assert "Choose the candidate" not in prompt


def test_thinking_stop_requires_the_complete_token_sequence():
    assert _ends_with_token_sequence([1, 2, 3], [2, 3]) is True
    assert _ends_with_token_sequence([1, 2], [2, 3]) is False
    assert _ends_with_token_sequence([1, 9, 3], [2, 3]) is False


@pytest.mark.model_integration
def test_sequence_arbiter_with_separate_model(engine):
    ticket = "My account was locked right after I authorized the payment renewal."
    decision_schema = category_schema(
        ["Billing & Refunds", "Account Security", "Technical Support"]
    )

    # Pass an explicit model_or_id to test separate model resolution
    arbiter = SequenceArbiter(model_or_id="Qwen/Qwen2.5-0.5B-Instruct")
    result = engine.evaluate(
        context=ticket, schema=decision_schema, cascade_threshold=0.99, arbiter=arbiter
    )
    res = result.decisions["category"]

    assert res.resolved_by == "SequenceArbiter"
    assert res.value in ["Billing & Refunds", "Account Security", "Technical Support"]
    assert arbiter._model is not None
    assert arbiter._tokenizer is not None


@pytest.mark.model_integration
def test_sequence_thinking_arbiter_with_separate_model(engine):
    ticket = "My account was locked right after I authorized the payment renewal."
    decision_schema = category_schema(
        ["Billing & Refunds", "Account Security", "Technical Support"]
    )

    # Pass an explicit model_or_id to test separate thinking model resolution
    arbiter = SequenceThinkingArbiter(
        model_or_id="Qwen/Qwen2.5-0.5B-Instruct",
        max_thinking_tokens=32,
        temperature=0.0,
    )
    result = engine.evaluate(
        context=ticket, schema=decision_schema, cascade_threshold=0.99, arbiter=arbiter
    )
    res = result.decisions["category"]

    assert res.resolved_by == "SequenceThinkingArbiter"
    assert res.reasoning is not None
    assert arbiter._model is not None
