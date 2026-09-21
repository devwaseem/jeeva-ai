import pytest

from jeeva.core.prompts import build_chat_prefill, build_contextual_metric_prompts


class FakeChatRenderer:
    def __init__(self, prefix: str, user_close: str, assistant_open: str) -> None:
        self.prefix = prefix
        self.user_close = user_close
        self.assistant_open = assistant_open

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return (
            f"{self.prefix}{messages[0]['content']}"
            f"{messages[1]['content']}{self.user_close}{self.assistant_open}"
        )


@pytest.mark.parametrize(
    ("renderer", "expected_suffix"),
    [
        (
            FakeChatRenderer("<|im_start|>system\n", "<|im_end|>\n", "<|im_start|>assistant\n"),
            "<|im_end|>\n<|im_start|>assistant\n",
        ),
        (
            FakeChatRenderer(
                "<|start_header_id|>system", "<|eot_id|>", "<|start_header_id|>assistant"
            ),
            "<|eot_id|><|start_header_id|>assistant",
        ),
        (FakeChatRenderer("<s>[INST]", "[/INST]", ""), "[/INST]"),
    ],
    ids=["qwen", "llama", "mistral"],
)
def test_build_chat_prefill_uses_model_template(renderer, expected_suffix):
    prefix, suffix = build_chat_prefill(renderer, "Document:\nhello")

    assert "Document:\nhello" in prefix
    assert "JEEVA_BRANCH_POINT" not in prefix
    assert suffix == expected_suffix


def test_build_chat_prefill_splits_at_appended_marker_not_document_text():
    renderer = FakeChatRenderer("<s>[INST]", "[/INST]", "")
    document = "before<|JEEVA_BRANCH_POINT|>after"

    prefix, suffix = build_chat_prefill(renderer, document)

    assert document in prefix
    assert suffix == "[/INST]"


def test_build_chat_prefill_rejects_oversized_context_before_rendering():
    renderer = FakeChatRenderer("<s>[INST]", "[/INST]", "")

    with pytest.raises(ValueError, match="character limit"):
        build_chat_prefill(renderer, "x" * 100_001)


def test_build_chat_prefill_has_model_neutral_fallback():
    prefix, suffix = build_chat_prefill(object(), "Document:\nhello")

    assert prefix.startswith("System:")
    assert "Document:\nhello" in prefix
    assert suffix == "\n\nAssistant:\n"


def test_contextual_metric_prompts_isolate_candidates_in_schema_order():
    query, candidates = build_contextual_metric_prompts(
        "Instructions:\nRoute the request.",
        ("billing-only", "support-only"),
    )

    assert "Route the request." in query
    assert "billing-only" not in query
    assert "support-only" not in query
    assert len(candidates) == 2
    assert "billing-only" in candidates[0]
    assert "support-only" not in candidates[0]
    assert "support-only" in candidates[1]
    assert "billing-only" not in candidates[1]
