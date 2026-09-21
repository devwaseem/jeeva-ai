import pytest

from jeeva.core.engine import DecisionEngine

_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_NO_LIVE_MODEL_TESTS = pytest.mark.skip(
    reason="Jeeva's release policy prohibits live pretrained-model execution."
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Prevent local and CI runs from downloading or executing pretrained models."""
    for item in items:
        if item.get_closest_marker("model_integration") is not None:
            item.add_marker(_NO_LIVE_MODEL_TESTS)
        if item.get_closest_marker("training_integration") is not None:
            item.add_marker(_NO_LIVE_MODEL_TESTS)


@pytest.fixture(scope="session")
def qwen_engine() -> DecisionEngine:
    """Load the shared real-model inference engine once per integration session."""
    return DecisionEngine(
        model_id=_MODEL_ID,
        default_cascade_threshold=0.30,
    )


@pytest.fixture(scope="session")
def qwen_single_order_engine(qwen_engine: DecisionEngine) -> DecisionEngine:
    """Reuse the loaded model with candidate-position balancing disabled."""
    return DecisionEngine.from_components(
        model=qwen_engine.model,
        tokenizer=qwen_engine.tokenizer,
        processor=qwen_engine.processor,
        model_id=qwen_engine.model_id,
        device=qwen_engine.device,
        dtype=qwen_engine.torch_dtype,
        default_cascade_threshold=0.30,
        candidate_position_balancing=False,
    )


@pytest.fixture(scope="session")
def qwen_dense_engine(qwen_engine: DecisionEngine) -> DecisionEngine:
    """Reuse the loaded model with dense decision readout state."""
    return DecisionEngine.from_components(
        model=qwen_engine.model,
        tokenizer=qwen_engine.tokenizer,
        processor=qwen_engine.processor,
        model_id=qwen_engine.model_id,
        device=qwen_engine.device,
        dtype=qwen_engine.torch_dtype,
        default_cascade_threshold=0.30,
        mode="dense",
    )
