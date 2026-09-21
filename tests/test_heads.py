import json
import tempfile
from pathlib import Path

import pytest
import torch

from jeeva.heads.base import BaseDecisionHead
from jeeva.heads.linear import LinearDecisionHead
from jeeva.heads.metric import BilinearMetricHead
from jeeva.training.loss import BrierScoreLoss
from tests.schema_helpers import choice, schema


def test_linear_decision_head_basic():
    hidden_dim = 64
    labels = ["Tech Support", "Billing", "Account Security"]
    head = LinearDecisionHead(hidden_dim=hidden_dim, labels=labels, temperature=1.0)

    assert head.hidden_dim == hidden_dim
    assert head.num_classes == 3
    assert head.labels == labels

    # Batch forward pass
    h = torch.randn(4, hidden_dim)
    logits = head(h)
    assert logits.shape == (4, 3)

    # Single item forward pass
    h_single = torch.randn(hidden_dim)
    logits_single = head(h_single)
    assert logits_single.shape == (3,)

    # Prediction mapping
    pred_label = head.predict(h_single)
    assert pred_label in labels


def test_linear_decision_head_serialization():
    hidden_dim = 32
    labels = ["Low", "Medium", "High"]
    head = LinearDecisionHead(hidden_dim=hidden_dim, labels=labels, temperature=0.5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        head.save_pretrained(tmp_dir)

        loaded_head = BaseDecisionHead.from_pretrained(tmp_dir)
        assert isinstance(loaded_head, LinearDecisionHead)
        assert loaded_head.hidden_dim == hidden_dim
        assert loaded_head.labels == labels
        assert loaded_head.temperature == 0.5

        test_input = torch.randn(2, hidden_dim)
        assert torch.allclose(head(test_input), loaded_head(test_input))


def test_saved_head_rejects_unsafe_dimensions_and_shape_mismatches(tmp_path: Path):
    head = LinearDecisionHead(hidden_dim=8, num_classes=2)
    head.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    config["hidden_dim"] = 100_000_000
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="parameters; maximum"):
        BaseDecisionHead.from_pretrained(tmp_path)

    config["hidden_dim"] = 9
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="tensor shapes do not match"):
        BaseDecisionHead.from_pretrained(tmp_path)

    config["hidden_dim"] = 8
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match backbone"):
        BaseDecisionHead.from_pretrained(tmp_path, expected_hidden_dim=9)


def test_bilinear_metric_head_basic():
    hidden_dim = 64
    rank = 16
    head = BilinearMetricHead(hidden_dim=hidden_dim, rank=rank, temperature=1.0)
    assert head.input_strategy == "static"

    h_query = torch.randn(2, hidden_dim)
    h_options = torch.randn(5, hidden_dim)

    scores = head(h_query, h_options)
    assert scores.shape == (2, 5)

    # Single query vector
    scores_single = head(h_query[0], h_options)
    assert scores_single.shape == (5,)


def test_bilinear_metric_head_serialization():
    hidden_dim = 32
    rank = 8
    head = BilinearMetricHead(
        hidden_dim=hidden_dim,
        rank=rank,
        temperature=0.75,
        learnable_temperature=True,
        input_strategy="contextual",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        head.save_pretrained(tmp_dir)

        loaded_head = BaseDecisionHead.from_pretrained(tmp_dir)
        assert isinstance(loaded_head, BilinearMetricHead)
        assert loaded_head.hidden_dim == hidden_dim
        assert loaded_head.rank == rank
        assert loaded_head.input_strategy == "contextual"

        q = torch.randn(1, hidden_dim)
        k = torch.randn(4, hidden_dim)
        assert torch.allclose(head(q, k), loaded_head(q, k))


def test_bilinear_metric_head_loads_legacy_static_config():
    head = BilinearMetricHead(hidden_dim=16, rank=4)

    with tempfile.TemporaryDirectory() as tmp_dir:
        head.save_pretrained(tmp_dir)
        config_path = Path(tmp_dir) / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.pop("input_strategy")
        config_path.write_text(json.dumps(config), encoding="utf-8")

        loaded_head = BaseDecisionHead.from_pretrained(tmp_dir)

    assert isinstance(loaded_head, BilinearMetricHead)
    assert loaded_head.input_strategy == "static"


@pytest.fixture(scope="module")
def shared_engine(qwen_engine):
    return qwen_engine


@pytest.mark.model_integration
def test_engine_with_linear_decision_head(shared_engine):
    # Hidden dimension of Qwen2.5-0.5B is 896
    hidden_dim = shared_engine.model.config.hidden_size
    options = ["Technical Support", "Billing & Invoicing", "Account Security"]
    head = LinearDecisionHead(hidden_dim=hidden_dim, labels=options)

    ticket = "I was double-billed for my subscription last month. Please refund invoice #49281 immediately."
    decision_schema = schema(category=choice(options, criterion="Select ticket category."))

    res = shared_engine.evaluate(context=ticket, schema=decision_schema, head={"category": head})
    assert "category" in res.decisions
    dec = res.decisions["category"]
    assert dec.value in options
    assert isinstance(dec.is_certain, bool)
    assert dec.margin >= 0.0
    assert len(dec.distribution) == 3


@pytest.mark.model_integration
def test_engine_with_bilinear_metric_head(shared_engine):
    hidden_dim = shared_engine.model.config.hidden_size
    head = BilinearMetricHead(hidden_dim=hidden_dim, rank=128)

    ticket = "I need help resetting my password and unlocking my account."
    options = ["Password Reset", "Billing Inquiries", "Network Outage"]
    decision_schema = schema(topic=choice(options, criterion="Select ticket category."))

    # Pass head as engine evaluation override
    res = shared_engine.evaluate(context=ticket, schema=decision_schema, head={"topic": head})
    assert "topic" in res.decisions
    dec = res.decisions["topic"]
    assert dec.value in options
    assert len(dec.distribution) == 3


def test_linear_head_probe_training_loop():
    hidden_dim = 32
    num_classes = 3
    head = LinearDecisionHead(hidden_dim=hidden_dim, num_classes=num_classes)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01)
    loss_fn = BrierScoreLoss(from_logits=True)

    # Synthetic features and targets
    X = torch.randn(20, hidden_dim)
    y = torch.randint(0, num_classes, (20,))

    initial_loss = None
    final_loss = None
    for step in range(25):
        optimizer.zero_grad()
        logits = head(X)
        loss = loss_fn(logits, y)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss
