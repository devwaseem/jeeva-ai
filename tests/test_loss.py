import pytest
import torch

from jeeva.training.loss import (
    BrierScoreLoss,
    CompositeDecisionLoss,
    backward_finite_loss,
    clip_finite_gradients,
    require_finite_loss,
    require_finite_value,
)


def test_brier_loss_one_hot():
    loss_fn = BrierScoreLoss(from_logits=False, reduction="mean")

    # Perfect prediction for class 0 among 3 classes
    preds = torch.tensor([[1.0, 0.0, 0.0]])
    targets = torch.tensor([0], dtype=torch.long)
    loss = loss_fn(preds, targets)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    # Worst prediction for class 0 (predicted class 1)
    worst_preds = torch.tensor([[0.0, 1.0, 0.0]])
    # (0 - 1)^2 + (1 - 0)^2 + (0 - 0)^2 = 1 + 1 = 2. Mean across 3 classes = 2/3
    worst_loss = loss_fn(worst_preds, targets)
    assert torch.isclose(worst_loss, torch.tensor(2.0 / 3.0), atol=1e-5)


def test_brier_loss_soft_distribution():
    loss_fn = BrierScoreLoss(from_logits=False, reduction="mean")

    # Both pred and target are distributions
    preds = torch.tensor([[0.8, 0.2]])
    targets = torch.tensor([[0.7, 0.3]])
    # diffs = [0.1, -0.1], sq = [0.01, 0.01], mean = 0.01
    loss = loss_fn(preds, targets)
    assert torch.isclose(loss, torch.tensor(0.01), atol=1e-5)


def test_brier_loss_from_logits_gradients():
    loss_fn = BrierScoreLoss(from_logits=True, reduction="mean")

    logits = torch.tensor([[2.0, 0.5, -1.0]], requires_grad=True)
    targets = torch.tensor([[0.9, 0.1, 0.0]])

    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_composite_loss_combines_soft_ce_and_brier():
    loss_fn = CompositeDecisionLoss(
        brier_weight=0.5,
        ordinal_weight=0.0,
        reduction="none",
    )
    logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
    target = torch.tensor([[0.75, 0.25]])

    loss = loss_fn(logits, target)
    probabilities = torch.softmax(logits, dim=-1)
    expected = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    expected = expected + 0.5 * torch.mean((probabilities - target) ** 2, dim=-1)
    assert torch.allclose(loss, expected)
    loss.sum().backward()
    assert logits.grad is not None


def test_composite_loss_penalizes_ordinal_distance():
    loss_fn = CompositeDecisionLoss(
        brier_weight=0.0,
        ordinal_weight=1.0,
        reduction="none",
    )
    levels = torch.tensor([0.0, 1.0, 10.0])
    target = torch.tensor([[1.0, 0.0, 0.0]])

    near = loss_fn(torch.tensor([[0.0, 8.0, 0.0]]), target, level_values=levels)
    far = loss_fn(torch.tensor([[0.0, 0.0, 8.0]]), target, level_values=levels)
    assert far.item() > near.item()


def test_composite_loss_rejects_non_increasing_levels():
    loss_fn = CompositeDecisionLoss()
    with torch.no_grad():
        try:
            loss_fn(
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[1.0, 0.0]]),
                level_values=torch.tensor([1.0, 1.0]),
            )
        except ValueError as exc:
            assert "strictly increasing" in str(exc)
        else:
            raise AssertionError("Expected invalid ordinal levels to fail.")


def test_require_finite_loss_accepts_finite_scalar():
    require_finite_loss(
        torch.tensor(0.25),
        context="epoch 1, sample 1/2, scorer bilinear, device mps, dtype torch.float16",
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_require_finite_loss_stops_on_non_finite_scalar(value):
    with pytest.raises(
        FloatingPointError,
        match=r"Training stopped: non-finite loss .*epoch 2, sample 3/10",
    ):
        require_finite_loss(
            torch.tensor(value),
            context="epoch 2, sample 3/10, scorer contextual, device mps, dtype torch.float16",
        )


def test_backward_finite_loss_stops_before_backpropagation():
    loss = torch.tensor(float("nan"), requires_grad=True)

    with pytest.raises(FloatingPointError, match="non-finite loss"):
        backward_finite_loss(loss, context="causal training regression")

    assert loss.grad is None


def test_require_finite_value_rejects_invalid_checkpoint_metric():
    with pytest.raises(FloatingPointError, match="non-finite checkpoint metric nan"):
        require_finite_value(
            float("nan"),
            name="checkpoint metric",
            context="epoch 1, scorer contextual",
        )


def test_clip_finite_gradients_clips_and_returns_original_norm():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 4.0])

    gradient_norm = clip_finite_gradients(
        [parameter],
        max_norm=1.0,
        context="finite gradient regression",
    )

    assert gradient_norm.item() == pytest.approx(5.0)
    assert parameter.grad.norm().item() == pytest.approx(1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_clip_finite_gradients_stops_before_optimizer_step(value):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([value])

    with pytest.raises(
        FloatingPointError,
        match="non-finite gradient norm at epoch 1, batch 1/15",
    ):
        clip_finite_gradients(
            [parameter],
            max_norm=1.0,
            context="epoch 1, batch 1/15, architecture encoder",
        )
