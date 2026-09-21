import pytest
import torch

from jeeva.core.candidate_scoring import (
    aggregate_candidate_scores,
    balanced_candidate_orders,
)


def test_balanced_orders_place_every_candidate_in_every_position_once():
    orders = balanced_candidate_orders(3)

    assert orders == ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for position in range(3):
        assert {order[position] for order in orders} == {0, 1, 2}


def test_balanced_orders_reject_candidate_counts_before_quadratic_allocation():
    with pytest.raises(ValueError, match="at most 26 candidates"):
        balanced_candidate_orders(27)


def test_pooling_cancels_a_pure_letter_position_bias():
    orders = balanced_candidate_orders(2)
    positional_scores = torch.tensor(
        [
            [0.0, 2.0],
            [0.0, 2.0],
        ]
    )

    aggregate = aggregate_candidate_scores(
        positional_scores,
        orders,
        temperature=1.0,
    )

    assert torch.equal(
        aggregate.semantic_trials,
        torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
    )
    assert torch.equal(aggregate.scores, torch.tensor([1.0, 1.0]))
    assert aggregate.robust_margin == 0.0
    assert aggregate.order_sensitivity > 0.7


def test_pooling_preserves_semantic_evidence_after_balancing_positions():
    orders = balanced_candidate_orders(2)
    semantic_scores = torch.tensor([4.0, 1.0])
    position_bias = torch.tensor([0.0, 2.0])
    positional_scores = torch.stack(
        [semantic_scores[list(order)] + position_bias for order in orders]
    )

    aggregate = aggregate_candidate_scores(
        positional_scores,
        orders,
        temperature=1.0,
    )

    assert torch.equal(aggregate.scores, torch.tensor([5.0, 2.0]))
    assert aggregate.robust_margin > 0.4
    assert torch.argmax(aggregate.scores).item() == 0


def test_pooling_rejects_non_permutation_orders():
    with pytest.raises(ValueError, match="must be a permutation"):
        aggregate_candidate_scores(
            torch.tensor([[1.0, 2.0]]),
            [(0, 0)],
            temperature=1.0,
        )
