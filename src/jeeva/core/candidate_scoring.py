from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

MAX_POSITION_BALANCED_CANDIDATES = 26


@dataclass(frozen=True)
class CandidateScoreAggregate:
    """Semantic candidate scores pooled across balanced display orders."""

    scores: torch.Tensor
    semantic_trials: torch.Tensor
    order_sensitivity: float
    robust_margin: float

    @property
    def trials(self) -> int:
        return int(self.semantic_trials.shape[0])


def balanced_candidate_orders(candidate_count: int) -> tuple[tuple[int, ...], ...]:
    """Return bounded cyclic orders covering every candidate and display position."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1.")
    if candidate_count > MAX_POSITION_BALANCED_CANDIDATES:
        raise ValueError(
            "Position-balanced letter readout supports at most "
            f"{MAX_POSITION_BALANCED_CANDIDATES} candidates; received {candidate_count}."
        )
    base = tuple(range(candidate_count))
    return tuple(base[shift:] + base[:shift] for shift in range(candidate_count))


def aggregate_candidate_scores(
    positional_scores: torch.Tensor,
    orders: Sequence[Sequence[int]],
    *,
    temperature: float,
    offset: Sequence[float] | torch.Tensor | None = None,
) -> CandidateScoreAggregate:
    """Remap positional scores to semantic candidates and pool them geometrically."""
    if positional_scores.ndim != 2:
        raise ValueError("positional_scores must have shape [trials, candidates].")

    trial_count, candidate_count = positional_scores.shape
    if len(orders) != trial_count:
        raise ValueError(f"Received {len(orders)} candidate orders for {trial_count} score trials.")

    expected = tuple(range(candidate_count))
    normalized_orders = tuple(tuple(int(index) for index in order) for order in orders)
    for order in normalized_orders:
        if tuple(sorted(order)) != expected:
            raise ValueError(
                f"Each candidate order must be a permutation of {expected}; received {order}."
            )

    order_tensor = torch.tensor(
        normalized_orders,
        dtype=torch.long,
        device=positional_scores.device,
    )
    semantic_trials = torch.empty_like(positional_scores)
    semantic_trials.scatter_(dim=1, index=order_tensor, src=positional_scores)

    if offset is not None:
        offset_tensor = torch.as_tensor(
            offset,
            dtype=semantic_trials.dtype,
            device=semantic_trials.device,
        )
        if offset_tensor.shape != (candidate_count,):
            raise ValueError(
                f"Candidate score offset has {offset_tensor.numel()} scores; "
                f"expected {candidate_count}."
            )
        semantic_trials = semantic_trials - offset_tensor

    pooled_scores = semantic_trials.mean(dim=0)
    scale = 1.0 / max(1e-4, temperature)
    trial_probabilities = torch.softmax(semantic_trials * scale, dim=-1)
    order_sensitivity = float(
        (trial_probabilities.amax(dim=0) - trial_probabilities.amin(dim=0)).amax().item()
    )

    if candidate_count == 1:
        robust_margin = 1.0
    else:
        winner = int(torch.argmax(pooled_scores).item())
        competitors = trial_probabilities.clone()
        competitors[:, winner] = float("-inf")
        trial_margins = trial_probabilities[:, winner] - competitors.amax(dim=1)
        robust_margin = max(0.0, float(trial_margins.amin().item()))

    return CandidateScoreAggregate(
        scores=pooled_scores,
        semantic_trials=semantic_trials,
        order_sensitivity=order_sensitivity,
        robust_margin=robust_margin,
    )
