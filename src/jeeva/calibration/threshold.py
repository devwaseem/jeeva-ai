from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


@dataclass
class CascadeDecision:
    """Represents the 2-tier cascade routing outcome for a prediction."""

    status: Literal["Certain", "Ambiguous"]
    margin: float
    threshold: float

    @property
    def is_certain(self) -> bool:
        return self.status == "Certain"

    @property
    def is_ambiguous(self) -> bool:
        return self.status == "Ambiguous"


def compute_margin(probs: torch.Tensor) -> float:
    """
    Computes confidence margin gap: Δ = P_(1) - P_(2).
    For binary distributions, Δ = |P_1 - P_0|.
    """
    if probs.numel() < 2:
        return 1.0
    top_values, _ = torch.topk(probs, k=2)
    return float((top_values[0] - top_values[1]).item())


def classify_cascade(margin: float, threshold: float) -> CascadeDecision:
    """
    Classifies a prediction into the 2-tier cascade:
    - Certain: Δ >= threshold (automate immediately)
    - Ambiguous: Δ < threshold (escalate to arbiter or HITL)
    """
    status: Literal["Certain", "Ambiguous"] = "Certain" if margin >= threshold else "Ambiguous"
    return CascadeDecision(status=status, margin=margin, threshold=threshold)


@dataclass(frozen=True)
class PrecisionThresholdResult:
    """Empirical precision-threshold fit over held-out predictions."""

    status: Literal["calibrated", "unattainable"]
    threshold: float | None
    coverage: float
    precision: float
    samples: int


class QuantileNotchCalibrator:
    """Fit an empirical acceptance threshold for a target held-out precision."""

    def __init__(self, target_accuracy: float = 0.95):
        if not (0.0 < target_accuracy <= 1.0):
            raise ValueError(f"target_accuracy must be in (0, 1], got {target_accuracy}")
        self.target_accuracy = target_accuracy
        self.threshold: float | None = None
        self.result: PrecisionThresholdResult | None = None

    def fit(
        self,
        margins: Sequence[float],
        correctness: Sequence[bool | int],
    ) -> PrecisionThresholdResult:
        """Maximize empirical coverage subject to the requested precision."""
        if len(margins) != len(correctness):
            raise ValueError("margins and correctness must have identical lengths.")
        if len(margins) == 0:
            raise ValueError("Cannot calibrate on empty dataset.")

        margin_array = np.asarray(margins, dtype=np.float64)
        if not np.all(np.isfinite(margin_array)):
            raise ValueError("Calibration margins must be finite.")
        correctness_array = np.asarray(correctness, dtype=bool)
        order = np.argsort(-margin_array, kind="stable")
        sorted_margins = margin_array[order]
        sorted_correctness = correctness_array[order]

        retained = 0
        retained_correct = 0
        selected: PrecisionThresholdResult | None = None
        best: PrecisionThresholdResult | None = None
        index = 0
        sample_count = len(sorted_margins)
        while index < sample_count:
            threshold = sorted_margins[index]
            group_end = index + 1
            while group_end < sample_count and sorted_margins[group_end] == threshold:
                group_end += 1
            retained += group_end - index
            retained_correct += int(np.sum(sorted_correctness[index:group_end]))
            coverage = retained / sample_count
            precision = retained_correct / retained
            candidate = PrecisionThresholdResult(
                status="calibrated",
                threshold=float(threshold),
                coverage=coverage,
                precision=precision,
                samples=sample_count,
            )
            if best is None or (candidate.precision, candidate.coverage) > (
                best.precision,
                best.coverage,
            ):
                best = candidate
            if precision >= self.target_accuracy:
                selected = candidate
            index = group_end

        if selected is None:
            assert best is not None
            result = PrecisionThresholdResult(
                status="unattainable",
                threshold=None,
                coverage=best.coverage,
                precision=best.precision,
                samples=sample_count,
            )
        else:
            result = selected
        self.result = result
        self.threshold = result.threshold
        return result

    def evaluate_coverage(
        self,
        margins: Sequence[float],
        correctness: Sequence[bool | int],
        threshold: float | None = None,
    ) -> tuple[float, float]:
        """
        Evaluates coverage (fraction of samples >= tau) and accuracy on the retained subset.
        Returns: (coverage, precision)
        """
        tau = self.threshold if threshold is None else threshold
        if tau is None:
            raise ValueError("Calibrator has not been fitted and no threshold was provided.")

        m_arr = np.array(margins, dtype=np.float64)
        c_arr = np.array(correctness, dtype=bool)

        retained_mask = m_arr >= tau
        coverage = float(np.mean(retained_mask))
        if np.sum(retained_mask) == 0:
            accuracy = 1.0
        else:
            accuracy = float(np.mean(c_arr[retained_mask]))

        return coverage, accuracy
