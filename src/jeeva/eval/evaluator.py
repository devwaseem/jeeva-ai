from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import torch

from jeeva._terminal import terminal_text
from jeeva.arbiters import BaseArbiter
from jeeva.data.dataset import JeevaDataset
from jeeva.schema.primitives import Score, normalize_schema


class _DecisionProvider(Protocol):
    def decide(
        self,
        context: str,
        schema: Any,
        *,
        arbiter: BaseArbiter | None = None,
    ) -> Any: ...


def compute_macro_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    """Computes Macro-F1 across all classes without external scikit-learn dependency."""
    if len(y_true) == 0:
        return 0.0

    f1_scores = []
    for c in range(num_classes):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def compute_ece(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    num_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) across reliability bins.
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    n = len(confidences)
    if n == 0:
        return 0.0

    conf_arr = np.array(confidences, dtype=np.float64)
    corr_arr = np.array(correctness, dtype=np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (conf_arr >= bin_lower) & (
            conf_arr < bin_upper if i < num_bins - 1 else conf_arr <= bin_upper
        )
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(corr_arr[in_bin])
            bin_conf = np.mean(conf_arr[in_bin])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


@dataclass
class EvaluationScorecard:
    """Discriminative, probability-calibration, and precision-threshold metrics."""

    field_metrics: dict[str, dict[str, float]]
    overall_accuracy: float
    overall_macro_f1: float
    overall_brier_score: float
    overall_ece: float
    precision_sweep: list[dict[str, float]]
    precision_targets: dict[str, dict[str, Any]]
    latency_percentiles: dict[str, dict[str, float]]
    num_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_table(self) -> str:
        lines = []
        lines.append("=" * 88)
        lines.append(f"  JEEVA EVALUATION SCORECARD ({self.num_samples} samples)")
        lines.append("=" * 88)
        lines.append(f"Categorical Accuracy: {self.overall_accuracy * 100:.2f}%")
        lines.append(f"Categorical Macro-F1: {self.overall_macro_f1 * 100:.2f}%")
        lines.append(f"Mean Brier Score:     {self.overall_brier_score:.4f}")
        lines.append(f"Categorical ECE:      {self.overall_ece * 100:.2f}%")
        lines.append("-" * 88)
        lines.append("PER-FIELD BREAKDOWN:")
        for field_name, metrics in self.field_metrics.items():
            if "mae" in metrics:
                tolerance = metrics.get("within_tolerance_accuracy")
                tolerance_text = (
                    f", within tolerance={tolerance * 100:.1f}%" if tolerance is not None else ""
                )
                lines.append(
                    f"  {terminal_text(field_name)}: MAE={metrics['mae']:.4f}, "
                    f"RMSE={metrics['rmse']:.4f}, "
                    f"RPS={metrics['ranked_probability_score']:.4f}, "
                    f"Brier={metrics['brier_score']:.4f}{tolerance_text}"
                )
            else:
                lines.append(
                    f"  {terminal_text(field_name)}: "
                    f"accuracy={metrics['accuracy'] * 100:.1f}%, "
                    f"macro-F1={metrics['macro_f1'] * 100:.1f}%, "
                    f"Brier={metrics['brier_score']:.4f}, ECE={metrics['ece'] * 100:.1f}%"
                )

        lines.append("-" * 88)
        lines.append("EMPIRICAL PRECISION TARGETS:")
        for target_name, target in self.precision_targets.items():
            if target["achievable"]:
                tau_s = f"tau* = {target['threshold']:.3f}"
                cov_s = f"Coverage: {target['coverage'] * 100:.1f}%"
                prec_s = f"Precision: {target['precision'] * 100:.1f}%"
                outcome = f"{tau_s:<16} {cov_s:<18} {prec_s}"
            else:
                outcome = (
                    "Unattainable "
                    f"(best precision={target['precision'] * 100:.1f}%, "
                    f"coverage={target['coverage'] * 100:.1f}%)"
                )
            lines.append(f"  Target {terminal_text(target_name)} -> {outcome}")

        lines.append("-" * 88)
        lines.append("LATENCY SLA PROFILES (ms):")
        lines.append(f"{'Metric':<18} {'P50':<10} {'P90':<10} {'P95':<10} {'P99':<10} {'Mean':<10}")
        for stage, pcts in self.latency_percentiles.items():
            lines.append(
                f"{terminal_text(stage):<18} {pcts['p50']:<10.1f} "
                f"{pcts['p90']:<10.1f} {pcts['p95']:<10.1f} "
                f"{pcts['p99']:<10.1f} {pcts['mean']:<10.1f}"
            )
        lines.append("=" * 88)
        return "\n".join(lines)


class Evaluator:
    """
    Comprehensive evaluation pipeline for Jeeva decision models.
    Measures accuracy, Macro-F1, proper scoring calibration (Brier + ECE),
    empirical precision/coverage thresholds, and latency percentiles.
    """

    def __init__(
        self,
        model: _DecisionProvider,
        schema: Any,
    ):
        self.model = model
        self.schema = schema
        self.primitives = normalize_schema(schema)

    def run(
        self,
        dataset: JeevaDataset,
        target_precisions: Sequence[float] = (0.95, 0.98, 0.99),
        arbiter: BaseArbiter | None = None,
    ) -> EvaluationScorecard:
        """Evaluate categorical and ordinal fields without conflating their metrics."""
        if len(dataset) == 0:
            raise ValueError("Evaluation dataset is empty.")

        keys = [
            primitive.key or f"field_{index}" for index, primitive in enumerate(self.primitives)
        ]
        field_predictions: dict[str, list[int]] = {key: [] for key in keys}
        field_ground_truth: dict[str, list[int]] = {key: [] for key in keys}
        field_confidences: dict[str, list[float]] = {key: [] for key in keys}
        field_correctness: dict[str, list[bool]] = {key: [] for key in keys}
        field_briers: dict[str, list[float]] = {key: [] for key in keys}
        field_score_predictions: dict[str, list[float]] = {key: [] for key in keys}
        field_score_targets: dict[str, list[float]] = {key: [] for key in keys}
        field_ranked_scores: dict[str, list[float]] = {key: [] for key in keys}

        all_certainty: list[float] = []
        all_correctness: list[bool] = []
        timings: dict[str, list[float]] = {
            "prefill_ms": [],
            "dispatch_ms": [],
            "harvest_ms": [],
            "total_ms": [],
        }

        for item in dataset:
            decision = self.model.decide(
                context=item["document"],
                schema=self.schema,
                arbiter=arbiter,
            )
            for timing_name in timings:
                if timing_name in decision.telemetry:
                    timings[timing_name].append(decision.telemetry[timing_name])
            field_results = decision.fields if hasattr(decision, "fields") else decision.decisions
            for lane_idx, primitive in enumerate(self.primitives):
                if item["lane_mask"][lane_idx] == 0.0:
                    continue
                key = keys[lane_idx]
                target = item["targets"][lane_idx]
                target_index = int(torch.argmax(target).item())
                result = field_results[key]
                probabilities = torch.tensor(
                    [result.distribution[label] for label in primitive.distribution_labels],
                    dtype=torch.float32,
                )
                certainty_score = (
                    result.margin if result.certainty_score is None else result.certainty_score
                )
                brier = float(torch.mean((probabilities - target).square()).item())
                field_briers[key].append(brier)

                if isinstance(primitive, Score):
                    target_value = float(
                        sum(
                            probability * level
                            for probability, level in zip(target.tolist(), primitive.level_values)
                        )
                    )
                    prediction = float(result.value)
                    predicted_cdf = torch.cumsum(probabilities, dim=-1)[:-1]
                    target_cdf = torch.cumsum(target, dim=-1)[:-1]
                    intervals = torch.diff(torch.tensor(primitive.level_values))
                    interval_weights = intervals / intervals.sum()
                    ranked_score = float(
                        torch.sum(interval_weights * (predicted_cdf - target_cdf).square()).item()
                    )
                    field_score_predictions[key].append(prediction)
                    field_score_targets[key].append(target_value)
                    field_ranked_scores[key].append(ranked_score)
                    if primitive.tolerance is not None:
                        correct = primitive.is_correct(prediction, target_value)
                        field_correctness[key].append(correct)
                        all_certainty.append(certainty_score)
                        all_correctness.append(correct)
                    continue

                predicted_index = primitive.selection_index(result.value)
                correct = bool(predicted_index == target_index)
                field_predictions[key].append(predicted_index)
                field_ground_truth[key].append(target_index)
                field_confidences[key].append(certainty_score)
                field_correctness[key].append(correct)
                all_certainty.append(certainty_score)
                all_correctness.append(correct)

        field_metrics: dict[str, dict[str, float]] = {}
        categorical_accuracies: list[float] = []
        categorical_f1s: list[float] = []
        categorical_eces: list[float] = []
        all_brier_scores: list[float] = []

        for index, primitive in enumerate(self.primitives):
            key = keys[index]
            briers = field_briers[key]
            if not briers:
                continue
            brier_score = float(np.mean(briers))
            all_brier_scores.append(brier_score)
            if isinstance(primitive, Score):
                predictions = np.array(field_score_predictions[key], dtype=np.float64)
                targets = np.array(field_score_targets[key], dtype=np.float64)
                errors = predictions - targets
                metrics = {
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "ranked_probability_score": float(np.mean(field_ranked_scores[key])),
                    "brier_score": brier_score,
                    "num_samples": float(len(targets)),
                }
                if primitive.tolerance is not None:
                    metrics["within_tolerance_accuracy"] = float(np.mean(field_correctness[key]))
                field_metrics[key] = metrics
                continue

            predictions = field_predictions[key]
            targets = field_ground_truth[key]
            correctness = field_correctness[key]
            accuracy = float(np.mean(correctness))
            macro_f1 = compute_macro_f1(
                targets,
                predictions,
                num_classes=len(primitive.options),
            )
            ece = compute_ece(field_confidences[key], correctness)
            field_metrics[key] = {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "brier_score": brier_score,
                "ece": ece,
                "num_samples": float(len(targets)),
            }
            categorical_accuracies.append(accuracy)
            categorical_f1s.append(macro_f1)
            categorical_eces.append(ece)

        overall_acc = float(np.mean(categorical_accuracies)) if categorical_accuracies else 0.0
        overall_f1 = float(np.mean(categorical_f1s)) if categorical_f1s else 0.0
        overall_brier = float(np.mean(all_brier_scores)) if all_brier_scores else 0.0
        overall_ece = float(np.mean(categorical_eces)) if categorical_eces else 0.0

        precision_sweep: list[dict[str, float]] = []
        certainty_array = np.array(all_certainty, dtype=np.float64)
        correctness_array = np.array(all_correctness, dtype=bool)
        for tau in np.linspace(0.05, 0.95, 19):
            mask = certainty_array >= tau
            coverage = float(np.mean(mask)) if len(certainty_array) > 0 else 0.0
            precision = float(np.mean(correctness_array[mask])) if np.sum(mask) > 0 else 0.0
            precision_sweep.append(
                {
                    "threshold": round(float(tau), 3),
                    "coverage": round(coverage, 4),
                    "precision": round(precision, 4),
                }
            )

        precision_targets: dict[str, dict[str, Any]] = {}
        for target_precision in target_precisions:
            matching = next(
                (
                    item
                    for item in precision_sweep
                    if item["coverage"] > 0.0 and item["precision"] >= target_precision
                ),
                None,
            )
            positive_coverage = [item for item in precision_sweep if item["coverage"] > 0.0]
            best = (
                max(positive_coverage, key=lambda item: (item["precision"], item["coverage"]))
                if positive_coverage
                else None
            )
            precision_targets[f"{int(target_precision * 100)}%"] = {
                "target_precision": target_precision,
                "threshold": matching["threshold"] if matching else None,
                "coverage": (
                    matching["coverage"]
                    if matching
                    else (best["coverage"] if best is not None else 0.0)
                ),
                "precision": (
                    matching["precision"]
                    if matching
                    else (best["precision"] if best is not None else 0.0)
                ),
                "achievable": matching is not None,
            }

        latency_percentiles: dict[str, dict[str, float]] = {}
        for stage, values in timings.items():
            if values:
                array_values = np.array(values)
                latency_percentiles[stage] = {
                    "p50": float(np.percentile(array_values, 50)),
                    "p90": float(np.percentile(array_values, 90)),
                    "p95": float(np.percentile(array_values, 95)),
                    "p99": float(np.percentile(array_values, 99)),
                    "mean": float(np.mean(array_values)),
                }
            else:
                latency_percentiles[stage] = {
                    "p50": 0.0,
                    "p90": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "mean": 0.0,
                }

        return EvaluationScorecard(
            field_metrics=field_metrics,
            overall_accuracy=overall_acc,
            overall_macro_f1=overall_f1,
            overall_brier_score=overall_brier,
            overall_ece=overall_ece,
            precision_sweep=precision_sweep,
            precision_targets=precision_targets,
            latency_percentiles=latency_percentiles,
            num_samples=len(dataset),
        )
