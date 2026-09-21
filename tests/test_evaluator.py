import pytest
import torch

from jeeva.core.engine import EvaluationResult
from jeeva.data.dataset import JeevaDataset
from jeeva.eval.evaluator import (
    EvaluationScorecard,
    Evaluator,
    compute_ece,
    compute_macro_f1,
)
from jeeva.schema import Choice, DecisionSchema, Score, normalize_schema


def test_compute_macro_f1():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    # Class 0: TP=2, FP=1, FN=0 -> P=2/3, R=1, F1=4/5=0.8
    # Class 1: TP=1, FP=0, FN=1 -> P=1, R=1/2, F1=2/3=0.6667
    # Macro-F1 = (0.8 + 0.6667)/2 = 0.7333
    f1 = compute_macro_f1(y_true, y_pred, num_classes=2)
    assert pytest.approx(f1, 1e-3) == 0.7333


def test_compute_ece():
    # Perfectly calibrated predictions
    confidences = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1]
    correctness = [True, True, True, True, False, False]
    ece = compute_ece(confidences, correctness, num_bins=5)
    assert ece >= 0.0


def test_scorecard_summary_table():
    scorecard = EvaluationScorecard(
        field_metrics={
            "category": {
                "accuracy": 0.95,
                "macro_f1": 0.94,
                "brier_score": 0.02,
                "ece": 0.03,
                "num_samples": 100,
            }
        },
        overall_accuracy=0.95,
        overall_macro_f1=0.94,
        overall_brier_score=0.02,
        overall_ece=0.03,
        precision_sweep=[
            {"threshold": 0.2, "coverage": 0.90, "precision": 0.96},
            {"threshold": 0.4, "coverage": 0.80, "precision": 0.99},
        ],
        precision_targets={
            "98%": {
                "target_precision": 0.98,
                "threshold": 0.4,
                "coverage": 0.80,
                "precision": 0.99,
                "achievable": True,
            }
        },
        latency_percentiles={
            "prefill_ms": {"p50": 12.0, "p90": 15.0, "p95": 16.0, "p99": 20.0, "mean": 13.0},
            "total_ms": {"p50": 25.0, "p90": 30.0, "p95": 32.0, "p99": 35.0, "mean": 26.0},
        },
        num_samples=100,
    )

    table_str = scorecard.summary_table()
    assert "JEEVA EVALUATION SCORECARD" in table_str
    assert "95.00%" in table_str
    assert "EMPIRICAL PRECISION TARGETS" in table_str
    assert "LATENCY SLA PROFILES" in table_str


def test_scorecard_summary_escapes_terminal_control_characters():
    scorecard = EvaluationScorecard(
        field_metrics={
            "category\x1b[2J": {
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "brier_score": 0.0,
                "ece": 0.0,
            }
        },
        overall_accuracy=1.0,
        overall_macro_f1=1.0,
        overall_brier_score=0.0,
        overall_ece=0.0,
        precision_sweep=[],
        precision_targets={},
        latency_percentiles={},
        num_samples=1,
    )

    rendered = scorecard.summary_table()

    assert "\x1b" not in rendered
    assert "category\\x1b[2J" in rendered


def test_evaluator_marks_zero_coverage_precision_target_unattainable():
    class Category(DecisionSchema):
        category = Choice(
            "Choose a category.",
            {"a": "Category A.", "b": "Category B."},
        )

    class FixedEngine:
        def decide(self, context, schema, arbiter=None):
            primitive = normalize_schema(schema)[0]
            result = primitive.result_from_probabilities(torch.tensor([0.5, 0.5]))
            return EvaluationResult(
                decisions={"category": result},
                telemetry={"total_ms": 1.0},
            )

    dataset = JeevaDataset(
        [{"text": "Example.", "category": "b"}],
        Category,
    )

    scorecard = Evaluator(FixedEngine(), Category).run(  # type: ignore[arg-type]
        dataset,
        target_precisions=(0.95,),
    )

    target = scorecard.precision_targets["95%"]
    assert target == {
        "target_precision": 0.95,
        "threshold": None,
        "coverage": 0.0,
        "precision": 0.0,
        "achievable": False,
    }
    assert "Unattainable" in scorecard.summary_table()


def test_evaluator_reports_ordinal_metrics_separately():
    class Severity(DecisionSchema):
        severity = Score(
            "Rate severity.",
            {0: "none", 2: "moderate", 4: "critical"},
            tolerance=1.0,
        )

    class FixedEngine:
        def decide(self, context, schema, arbiter=None):
            primitive = normalize_schema(schema)[0]
            result = primitive.result_from_probabilities(torch.tensor([0.0, 1.0, 0.0]))
            return EvaluationResult(
                decisions={"severity": result},
                telemetry={"total_ms": 1.0},
            )

    dataset = JeevaDataset(
        [
            {"text": "Low.", "severity": 1.0},
            {"text": "High.", "severity": 3.0},
        ],
        Severity,
    )
    scorecard = Evaluator(FixedEngine(), Severity).run(dataset)  # type: ignore[arg-type]
    metrics = scorecard.field_metrics["severity"]

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["within_tolerance_accuracy"] == pytest.approx(1.0)
    assert "accuracy" not in metrics
    assert scorecard.overall_accuracy == 0.0
    assert "MAE=1.0000" in scorecard.summary_table()
