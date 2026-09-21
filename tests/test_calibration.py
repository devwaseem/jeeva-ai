import pytest
import torch

from jeeva.calibration.threshold import (
    QuantileNotchCalibrator,
    classify_cascade,
    compute_margin,
)


def test_compute_margin():
    probs = torch.tensor([0.7, 0.2, 0.1])
    margin = compute_margin(probs)
    assert pytest.approx(margin, 1e-4) == 0.5

    single_prob = torch.tensor([1.0])
    assert compute_margin(single_prob) == 1.0


def test_classify_cascade():
    dec_certain = classify_cascade(margin=0.45, threshold=0.3)
    assert dec_certain.is_certain
    assert not dec_certain.is_ambiguous
    assert dec_certain.status == "Certain"

    dec_ambig = classify_cascade(margin=0.25, threshold=0.3)
    assert not dec_ambig.is_certain
    assert dec_ambig.is_ambiguous
    assert dec_ambig.status == "Ambiguous"


def test_quantile_notch_calibrator():
    # Samples with high margin are correct, low margin are incorrect
    margins = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85]
    correct = [False, False, True, True, True, True]

    calibrator = QuantileNotchCalibrator(target_accuracy=1.0)
    result = calibrator.fit(margins, correct)

    assert result.status == "calibrated"
    assert result.threshold is not None
    assert result.threshold >= 0.35
    assert result.precision == 1.0
    assert result.coverage == 4.0 / 6.0

    coverage, precision = calibrator.evaluate_coverage(margins, correct)
    assert precision == result.precision
    assert coverage == result.coverage


def test_quantile_notch_calibrator_never_splits_equal_margin_groups():
    calibrator = QuantileNotchCalibrator(target_accuracy=1.0)

    result = calibrator.fit([0.5, 0.5], [False, True])

    assert result.status == "unattainable"
    assert result.threshold is None
    assert result.coverage == 1.0
    assert result.precision == 0.5
    assert calibrator.threshold is None


def test_quantile_notch_calibrator_reports_unattainable_at_maximum_margin():
    calibrator = QuantileNotchCalibrator(target_accuracy=0.95)

    result = calibrator.fit([1.0], [False])

    assert result.status == "unattainable"
    assert result.threshold is None
    assert result.coverage == 1.0
    assert result.precision == 0.0


def test_quantile_notch_calibrator_uses_next_complete_margin_group():
    calibrator = QuantileNotchCalibrator(target_accuracy=1.0)

    result = calibrator.fit(
        [0.2, 0.5, 0.5, 0.8],
        [False, False, True, True],
    )

    assert result.status == "calibrated"
    assert result.threshold == 0.8
    assert result.coverage == 0.25
    assert result.precision == 1.0
