"""Stable terminal presentation for Jeeva model fitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jeeva._terminal import terminal_text

_WIDTH = 88
_LABEL_WIDTH = 21

TrainingSection = tuple[str, Sequence[tuple[str, object]]]


def _print_section(name: str, rows: Sequence[tuple[str, object]]) -> None:
    print(terminal_text(name).upper())
    for label, value in rows:
        safe_label = terminal_text(label)
        print(f"  {safe_label:<{_LABEL_WIDTH}} {terminal_text(value)}")
    print()


def print_training_header(sections: Sequence[TrainingSection]) -> None:
    """Print one architecture-neutral Jeeva training header."""
    print("=" * _WIDTH)
    print("JEEVA DECISION MODEL TRAINING")
    print("=" * _WIDTH)
    for name, rows in sections:
        _print_section(name, rows)
    print("-" * _WIDTH)
    print("Configuration ready. Preparing training.")
    print("=" * _WIDTH, flush=True)


def print_training_complete(report: Mapping[str, Any]) -> None:
    """Print a compact summary after Jeeva finishes fitting a model."""
    rows: list[tuple[str, object]] = [
        ("Architecture", str(report.get("architecture", "unknown")).capitalize()),
        ("Adaptation", str(report.get("adaptation", "unknown")).upper()),
    ]
    history = report.get("history")
    if isinstance(history, Sequence) and history:
        final = history[-1]
        if isinstance(final, Mapping):
            rows.append(("Epochs completed", len(history)))
            train_loss = final.get("train_loss")
            if isinstance(train_loss, int | float):
                rows.append(("Final train loss", f"{train_loss:.6f}"))
            eval_brier = final.get("eval_brier_score")
            if isinstance(eval_brier, int | float):
                rows.append(("Validation Brier", f"{eval_brier:.6f}"))

            durations: list[float] = []
            for epoch in history:
                if not isinstance(epoch, Mapping):
                    continue
                duration = epoch.get("duration_seconds")
                if isinstance(duration, int | float):
                    durations.append(float(duration))
            if durations:
                rows.append(("Training time", f"{sum(durations):.1f}s"))

    trainable_parameters = report.get("trainable_parameters")
    if isinstance(trainable_parameters, int):
        rows.append(("Trainable parameters", f"{trainable_parameters:,}"))

    print()
    print("=" * _WIDTH)
    print("TRAINING COMPLETE")
    print("=" * _WIDTH)
    _print_section("Result", rows)
    print("Model fitted. Call save(...) to persist the trained artifact.")
    print("=" * _WIDTH, flush=True)
