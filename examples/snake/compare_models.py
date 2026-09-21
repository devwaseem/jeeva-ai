"""Compare zero-shot causal and fitted encoder decisions on one held-out Snake corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from examples.snake.play_snake import DirectionDecision
from jeeva import Decision, DecisionModel, DecisionSchema


class DecisionProvider(Protocol):
    @property
    def architecture(self) -> str: ...

    def decide(self, context: str, schema: type[DecisionSchema]) -> Decision: ...


@dataclass(frozen=True)
class EvaluationCase:
    context: str
    direction: str


@dataclass(frozen=True)
class ComparisonResult:
    label: str
    source: str
    architecture: str
    cases: int
    correct: int
    score: float
    throughput_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float


def load_cases(path: Path, *, limit: int | None = None) -> tuple[list[EvaluationCase], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    cases: list[EvaluationCase] = []
    allowed = set(DirectionDecision.fields()["direction"].distribution_labels)

    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        context = record.get("text")
        direction = record.get("direction")
        if not isinstance(context, str) or not context:
            raise ValueError(f"Line {line_number} has no non-empty text context.")
        if not isinstance(direction, str) or direction not in allowed:
            raise ValueError(f"Line {line_number} has an invalid direction target.")
        cases.append(EvaluationCase(context=context, direction=direction))
        if limit is not None and len(cases) >= limit:
            break

    if not cases:
        raise ValueError("The evaluation dataset contains no cases.")
    return cases, digest


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_model(
    model: DecisionProvider,
    cases: Sequence[EvaluationCase],
    *,
    label: str,
    source: str,
) -> ComparisonResult:
    # Exclude one schema compilation and backend warm-up decision from measurements.
    model.decide(cases[0].context, DirectionDecision)

    correct = 0
    latencies_ms: list[float] = []
    for case in cases:
        started = time.perf_counter()
        decision = model.decide(case.context, DirectionDecision)
        latencies_ms.append((time.perf_counter() - started) * 1_000.0)
        correct += decision.value["direction"] == case.direction

    elapsed_seconds = sum(latencies_ms) / 1_000.0
    return ComparisonResult(
        label=label,
        source=source,
        architecture=model.architecture,
        cases=len(cases),
        correct=correct,
        score=correct / len(cases),
        throughput_per_second=len(cases) / elapsed_seconds,
        latency_p50_ms=_percentile(latencies_ms, 0.50),
        latency_p95_ms=_percentile(latencies_ms, 0.95),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--causal-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Off-the-shelf causal model used without fitting.",
    )
    parser.add_argument(
        "--encoder-artifact",
        type=Path,
        default=Path("examples/snake/checkpoints/snake_encoder"),
        help="Fitted encoder artifact created by train_snake_encoder.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("examples/snake/data/snake_eval.jsonl"),
        help="Held-out Snake JSONL evaluated by both models.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases for a local smoke run.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"evaluation dataset not found: {args.data}")
    if not args.encoder_artifact.is_dir():
        parser.error(f"fitted encoder artifact not found: {args.encoder_artifact}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    cases, digest = load_cases(args.data, limit=args.limit)
    causal = DecisionModel(args.causal_model, device=args.device, dtype=args.dtype)
    encoder = DecisionModel(args.encoder_artifact, device=args.device, dtype=args.dtype)
    results = [
        evaluate_model(causal, cases, label="zero-shot causal", source=args.causal_model),
        evaluate_model(
            encoder,
            cases,
            label="fitted encoder",
            source=str(args.encoder_artifact),
        ),
    ]
    report = {
        "dataset": str(args.data),
        "dataset_sha256": digest,
        "measurement": "warm wall-clock DecisionModel.decide latency on this machine",
        "results": [asdict(result) for result in results],
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"Dataset: {args.data}")
    print(f"SHA-256: {digest}")
    print("Measurements are local warm wall-clock results, not universal claims.\n")
    for result in results:
        print(f"{result.label} ({result.architecture}; {result.source})")
        print(f"  score:      {result.score:.3f} ({result.correct}/{result.cases})")
        print(f"  throughput: {result.throughput_per_second:.2f} decisions/s")
        print(
            f"  latency:    p50 {result.latency_p50_ms:.2f} ms; p95 {result.latency_p95_ms:.2f} ms"
        )


if __name__ == "__main__":
    main()
