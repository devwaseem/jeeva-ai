# Evaluation

Evaluate the public behavior your application consumes: dictionary correctness, per-class behavior, calibration coverage, and latency on target hardware.

## Held-out evaluation

Keep evaluation examples separate from fitting and calibration data.

```python
from collections import Counter


def evaluate(model, schema, examples):
    correct = 0
    total = 0
    calibration = Counter()

    for example in examples:
        result = model.decide(example.context, schema)
        correct += result.value == example.target
        total += 1
        calibration[result.calibration] += 1

    return {
        "accuracy": correct / total,
        "count": total,
        "calibration_coverage": dict(calibration),
    }
```

For multi-field schemas, also compute metrics per field. Aggregate exact-match accuracy can hide a weak field, while averaging field accuracy can hide unsafe combinations.

## Metrics to record

At minimum:

- exact dictionary target match
- per-field accuracy
- per-class precision, recall, and support
- confusion matrices for categorical fields
- calibration coverage (`validated`, `partial`, `unvalidated`)
- accepted precision and review rate under `is_certain`
- latency percentiles on deployment hardware
- failures by schema fingerprint

For high-impact decisions, define application-specific loss and mandatory review conditions before evaluating.

## Evaluating certainty

Only evaluate certainty on results with held-out calibration evidence:

```python
accepted = []
reviewed = []

for example in evaluation_examples:
    result = model.decide(example.context, EvaluationSchema)
    if result.is_certain is True:
        accepted.append(result.value == example.target)
    else:
        reviewed.append(result.value == example.target)

accepted_precision = sum(accepted) / len(accepted) if accepted else None
review_rate = len(reviewed) / len(evaluation_examples)
```

Report both accepted precision and review rate. A threshold can increase apparent precision by sending more cases to review; neither number is meaningful alone.

`FitConfig.target_precision` selects thresholds from the fit validation sample. It is not evidence that the same precision holds under distribution shift. Measure it again on an independent evaluation set.

## Comparing zero-shot and trained models

The same evaluator can compare a base model and saved artifact because both use `DecisionModel`:

```python
base = DecisionModel("Qwen/Qwen2.5-1.5B-Instruct", device="mps")
trained = DecisionModel("artifacts/ticket-model", device="mps")

base_report = evaluate(base, TicketDecision, evaluation_examples)
trained_report = evaluate(trained, TicketDecision, evaluation_examples)
```

Zero-shot results have boolean certainty from the default threshold, so accepted/review behavior can be measured immediately. Report those measurements as heuristic-threshold results until held-out calibration fits schema-specific thresholds.

## Schema-specific reporting

Group metrics by `result.telemetry["schema_fingerprint"]` when comparing runs. Never merge evidence from semantically different schemas into one calibration claim.

## Promotion checklist

Promote an artifact only when:

1. the exact artifact directory was evaluated;
2. every required schema has sufficient evaluation support;
3. per-class and per-field floors pass;
4. accepted precision and review rate meet policy;
5. latency and memory fit the deployment device;
6. rollback retains the previous artifact;
7. the application distinguishes boolean certainty from its calibration provenance.
