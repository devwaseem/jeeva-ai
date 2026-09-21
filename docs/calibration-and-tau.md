# Calibration and Certainty

Jeeva separates model scoring evidence from validated certainty.

For each finite field, the scorer produces a distribution over candidates. The winning margin is:

$$
\Delta = p_{(1)} - p_{(2)}
$$

where $p_{(1)}$ and $p_{(2)}$ are the largest and second-largest candidate scores after normalization.

A field always has a certainty threshold. Resolution precedence is explicit: a primitive's manual `tau` wins, otherwise a calibrated threshold is used when available, otherwise the engine default applies.

## Fit calibration

```python
from jeeva import FitConfig

trained = base.fit(
    training_examples,
    schema=TicketDecision,
    validation=validation_examples,
    config=FitConfig(
        target_precision=0.95,
        minimum_calibration_examples=20,
    ),
)
```

For each field, Jeeva runs the fitted model over the validation examples and records
each winning margin together with whether the prediction was correct. It evaluates
complete equal-margin groups and selects the smallest observed margin whose accepted
predictions meet `target_precision`; this maximizes empirical coverage at that
precision objective. The threshold, observed precision, coverage, and sample count
are stored under `calibration` in the saved artifact's `jeeva.json` manifest.

If a field has fewer than `minimum_calibration_examples` labeled observations, its
status is `"insufficient_data"`. If no positive-coverage threshold reaches the target,
its status is `"unattainable"` and Jeeva records the best observed precision and
coverage without persisting a learned threshold. Both cases remain `"unvalidated"` at
runtime and use manual `tau`, when declared, or the default threshold.

## Runtime semantics

```python
result = trained.decide(text, TicketDecision)
field = result.fields["department"]
```

`field.is_certain` is always boolean:

- `True`: the field's margin clears its active threshold.
- `False`: the margin falls below its active threshold, allowing an arbiter to run.

`field.threshold_source` identifies the active threshold's source:

- `"manual"`: declared as `tau` on the primitive.
- `"calibrated"`: fitted from held-out examples.
- `"default"`: the engine's zero-shot threshold.

`field.tau` reports the numeric threshold actually used. `field.calibration`
separately reports whether held-out calibration evidence exists. The aggregate
`result.is_certain` is `True` only when every field is certain.
`result.calibration` reports `"validated"`, `"partial"`, or `"unvalidated"`
calibration coverage; it does not override a manual threshold.

## Schema identity

Calibration is keyed to a canonical fingerprint derived from field order, primitive kind, criteria, candidate values, and examples. Changing any of those creates a new identity. Manual `tau` is intentionally excluded, so operators can tune escalation without invalidating semantic compatibility.

## Choosing a target precision

Higher `target_precision` generally increases review volume. Select it from application risk, then measure both accepted precision and review rate on an independent evaluation set.

```python
accepted = [case for case in cases if case.result.is_certain is True]
accepted_precision = sum(case.correct for case in accepted) / len(accepted)
review_rate = 1 - len(accepted) / len(cases)
```

The validation-derived threshold is not a guarantee under distribution shift. Monitor schema-specific performance and recalibrate with representative data.

## Raw evidence

Even without calibration, the result exposes:

```python
field.distribution
field.margin
field.tau
field.threshold_source
field.entropy
field.normalized_entropy
```

These are useful for analysis and ranking. They should not be labeled as confidence or used to assert a precision SLA without validation.
