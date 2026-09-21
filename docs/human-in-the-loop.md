# Human-in-the-Loop Decisions

Jeeva can run an automatic arbiter for uncertain fields; your application owns the final human-review policy and every consequential side effect.

## Arbitrate before human review

```python
from jeeva import SequenceArbiter

arbiter = SequenceArbiter()
result = model.decide(
    ticket_text,
    TicketDecision,
    arbiter=arbiter,
)

if result.is_ambiguous:
    enqueue_human_review(
        context=ticket_text,
        suggestion=result.value,
        evidence=result.fields,
    )
else:
    dispatch_automatically(result.value)
```

The arbiter receives only fields whose initial decision is uncertain. If its replacement still falls below the active threshold, `result.is_ambiguous` remains `True` and the application can escalate to a person.

`Decision.calibration` answers whether calibration evidence exists:

- `"unvalidated"`: no fields have held-out calibration evidence.
- `"partial"`: some fields have held-out calibration evidence.
- `"validated"`: every field has evidence for the exact schema.

Each field's `threshold_source` separately reports `"manual"`, `"calibrated"`,
or `"default"`. An unvalidated threshold can still trigger an arbiter.

## Field-level policies

A decision still has boolean aggregate certainty. Each field resolves its active threshold as manual `tau`, then a fitted threshold when available, then the default:

```python
route = result.fields["department"]
risk = result.fields["risk"]

if route.is_certain and risk.value != "high":
    assign_queue(result.value["department"])
else:
    enqueue_review(ticket_text, result.value, result.fields)
```

Field evidence includes the candidate distribution, winning margin, active `tau`, threshold source, entropy metrics when available, certainty, calibration provenance, final resolver, and optional arbiter reasoning.

## Produce calibration evidence

Fit with representative held-out data:

```python
from jeeva import FitConfig

trained = base.fit(
    training_examples,
    schema=TicketDecision,
    validation=validation_examples,
    config=FitConfig(
        target_precision=0.97,
        minimum_calibration_examples=30,
    ),
)
```

Thresholds are stored per schema fingerprint and field. Changing field order, primitive kind, criteria, candidate values, or examples creates a new schema identity. Manual `tau` is excluded from that identity.

`target_precision` is an optimization objective over the supplied validation sample, not a production guarantee. Promotion still requires evaluation on a dataset representative of the deployment population.

## Audit records

Persist enough information to reconstruct the decision and improve the model:

```python
record = {
    "context": ticket_text,
    "prediction": result.value,
    "calibration": result.calibration,
    "is_certain": result.is_certain,
    "fields": {
        path: {
            "value": field.value,
            "distribution": dict(field.distribution),
            "margin": field.margin,
            "tau": field.tau,
            "threshold_source": field.threshold_source,
            "is_certain": field.is_certain,
            "calibration": field.calibration,
            "resolved_by": field.resolved_by,
            "reasoning": field.reasoning,
        }
        for path, field in result.fields.items()
    },
    "schema_fingerprint": result.telemetry["schema_fingerprint"],
}
```

Handle sensitive context according to your retention and access-control policy.

## Active learning

1. Collect human resolutions for ambiguous, unvalidated, and incorrect decisions.
2. Convert them into `Example` objects with dictionary targets.
3. Keep a stable, independent validation set.
4. Continue fitting from the current saved artifact.
5. Compare the candidate and current artifact on identical held-out data.
6. Promote by explicit application metrics, not training loss alone.

```python
current = DecisionModel("artifacts/ticket-model", device="cuda")
candidate = current.fit(
    new_training,
    schema=TicketDecision,
    validation=stable_validation,
)
candidate.save("artifacts/ticket-model-candidate")
```
