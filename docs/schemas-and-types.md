# Schemas and Result Types

Jeeva has one public schema form: a `DecisionSchema` class containing `Noul`,
`Choice`, and `Score` declarations. The schema defines decision semantics; the
result value is a plain dictionary.

## Boolean decisions with `Noul`

```python
from jeeva import DecisionSchema, Noul


class OutageDecision(DecisionSchema):
    is_outage = Noul(
        "Does the message report a currently active production outage?",
        {
            True: "A production outage is currently active.",
            False: "No production outage is currently active.",
        },
        examples={
            True: ("Checkout is down for every customer.",),
            False: ("We had an outage last month.",),
        },
        tau=0.65,
    )
```

A `Noul` returns `True` or `False`. Its criteria give both outcomes explicit
semantics. Its certainty margin is `abs(P(true) - P(false))`.

## Finite choices with `Choice`

```python
from jeeva import Choice, DecisionSchema


class TicketDecision(DecisionSchema):
    department = Choice(
        "Choose the department that should handle the request.",
        {
            "billing": "Billing, invoices, charges, and refunds.",
            "support": "Product failures and technical troubleshooting.",
            "sales": "Purchasing, plans, and upgrades.",
        },
        examples={
            "billing": ("I was charged twice.",),
            "support": ("The API returns HTTP 500.",),
        },
        tau=0.35,
    )
```

The ordered criteria mapping is the complete set of stable result values and
their semantic descriptions. At least two unique values and descriptions are
required. A `Choice` margin is the difference between the largest and
second-largest candidate probabilities.

## Ordinal values with `Score`

```python
from jeeva import DecisionSchema, Score


class IncidentDecision(DecisionSchema):
    severity = Score(
        "Rate the operational severity.",
        {
            0: "No user-visible impact.",
            1: "Minor degradation.",
            2: "Material degradation.",
            3: "Critical outage.",
        },
        tolerance=0.5,
        tau=0.7,
    )
```

`Score` returns the expected value over its ordered numeric anchors:

```text
value = sum(level_probability × level_value)
```

Levels are sorted numerically and require unique, non-empty descriptions.
`field.winning_level` is the highest-probability anchor, while
`field.standard_deviation` measures distribution spread. Score certainty uses
normalized spread rather than top-two margin. `tolerance` defines correctness
for held-out threshold calibration; inference works without it.

## When to add examples

Examples are optional few-shot guidance rendered into the decision prompt. They
can improve accuracy when labels are domain-specific, easily confused, or not
well represented by the base model. They are not guaranteed to help: noisy,
redundant, or unbalanced examples can bias the result.

Examples also lengthen every affected decision lane, increasing inference
latency and cache memory. Start without them, then add a small, representative,
balanced set only when held-out evaluation shows an improvement.

## Decide

```python
result = model.decide("I was charged twice.", TicketDecision)

result.value
# {"department": "billing"}
```

`result.value` is `dict[str, bool | str | float]`. `result.fields` contains
evidence for each lane:

```python
field = result.fields["department"]
field.value
field.distribution
field.margin
field.certainty_score
field.score
field.entropy
field.normalized_entropy
field.winning_level  # Score only
field.standard_deviation  # Score only
field.is_certain
field.resolved_by
field.reasoning
field.tau
field.threshold_source  # "manual", "calibrated", or "default"
```

`Decision.is_certain` is true only when every final field is certain.
`Decision.is_ambiguous` is its inverse.


## Threshold precedence

The active certainty threshold uses this order:

1. The primitive's explicit `tau`
2. A threshold calibrated from held-out examples
3. The model default

Changing `tau` changes runtime escalation policy and does not invalidate a
trained artifact. Changing field names, primitive kinds, criteria, candidates,
Score levels or tolerance, or examples changes the semantic fingerprint. A
fitted encoder requires the original fingerprint and must be refitted after such
a change. A fitted causal model may evaluate the new schema dynamically, but no
stored calibration transfers to the new fingerprint and its quality must be
validated independently.

## Validation

Schema declarations fail early when:

- criteria are empty;
- `Choice` has fewer than two values;
- `Score` has fewer than two finite numeric levels;
- Score descriptions or level values are duplicated;
- Score tolerance is not a finite positive number;
- example keys do not match possible results;
- examples contain empty text;
- `tau` is outside `[0, 1]`;
- schemas exceed 16 fields, 64 candidates per field, or 128 candidates total;
- field prompts exceed 16,384 characters, option text exceeds 4,096 characters,
  or aggregate schema text exceeds 131,072 characters; or
- position-balanced letter readout receives more than 26 candidates.

Training targets are strict dictionaries. `Noul` targets must be actual
Booleans, `Choice` targets must be declared string keys, and `Score` targets
must be finite numbers inside the configured range. Unknown fields are
rejected. Targets may omit fields for partial-label training.
