# Jeeva Documentation

Jeeva turns a compatible local causal language model or bidirectional encoder into a finite decision model.

```python
from jeeva import Choice, DecisionModel, DecisionSchema, Noul


class RouteDecision(DecisionSchema):
    route = Choice(
        "Choose the team that should handle the request.",
        {
            "billing": "Billing and payment requests.",
            "support": "Technical support requests.",
            "sales": "Purchasing and upgrade requests.",
        },
    )
    urgent = Noul(
        "Does the request need immediate attention?",
        {
            True: "Immediate attention is required.",
            False: "Normal handling is sufficient.",
        },
        tau=0.6,
    )


model = DecisionModel("Qwen/Qwen2.5-1.5B-Instruct", device="auto")
result = model.decide("Checkout is down for every customer.", RouteDecision)
print(result.value)
```

## Guides

- [Quickstart](quickstart.md): install, define a schema, decide, fit, and save.
- [Schemas and Result Types](schemas-and-types.md): `DecisionSchema`, `Noul`, `Choice`, and dictionary results.
- [Arbiters](arbiters.md): local re-scoring, bounded reasoning, remote resolution, and human fallback.
- [Training and Fine-Tuning](training-and-finetuning.md): dictionary examples, `FitConfig`, calibration, and artifacts.
- [Evaluation](evaluation.md): held-out accuracy, calibration coverage, and promotion criteria.
- [Production Serving](production-serving.md): lifecycle ownership and HTTP integration.
- [Human in the Loop](human-in-the-loop.md): application-level review policies.
- [Compatibility Contract](compatibility.md): capability requirements, evidence boundaries, and application qualification.
- [Release Checklist](release-checklist.md): deterministic verification and trusted publication.

## Core rules

1. Construct `DecisionModel` once for a model/device pair.
2. Pass a `DecisionSchema` class to `decide` or `decide_many`.
3. Read the dictionary answer from `result.value`.
4. Set primitive `tau` when application experience should control escalation.
5. Pass an arbiter when uncertain fields need automatic resolution.
6. Inspect `threshold_source` and `calibration` before automating.
7. Fine-tune one schema through `model.fit(..., schema=..., config=FitConfig(...))`.
8. Save and load self-describing artifact directories through `DecisionModel`.

Raw models expose only the architecture choice; fitted scorers, heads, and execution
details are artifact-owned internals.
