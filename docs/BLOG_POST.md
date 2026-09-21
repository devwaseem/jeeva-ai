# Stop Generating JSON for Finite Decisions

Many LLM tasks do not need generation. Routing a ticket, assigning an invoice type, checking a policy proposition, and choosing an agent branch all have finite answers known before inference begins.

Generating JSON for those tasks adds a sequential decoder loop and a parser around a problem that is fundamentally classification.

Jeeva takes a different path: it executes the schema directly against model
representations. A causal runtime prefills the context once and reuses its KV cache,
while a fitted encoder scores every field candidate in one packed forward pass.
The application receives a constrained dictionary rather than model-authored JSON.

## One semantic contract

```python
from jeeva import Choice, DecisionModel, DecisionSchema, Noul


class TicketDecision(DecisionSchema):
    department = Choice(
        "Choose the department that should handle the request.",
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
    risk = Choice(
        "Classify the operational risk if the request is delayed.",
        {
            "low": "Delay is unlikely to cause material impact.",
            "medium": "Delay may cause meaningful but contained impact.",
            "high": "Delay is likely to cause severe or widespread impact.",
        },
    )


model = DecisionModel("Qwen/Qwen2.5-1.5B-Instruct", device="auto")
result = model.decide(ticket_text, TicketDecision)
```

For a bidirectional encoder, the same interface applies after fitting:

```python
encoder = DecisionModel(
    "answerdotai/ModernBERT-base",
    architecture="encoder",
    device="auto",
)
trained_encoder = encoder.fit("data/tickets.jsonl", schema=TicketDecision)
```

Unlike a causal model, a raw encoder must be fitted before inference.

The schema constrains every output and states the criterion for each decision.
The result keeps the dictionary answer separate from its evidence:

```python
result.value
result.fields["department"].distribution
result.fields["department"].margin
result.fields["department"].tau
result.fields["department"].threshold_source
result.is_certain
```

The same model can answer another schema on the next call. Device and dtype remain model state; schemas remain request contracts.

## Certainty works zero-shot

A score margin is not automatically a correctness probability, but it can still drive an ambiguity gate. Jeeva always reports boolean certainty by comparing each field's margin with its active threshold. Zero-shot decisions use the default threshold, so arbiters work before fitting:

```python
from jeeva import SequenceArbiter

result = model.decide(
    ticket_text,
    TicketDecision,
    arbiter=SequenceArbiter(),
)

if result.is_ambiguous:
    enqueue_human_review(result)
else:
    automate(result.value)
```

Only uncertain fields reach the arbiter, and `field.resolved_by` records the final resolver. `field.threshold_source` reports whether certainty used a manual, calibrated, or default threshold.

## Fine-tuning without a second API

Training uses the same model and schema:

```python
from jeeva import Example, FitConfig

examples = [
    Example(
        context="I was billed twice.",
        target={"department": "billing", "urgent": True, "risk": "high"},
    ),
]

trained = model.fit(
    examples,
    schema=TicketDecision,
    validation=held_out_examples,
    config=FitConfig(
        epochs=3,
        learning_rate=2e-4,
        scheduler="cosine",
        lora_rank=8,
    ),
)
trained.save("artifacts/ticket-model")
```

`fit` produces a new `DecisionModel`. A saved directory records its runtime
architecture, fitted weights or adapter, tokenizer assets, base model reference,
fit configuration, schema descriptors, training report, and calibration evidence.

Loading the artifact requires no scorer or head choice:

```python
production = DecisionModel("artifacts/ticket-model", device="cuda")
```

The artifact owns its execution details. Users own the model, schemas, examples, and application policy.

## Application-controlled escalation

Primitive `tau` values make application experience explicit:

```python
class FraudDecision(DecisionSchema):
    suspicious = Noul(
        "Does the transaction show indicators of fraud?",
        {
            True: "The transaction shows one or more fraud indicators.",
            False: "The transaction shows no fraud indicators.",
        },
        tau=0.80,
    )
```

Manual `tau` wins over calibration and the engine default. It is excluded from
the semantic fingerprint, so operators can tune escalation without retraining.

## The practical boundary

Jeeva is a good fit when:

- outputs are finite and known in advance;
- local model ownership matters;
- typed application integration matters;
- generating and parsing JSON is unnecessary work;
- the team can evaluate on representative held-out data.

It is not a replacement for free-form generation. It is also not a reason to trust unvalidated model scores. The useful boundary is narrow and explicit: finite typed decisions, local execution, measurable evidence.
