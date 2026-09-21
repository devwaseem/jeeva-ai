# Quickstart

## Install

```bash
pip install jeeva-ai
```

Install `jeeva-ai[training]` when fitting or loading LoRA-trained artifacts.

## Define a decision

```python
from jeeva import Choice, DecisionModel, DecisionSchema, Noul


class TicketDecision(DecisionSchema):
    queue = Choice(
        "Choose the team that should handle the request.",
        {
            "billing": "Billing, invoices, charges, and refunds.",
            "support": "Product failures and technical troubleshooting.",
            "sales": "Purchasing, plans, and upgrades.",
        },
        examples={
            "billing": ("I was charged twice.",),
            "support": ("Checkout returns HTTP 500.",),
        },
        tau=0.35,
    )
    urgent = Noul(
        "Does the request need immediate attention?",
        {
            True: "Immediate attention is required.",
            False: "Normal handling is sufficient.",
        },
        tau=0.60,
    )
    sentiment = Choice(
        "Classify the author's sentiment.",
        {
            "negative": "The author expresses negative sentiment.",
            "neutral": "The author expresses neutral sentiment.",
            "positive": "The author expresses positive sentiment.",
        },
    )


model = DecisionModel("Qwen/Qwen2.5-1.5B-Instruct", device="auto")
model.warmup()  # Optional for latency-sensitive applications.
result = model.decide(
    "Checkout has returned HTTP 500 for every customer since 09:00.",
    TicketDecision,
)

print(result.value["queue"])
print(result.value["urgent"])
print(result.fields["queue"].distribution)
print(result.fields["queue"].tau)
print(result.fields["queue"].threshold_source)
```

The model owns device and dtype. The schema defines finite decision lanes;
`result.value` is a plain dictionary.

```python
class RefundDecision(DecisionSchema):
    eligible = Noul(
        "Does the request meet the refund policy?",
        {
            True: "Every required refund-policy condition is satisfied.",
            False: "At least one refund-policy condition is not satisfied.",
        },
    )


refund = model.decide(refund_request, RefundDecision)
tickets = model.decide_many(ticket_texts, TicketDecision)
```


## Add an arbiter

Pass an arbiter to resolve uncertain fields:

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
    dispatch(result.value)
```

Manual primitive `tau` values control escalation directly. Without one, Jeeva
uses a calibrated threshold when available, then the model default.

## Fit and save

For large datasets, use JSONL paths directly:

```python
from jeeva import FitConfig

trained = model.fit(
    "data/tickets-train.jsonl",
    schema=TicketDecision,
    validation="data/tickets-eval.jsonl",
    config=FitConfig(epochs=3, learning_rate=2e-4),
)
```

Each line is `{\"context\": \"...\", \"target\": {...}}`. Jeeva validates and indexes the files, then reads records from disk on demand.

For small in-memory datasets:

```python
from jeeva import Example, FitConfig

examples = [
    Example(
        context="Please send last month's invoice.",
        target={"queue": "billing", "urgent": False, "sentiment": "neutral"},
    ),
    Example(
        context="Production is completely unavailable.",
        target={"queue": "support", "urgent": True, "sentiment": "negative"},
    ),
]

trained = model.fit(
    examples,
    schema=TicketDecision,
    config=FitConfig(
        epochs=3,
        learning_rate=2e-4,
        scheduler="cosine",
        lora_rank=8,
    ),
)
trained.save("artifacts/ticket-model")
```

`fit` returns a new model. The original remains unchanged. Representative
held-out examples supplied through `validation=` drive field calibration.

```python
loaded = DecisionModel("artifacts/ticket-model", device="cuda")
result = loaded.decide(new_ticket, TicketDecision)
```

The artifact determines its trained scoring implementation automatically. There is no public mode or head selection.

## Next steps

- [Schemas and result types](schemas-and-types.md)
- [Training and fine-tuning](training-and-finetuning.md)
- [Evaluation](evaluation.md)
- [Production serving](production-serving.md)
