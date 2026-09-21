# Production Serving

Create one long-lived `DecisionModel` per process/device and reuse it across requests. Model loading belongs in application startup, not a request handler.

## FastAPI example

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from jeeva import Choice, DecisionModel, DecisionSchema, Noul, SequenceArbiter


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
    )


class TicketRequest(BaseModel):
    text: str


class TicketResponse(BaseModel):
    decision: dict[str, bool | str]
    is_certain: bool
    calibration: str
    margins: dict[str, float]


model: DecisionModel | None = None
arbiter: SequenceArbiter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, arbiter
    model = DecisionModel("artifacts/ticket-model", device="cuda")
    model.warmup()
    arbiter = SequenceArbiter()
    yield
    model = None
    arbiter = None


app = FastAPI(lifespan=lifespan)


@app.post("/tickets/decide", response_model=TicketResponse)
def decide_ticket(request: TicketRequest) -> TicketResponse:
    assert model is not None and arbiter is not None
    result = model.decide(request.text, TicketDecision, arbiter=arbiter)
    return TicketResponse(
        decision=result.value,
        is_certain=result.is_certain,
        calibration=result.calibration,
        margins={path: field.margin for path, field in result.fields.items()},
    )
```



## Concurrency

Construct the arbiter during application startup as well. A local arbiter may reuse the decision model or keep a dedicated model loaded; do not recreate it per request. Benchmark concurrent access for the selected model, device backend, and server worker strategy. Multiple web workers generally load multiple model copies. On GPU or Apple Silicon, start with one worker per loaded model and introduce request queuing or batching based on measured throughput and memory.

Use `decide_many` when several contexts sharing one schema are available together:

```python
results = model.decide_many(request_texts, TicketDecision)
```

Do not move a loaded model between devices per request. Create separate model owners when deployment requires separate devices.

## Artifact deployment

Deploy the exact directory produced by `DecisionModel.save`. Loading requires no public mode, head, or scorer flags:

```python
model = DecisionModel("/models/ticket-model", device="cuda", dtype="bfloat16")
```

Adapter artifacts require `jeeva-ai[training]` because PEFT attaches the saved LoRA weights at load time. A directory produced with `save(..., merge=True)` contains standalone model weights and loads without PEFT, at the cost of full-model storage.

Prefer an adapter artifact when the base model is already managed by the serving environment, storage efficiency matters, or the same adapter will be fitted again. Prefer a merged artifact when the deployment must be self-contained, PEFT is not allowed in the runtime, or the serving platform only accepts ordinary full-weight Transformers models.

Merging is a packaging decision, not an accuracy improvement. Keep the adapter artifact as the training and rollback source even when production deploys the merged copy.

Treat artifact promotion as an atomic directory/version change. Retain the previous artifact for rollback and perform a real decision smoke test after loading the candidate.

## Response semantics

Expose `is_certain` as a boolean for routing and arbiter decisions, and expose `calibration` separately so clients know whether the active thresholds are validated, partially validated, or the zero-shot defaults.

Avoid exposing raw telemetry or distributions to untrusted clients unless required; they may reveal model or schema details. Keep full evidence in protected audit logs.

## Health checks

A deeper deployment smoke check should run a representative `decide` call and validate the dictionary response before traffic is shifted.
