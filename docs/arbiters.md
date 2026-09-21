# Arbiters

Arbiters give decisions a second resolution strategy. Pass an arbiter directly to `DecisionModel.decide` or `decide_many`; by default, Jeeva invokes it only for fields whose initial `is_certain` value is `False`.

```python
from jeeva import DecisionModel, SequenceArbiter

model = DecisionModel("Qwen/Qwen2.5-1.5B-Instruct", device="mps")
arbiter = SequenceArbiter()

result = model.decide(
    ticket_text,
    TicketDecision,
    arbiter=arbiter,
)
```

The returned `Decision` is the final result after arbitration. Fields that were already certain are unchanged unless the arbiter explicitly opts into resolving every field.

## Resolution evidence

Every field records who produced its final value:

```python
field = result.fields["department"]

field.resolved_by  # "gatekeeper", "SequenceArbiter", ...
field.reasoning  # Optional explanation from a reasoning or remote arbiter
```

Execution telemetry records whether arbitration ran:

```python
result.telemetry["arbiter_invoked_fields"]
result.telemetry["arbiter_name"]
result.telemetry["arbiter_ms"]
```

If the arbiter's replacement still falls below the active threshold, the final field remains uncertain. A complete policy can therefore use a local arbiter first and retain human review as the final fallback:

```python
result = model.decide(ticket_text, TicketDecision, arbiter=arbiter)

if result.is_ambiguous:
    review_queue.put(result)
else:
    apply_decision(result.value)
```

## Built-in arbiters

### `SequenceArbiter`

`SequenceArbiter` scores each complete candidate phrase by length-normalized sequence likelihood. It generates no answer text.

By default it reuses the primary model and its prefilled context cache:

```python
from jeeva import SequenceArbiter

arbiter = SequenceArbiter()
result = model.decide(text, TicketDecision, arbiter=arbiter)
```

Give it a dedicated model when ambiguous cases should receive more capacity or a separately fitted artifact:

```python
arbiter = SequenceArbiter(
    model_or_id="Qwen/Qwen2.5-1.5B-Instruct",
    device="mps",
)
```

A dedicated model performs its own context prefill and remains loaded on the arbiter instance for later calls. Construct both the decision model and arbiter once, then reuse them.

An encoder primary cannot provide a causal KV cache. `SequenceArbiter()` and
`SequenceThinkingArbiter()` therefore require `model_or_id=...` when used with an
encoder `DecisionModel`; Jeeva rejects primary-model reuse before inference.
Cloud and custom arbiters that do not reuse the primary model remain compatible.

### `SequenceThinkingArbiter`

`SequenceThinkingArbiter` generates a bounded, verdict-free evidence trace,
then scores the finite candidates conditioned on that trace. By default it uses
single-token letter logits rather than generating and parsing an answer. Reasoning
generation is greedy by default so identical inputs remain reproducible:

```python
from jeeva import SequenceThinkingArbiter

arbiter = SequenceThinkingArbiter(
    max_thinking_tokens=64,
    readout="letter",
)
result = model.decide(text, TicketDecision, arbiter=arbiter)

print(result.fields["department"].reasoning)
```

The default remains a low-latency fallback: thinking runs only for uncertain
gatekeeper fields. Set `always_think=True` when every field should be reconsidered,
including fields the zero-shot gatekeeper marked certain:

```python
arbiter = SequenceThinkingArbiter(
    always_think=True,
    max_thinking_tokens=96,
    temperature=0.0,
)
result = model.decide(text, TicketDecision, arbiter=arbiter)

assert all(field.resolved_by == "SequenceThinkingArbiter" for field in result.fields.values())
```

Use `result.fields[name].resolved_by` to verify that thinking actually ran. A
`"gatekeeper"` value with `reasoning=None` means the field bypassed the thinking
arbiter. Setting `temperature` above zero enables nucleus sampling controlled by
`top_p`; keep `temperature=0.0` for classification and reproducible evaluation.

`readout="letter"` maps candidates to `A`, `B`, and so on, then compares those
next-token logits after the evidence trace. It does not sample or parse an answer.
This is the default and avoids candidate continuation passes, option-length effects,
and free-form output failures. Letter readout supports up to 26 candidates and
requires each letter to be one token for the arbiter model.

Use `readout="sequence"` when candidates should be scored through their full text
or the option set cannot use letter labels. Sequence readout retains the previous
length-normalized continuation-likelihood behavior.

Thinking resolves each field independently. It cannot guarantee a relationship
between fields unless that relationship is represented in the schema or enforced
by application validation. For example, a domain rule such as `PHI implies PII`
is not automatically available to either field's arbiter.

The reasoning is model-generated diagnostic material, not proof that the selected value is correct. Store or display it only under the same security and privacy controls as the source context.

### `CloudArbiter`

`CloudArbiter` sends only ambiguous fields to an OpenAI-compatible endpoint. It accepts an injected OpenAI-style client or callable; otherwise it uses the OpenAI SDK with `api_key` and optional `base_url`.
Install `jeeva-ai[cloud]` when Jeeva should construct the OpenAI SDK client. Injected
clients and callables do not require that extra.

```python
from jeeva import CloudArbiter

arbiter = CloudArbiter(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    api_key="local",
    base_url="http://localhost:8000/v1",
)
result = model.decide(text, TicketDecision, arbiter=arbiter)
```

A callable client can integrate another provider without coupling Jeeva to its SDK:

```python
def resolve_remotely(prompt: str) -> dict[str, object]:
    response = provider.generate(prompt)
    return {
        "probabilities": response.probabilities,
        "reasoning": response.reasoning,
    }


arbiter = CloudArbiter(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    client=resolve_remotely,
)
```

The provider response must contain a `probabilities` object with exactly one finite, nonnegative value for every candidate label. Values must sum to `1.0` and have one unique maximum; Jeeva derives the selected value from that complete distribution. Missing, extra, tied, invalid, or unavailable provider responses raise an error rather than producing a fabricated fallback.

Sending context to a remote endpoint is an application data-disclosure decision. Use a local arbiter when the context must stay on the deployment machine.

## Batch decisions

The same arbiter can resolve uncertain fields across a batch:

```python
results = model.decide_many(
    ticket_texts,
    TicketDecision,
    arbiter=SequenceArbiter(),
)
```

Batch evaluation currently processes contexts serially while reusing the loaded model and arbiter. Batch telemetry includes the number of items for which an arbiter produced replacements.

## Certainty and calibration

Arbiters use each field's active threshold:

- zero-shot fields use the default threshold;
- calibrated fields use the threshold learned from held-out examples for that schema and field.

This keeps arbitration aligned with `field.is_certain`. Calibration records where the threshold came from; it does not control whether the arbiter is available.

## Human review

There is no built-in `HumanArbiter` because queues, permissions, audit requirements, and irreversible actions belong to the application. Use `result.is_ambiguous` after the configured arbiter as the handoff to human review. See [Human-in-the-Loop Decisions](human-in-the-loop.md).

## Custom arbiters

Advanced integrations can subclass `BaseArbiter` and implement `name` and `resolve`. The resolver receives the selected field primitives—ambiguous fields by default, or every field when `resolve_all_fields` is `True`—the preliminary decisions, and the engine context needed by local model adapters. It returns replacement `DecisionResult` values keyed by field path.
Use a custom arbiter when resolution has reusable model behavior. Keep business authorization and side effects outside the arbiter so decision computation remains testable.
