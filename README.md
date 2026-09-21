<div align="center">

# Jeeva (ஜீவா)

### Turn a local language model into a finite decision engine — no generated JSON, no vendor API

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg?style=flat-square)](https://pypi.org/project/jeeva-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![CI](https://github.com/waseem07799/jeeva/actions/workflows/ci.yml/badge.svg)](https://github.com/waseem07799/jeeva/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg?style=flat-square)](https://github.com/waseem07799/jeeva/blob/main/LICENSE)

[Why Jeeva?](#why-jeeva) · [Quickstart](#quickstart) · [Arbiters](#know-when-to-escalate) · [Fine-tuning](#fit-jeeva-to-your-domain) · [Deployment](#save-once-deploy-anywhere) · [Documentation](#documentation)

</div>

Most application prompts are not asking an LLM to write. They are asking it to **choose**:

- Which team should own this ticket?
- Does this document contain personal data?
- Which policy applies?
- What should the agent do next?

The usual approach generates a JSON object token by token, then validates and repairs the text. Jeeva takes the shorter path: declare a semantic `DecisionSchema`, score its finite values directly, and get back a plain dictionary with field-level evidence.

```text
Causal runtime:  context → shared KV cache → score finite choices → dictionary
Encoder runtime: context + packed candidate markers → isolated scoring → dictionary
```

Bring either a compatible decoder-only causal model or a ModernBERT encoder from
Hugging Face. Causal models work zero-shot; encoders use a trainable candidate-marker
head and must be fitted before inference. Both run locally, and both generate
**zero output tokens** for the decision itself.

## Why Jeeva?

Generic classifiers rarely understand your taxonomy. Hosted LLMs can understand it, but add network latency, data-boundary concerns, and brittle generated JSON. Jeeva is for the middle ground: domain-specific decisions from a model you control.

| What you need | What Jeeva provides |
|---|---|
| A real application contract | `DecisionSchema` fields built from `Noul`, `Choice`, and `Score` |
| No parsing surprises | A constrained dictionary, not generated JSON text |
| Efficient local execution | Shared-KV causal lanes or true batched encoder scoring |
| A way to handle doubt | Manual `tau`, calibrated thresholds, field margins, and arbiters |
| Domain adaptation | Frozen-head, full, or LoRA fitting from dictionary labels or JSONL |
| Portable deployment | Self-describing full-weight, adapter, or merged artifacts |

Use Jeeva for routing, policy checks, document classification, safety decisions, workflow branching, or any task where the answer belongs to a known finite set. It is not a replacement for free-form generation.

## Installation

```bash
pip install jeeva-ai
```

Add the training extra when fitting or loading LoRA artifacts:

```bash
pip install "jeeva-ai[training]"
```

For memory-constrained causal training with a 4-bit backbone, install the QLoRA
extra instead:

```bash
pip install "jeeva-ai[qlora]"
```

Optional integrations:

```bash
pip install "jeeva-ai[multimodal]"
pip install "jeeva-ai[serve]"
pip install "jeeva-ai[cloud]"
```

## Quickstart

Define the decision you want—not the JSON you hope the model will generate:

```python
from jeeva import Choice, DecisionModel, DecisionSchema, Noul


class TicketDecision(DecisionSchema):
    department = Choice(
        "Choose the department that should handle the request.",
        {
            "billing": "Billing, invoices, charges, refunds, and payment disputes.",
            "support": "Product failures, outages, bugs, and technical troubleshooting.",
            "sales": "Purchasing, plans, upgrades, and commercial questions.",
        },
        tau=0.35,
    )
    urgent = Noul(
        "Does the ticket need immediate attention?",
        {
            True: "Immediate intervention is required.",
            False: "The normal response process is sufficient.",
        },
        tau=0.60,
    )
    risk = Choice(
        "Classify the operational risk if the request is delayed.",
        {
            "low": "Delay is unlikely to cause material impact.",
            "medium": "Delay may cause meaningful but contained impact.",
            "high": "Delay is likely to cause severe or widespread impact.",
        },
        tau=0.40,
    )


model = DecisionModel("Qwen/Qwen2.5-1.5B-Instruct", device="mps")
model.warmup()  # Optional: pay first-inference setup cost before serving traffic.
result = model.decide(
    "Production checkout is returning 500s for every customer.",
    TicketDecision,
)

result.value["department"]  # "support"
result.value["urgent"]  # True
result.value["risk"]  # "high"
result.fields["risk"].margin  # Evidence for the field-level decision
```

`result.value` is a plain dictionary whose keys and values are constrained by
`TicketDecision`. No JSON parsing or retry prompt is required.

For zero-shot inference, calibrate a schema once against the loaded model to remove
context-free candidate bias:

```python
calibrated_ticket = TicketDecision.calibrate(model)

result = model.decide(
    "Production checkout is returning 500s for every customer.",
    calibrated_ticket,
)
```

Calibration uses a fixed internal reference policy and is bound to that loaded
`DecisionModel`. Recalibrate after reloading the model. Calibrated schemas are rejected
by fitted or merged models because their trained scoring path already learns the
decision bias.

This is a prior correction, not a confidence booster: it preserves the ordering of
examples for each field but moves the decision threshold. It can narrow a margin, flip
a correct answer, or reduce accuracy when the model's context-free prior happens to be
useful for the target distribution. Compare raw and calibrated schemas on representative
labeled examples before choosing the calibrated schema for deployment.

### Candidate-position balancing

Causal models can prefer a letter or display position independently of candidate
meaning. This is most visible in smaller models, but model size does not
guarantee immunity: larger instruction-tuned models can still exhibit primacy,
recency, and label bias. Jeeva therefore enables candidate-position balancing
by default for the causal model's output head. Applications that have measured
their model and need the lowest latency or memory use can explicitly disable it:

```python
model = DecisionModel(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device="mps",
    candidate_position_balancing=False,
)
```

Encoder candidate-marker scoring and both causal metric scorers avoid letter
positions, so candidate-position balancing is inactive for them.

For a field with \(K\) candidates, balancing evaluates \(K\) cyclic orders in
one batched branch dispatch. Every semantic candidate occupies every letter
position once; Jeeva remaps the letter logits to semantic candidates, averages
them, and then computes the final probability distribution. The document
prefill is still shared, so total latency does not generally grow linearly with
the branch count; short-branch computation, activations, and expanded KV-cache
memory do increase with \(K\).

Inspect the mitigation and its cost through telemetry:

```python
result.telemetry["candidate_position_balancing"]  # True
result.telemetry["num_lanes"]  # Logical schema fields
result.telemetry["num_branches"]  # Physical order trials
result.telemetry["candidate_ordering"]["department"]
```

The per-field entry contains:

```python
{
    "strategy": "cyclic_position_balance",
    "trials": 3,
    "sensitivity": 0.42,
    "robust_margin": 0.0,
}
```

`sensitivity` measures how much the same semantic candidate's probability
changes when only its display position changes. For each candidate, Jeeva takes
its highest probability across the balanced orders minus its lowest, then
reports the largest change. A value of `0.0` means order had no measured effect;
`0.42` means at least one candidate moved by 42 percentage points; `1.0` is the
maximum possible change. Sensitivity does not select or replace the winner. It
diagnoses how presentation-dependent the underlying model remains after the
averaged result has been protected from fixed position bias.

`robust_margin` is the pooled winner's weakest top-two margin across the
individual orders. If that candidate loses in any order, the robust margin is
`0.0`, so the balanced result remains available but is not reported as certain.

`SequenceThinkingArbiter` enables the same mitigation by default for its letter
readout. Disable it independently when needed:

```python
from jeeva import SequenceThinkingArbiter

arbiter = SequenceThinkingArbiter(candidate_position_balancing=False)
```


Use one long-lived `DecisionModel` per loaded model and device:

```python
model = DecisionModel(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device="cuda",  # "auto", "cuda", "mps", or "cpu"
    dtype="bfloat16",  # "auto", "float16", "bfloat16", or "float32"
)
```

## Know when to escalate

By default, Jeeva invokes an arbiter only for uncertain fields:

```python
from jeeva import SequenceArbiter

arbiter = SequenceArbiter()
result = model.decide(
    ticket_text,
    TicketDecision,
    arbiter=arbiter,
)

if result.is_ambiguous:
    enqueue_human_review(result)
else:
    automate(result.value)
```

`SequenceArbiter` re-scores complete candidate phrases with the same local model and generates no answer text. `SequenceThinkingArbiter(always_think=True)` can deliberately reason over every field when the gatekeeper's raw certainty is not trusted. Jeeva also includes an OpenAI-compatible endpoint arbiter. The final field evidence shows which resolver won:

```python
field = result.fields["department"]
field.resolved_by  # "gatekeeper", "SequenceArbiter", ...
field.reasoning  # Optional arbiter explanation
```

`is_certain` and `is_ambiguous` are always boolean—even before fine-tuning. Zero-shot decisions use Jeeva's default margin threshold, so arbitration works from the first call.

Threshold evidence shows both the active value and where it came from:

```python
field = result.fields["department"]
field.tau  # Active threshold
field.threshold_source  # "manual", "calibrated", or "default"
result.calibration  # "validated", "partial", or "unvalidated"
```

An explicit primitive `tau` overrides a calibrated threshold. Calibration is
used only when the primitive does not declare `tau`; otherwise Jeeva falls back
to the model default.

See [Arbiters](https://github.com/waseem07799/jeeva/blob/main/docs/arbiters.md) for resolution strategies and [Calibration and Certainty](https://github.com/waseem07799/jeeva/blob/main/docs/calibration-and-tau.md) for threshold semantics.

## One model, many decisions

The model owns loaded weights, device, and dtype. Before fitting, the schema stays with the call, so the same zero-shot model can answer unrelated questions:

```python
class ApprovalDecision(DecisionSchema):
    approved = Noul(
        "Does the request satisfy the approval policy?",
        {
            True: "The request satisfies every required policy condition.",
            False: "The request violates or does not establish a required condition.",
        },
    )


class LanguageDecision(DecisionSchema):
    language = Choice(
        "Identify the message's primary language.",
        {
            "en": "The message is primarily English.",
            "es": "The message is primarily Spanish.",
            "fr": "The message is primarily French.",
        },
    )


approval = model.decide(request_text, ApprovalDecision)
language = model.decide(request_text, LanguageDecision)
```

A fitted encoder is bound to the exact schema used during fitting. Calling
`decide` or `decide_many` with renamed fields, changed instructions, changed
candidate descriptions, different candidate order, or changed examples raises
`SchemaError`; use the unchanged training schema or refit the encoder. A fitted
causal model can evaluate a new schema because its scorer derives candidates at
runtime, but the new schema has no stored calibration and must be evaluated before
production use. Runtime `tau` values may change without invalidating either
artifact.

Apply one schema to a batch with `decide_many`:

```python
results = model.decide_many(ticket_texts, TicketDecision)
```

Decision schemas contain only:

- `Noul("instructions", {True: "meaning", False: "meaning"}, examples=..., tau=...)`
  for Boolean decisions
- `Choice("instructions", {"stable_value": "semantic description", ...}, examples=...,
  tau=...)` for finite choices
- `Score("instructions", {0: "lowest", 1: "middle", 2: "highest"}, tolerance=...,
  tau=...)` for ordinal decisions whose result is a probability-weighted numeric value

`Score` keeps the complete level distribution and reports its winning level,
standard deviation, and spread-based certainty:

```python
from jeeva import DecisionSchema, Score


class SeverityDecision(DecisionSchema):
    severity = Score(
        "Rate the operational severity.",
        {
            0: "No user-visible impact.",
            1: "Minor degradation.",
            2: "Material degradation.",
            3: "Critical outage.",
        },
        tolerance=0.5,
    )


result = model.decide("Checkout is unavailable globally.", SeverityDecision)
result.value["severity"]  # A float in [0.0, 3.0]
result.fields["severity"].winning_level  # Highest-probability configured level
result.fields["severity"].standard_deviation
```

Instructions, stable candidate values, semantic descriptions, examples, and Score
tolerance define the semantic contract used by prompts, training, and artifact
compatibility. Runtime `tau` does not affect the fingerprint.

## Fit Jeeva to your domain

Zero-shot behavior is often enough to prove the workflow. When your labels reflect private products, policies, or operational rules, fit the loaded model in place on your own examples:

```python
from jeeva import DecisionModel, FitConfig

base = DecisionModel("Qwen/Qwen2.5-1.5B-Instruct", device="mps")
trained = base.fit(
    "data/tickets-train.jsonl",
    schema=TicketDecision,
    validation="data/tickets-eval.jsonl",
    config=FitConfig(epochs=3, learning_rate=2e-4),
)
trained.save("artifacts/ticket-model")
```

`fit` reuses the loaded backbone, updates this `DecisionModel` in place, and
returns the same object. Create separate `DecisionModel` instances when training
independent variants.

Causal training uses the model's own output head by default. Two learned metric
strategies are available explicitly:

```python
static_metric = DecisionModel(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device="mps",
).fit(
    examples,
    schema=TicketDecision,
    config=FitConfig(decoder_head="bilinear"),
)
contextual_metric = DecisionModel(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device="mps",
).fit(
    examples,
    schema=TicketDecision,
    config=FitConfig(decoder_head="contextual"),
)
```

`"bilinear"` compares the field query with context-free candidate embeddings.
`"contextual"` appends one isolated query branch and one isolated branch per
candidate to the shared document prefill, then scores their final hidden states
with the same normalized bilinear head. This lets each candidate representation
use the document and field instruction without letting candidates see one
another.

For a bidirectional encoder, select the architecture when constructing the raw model:

```python
base = DecisionModel(
    "answerdotai/ModernBERT-base",
    architecture="encoder",
    device="mps",
)
trained = base.fit(
    "data/tickets-train.jsonl",
    schema=TicketDecision,
    validation="data/tickets-eval.jsonl",
    config=FitConfig(epochs=3, batch_size=8),
)
```

Raw encoders intentionally reject `decide` and `warmup`: the candidate-marker head
has no useful behavior until `fit` has trained it. Saved artifacts record their runtime
architecture, so loading never requires an `architecture` argument.

Each JSONL row contains the context and a dictionary target:

```json
{"context":"I was charged twice.","target":{"department":"billing","urgent":true,"risk":"high"}}
```

Soft probability targets are explicit and candidate-keyed:

```json
{"context":"The request could be billing or support.","target":{"department":{"$distribution":{"billing":0.65,"support":0.30,"sales":0.05}},"urgent":false}}
```

Every declared candidate must appear, probabilities must be finite and
nonnegative, and their sum must be 1. Jeeva rejects malformed distributions
rather than normalizing them silently.

Flat label records are accepted too:

```json
{"context":"I was charged twice.","department":"billing","urgent":true,"risk":"high"}
```

Large JSONL datasets are indexed once and read from disk on demand; Jeeva keeps only a compact byte-offset index in memory.

For small or programmatically generated datasets, use `Example`:

```python
from jeeva import Example, TargetDistribution

examples = [
    Example(
        context="I was charged twice.",
        target={"department": "billing", "urgent": True, "risk": "high"},
    ),
    Example(
        context="Where can I download an invoice?",
        target={"department": "billing", "urgent": False, "risk": "low"},
    ),
]

soft_example = Example(
    context="The request could be billing or support.",
    target={
        "department": TargetDistribution({"billing": 0.65, "support": 0.30, "sales": 0.05}),
        "urgent": False,
    },
)
examples.append(soft_example)

trained = base.fit(examples, schema=TicketDecision)
```

Targets may omit fields for partial-label training, but unknown fields, values
outside a `Choice`, and Score values outside the configured range are rejected.
Hard and soft fields may be mixed in one example. Continuous Score labels are
linearly interpolated between adjacent levels. Training reports record
`hard_target_fields` and `soft_target_fields`.
With no explicit validation set, Jeeva makes a deterministic 80/20 split. Add a
top-level `"group"` to JSONL rows—or `group=` to `Example`—to keep related templates
in the same split and prevent template leakage. By default, Jeeva learns a
field-level certainty cutoff targeting 95% precision on held-out examples when at
least five examples are available for that field. Score threshold calibration
requires an explicit `tolerance`. An explicit primitive `tau` continues to override
a learned cutoff. Soft validation targets contribute to encoder temperature/Brier
calibration but not to hard correctness or target-precision thresholds.

`fit` returns a new `DecisionModel`; it does not mutate the receiver. Each fit
trains one explicit `DecisionSchema`.

## Save once, deploy anywhere

A saved Jeeva directory is self-describing. Load a trained artifact exactly like a base model:

```python
model = DecisionModel("artifacts/ticket-model", device="cuda")
result = model.decide(text, TicketDecision)
```

`adaptation="auto"` uses LoRA for causal and encoder models. On MPS or CUDA,
causal text models can explicitly select NF4 QLoRA so the frozen backbone stays
4-bit while LoRA parameters and Jeeva's learned decision head remain floating
point:

```python
trained = base.fit(
    examples,
    schema=TicketDecision,
    config=FitConfig(adaptation="qlora"),
)
trained.save("artifacts/ticket-model")
```

QLoRA uses NF4 with double quantization and FP16 compute when `dtype="auto"`.
It is currently limited to causal text models on MPS or CUDA. Saved adapters
load for ordinary unquantized inference, so deployment does not require
bitsandbytes. Continuing a QLoRA artifact with `adaptation="auto"` preserves
QLoRA training. For encoders, `adaptation="head"` freezes the backbone and
trains only its nonlinear candidate-marker scorer. Select `"full"` only when
the dataset is large enough to justify updating the complete backbone.

When a LoRA-backed deployment needs a standalone Hugging Face model directory,
merge the adapter into the base weights:

```python
trained.save("artifacts/ticket-model-merged", merge=True)
standalone = DecisionModel("artifacts/ticket-model-merged", device="cuda")
```

| Keep the adapter when… | Merge the weights when… |
|---|---|
| You plan to continue fitting it. | The runtime should not depend on PEFT. |
| Small artifacts and fast uploads matter. | The base model will not be available separately. |
| You need to inspect, disable, or swap adapters. | A registry or serving stack requires full weights. |

Merged artifacts are roughly the size of the base model and require additional memory during export. Quantized models and some adapter types cannot be merged; Jeeva fails rather than writing a partial artifact. The live trained model remains usable after export.

## Result evidence

Every call returns a `Decision`:

```python
result.value  # dict[str, bool | str]
result.fields  # Mapping[str, FieldDecision]
result.is_certain  # bool
result.is_ambiguous  # bool
result.calibration  # "validated", "partial", or "unvalidated"
result.telemetry  # execution metadata

field = result.fields["department"]
field.value
field.distribution
field.margin
field.score
field.entropy
field.normalized_entropy
field.tau  # Active certainty threshold
field.threshold_source  # "manual", "calibrated", or "default"
field.is_certain  # bool
field.calibration  # "validated" or "unvalidated"
field.resolved_by  # "gatekeeper" or the arbiter name
field.reasoning  # Optional arbiter explanation
```


## Configuration

`FitConfig` keeps the common path short while leaving training controls explicit:

| Field | Default | Purpose |
|---|---:|---|
| `epochs` | `3` | Training epochs |
| `progress_updates_per_epoch` | `10` | Maximum newline-delimited progress updates per epoch; `0` disables intermediate updates |
| `learning_rate` | architecture-tuned | Backbone/adapter rate: unused for head-only fitting, `2e-4` for LoRA/QLoRA, `2e-5` for full fitting |
| `decoder_head` | `"letter"` | Causal scorer: letter readout through the model output head, static `"bilinear"`, or `"contextual"` metric head |
| `head_learning_rate` | `1e-3` when used | Causal metric or encoder candidate-marker head rate |
| `scheduler` | `"cosine"` | `"cosine"`, `"linear"`, or `"constant"` |
| `minimum_learning_rate_ratio` | `0.05` | Final/minimum learning-rate ratio |
| `weight_decay` | `0.01` | AdamW weight decay |
| `maximum_gradient_norm` | `1.0` | Gradient-clipping norm |
| `adaptation` | `"auto"` | Defaults to `"lora"`; `"qlora"` selects causal MPS/CUDA NF4 training; encoder-only `"head"` freezes the backbone |
| `lora_rank` | `8` | LoRA rank |
| `lora_alpha` | `2 × rank` | LoRA scaling; inferred when omitted |
| `lora_dropout` | `0.05` | LoRA dropout |
| `batch_size` | `8` | Encoder examples per packed candidate-marker forward |
| `max_length` | `512` | Encoder token budget; candidate blocks are preserved and context tokens are truncated |
| `brier_weight` | `0.5` | Calibration-loss weight |
| `ordinal_weight` | `0.25` | Ordered-distribution loss weight for `Score` |
| `save_best` | `True` | Retain the lowest validation metric |
| `validation_fraction` | `0.2` | Automatic validation share |
| `seed` | random 32-bit value | Validation split and per-epoch shuffle seed; set explicitly for reproducible training |
| `target_precision` | `0.95` | Calibration threshold objective |
| `minimum_calibration_examples` | `5` | Minimum held-out evidence per field |

Device and dtype intentionally belong to `DecisionModel`, not `FitConfig`, because they describe loaded model state rather than a training recipe.

## Multimodal context

With the multimodal extra, pass a `MultimodalContext`; the schema and result contracts stay the same:

```python
from jeeva import MultimodalContext

context = MultimodalContext(text="Classify this document", images=[image])
result = model.decide(context, DocumentDecision)
```

Multimodal support is experimental. Benchmark the complete processor and inference path on your target hardware.

## Documentation

- [Quickstart](https://github.com/waseem07799/jeeva/blob/main/docs/quickstart.md)
- [Schemas and result types](https://github.com/waseem07799/jeeva/blob/main/docs/schemas-and-types.md)
- [Calibration and certainty](https://github.com/waseem07799/jeeva/blob/main/docs/calibration-and-tau.md)
- [Arbiters](https://github.com/waseem07799/jeeva/blob/main/docs/arbiters.md)
- [Training and fine-tuning](https://github.com/waseem07799/jeeva/blob/main/docs/training-and-finetuning.md)
- [Evaluation](https://github.com/waseem07799/jeeva/blob/main/docs/evaluation.md)
- [Production serving](https://github.com/waseem07799/jeeva/blob/main/docs/production-serving.md)
- [Human-in-the-loop policies](https://github.com/waseem07799/jeeva/blob/main/docs/human-in-the-loop.md)
- [Compatibility contract](https://github.com/waseem07799/jeeva/blob/main/docs/compatibility.md)
- [Security policy](https://github.com/waseem07799/jeeva/blob/main/SECURITY.md)
- [Changelog](https://github.com/waseem07799/jeeva/blob/main/CHANGELOG.md)

## Development

Install the development environment and run the deterministic suite:

```bash
uv sync --locked --extra training --extra serve --extra multimodal
uv run pytest
```

Jeeva's repository tests do not download or execute pretrained checkpoints. Public
lifecycle tests use mocked runtimes and trainers; numerical training tests use local
synthetic components. Model quality and hardware measurements belong to an application's
pinned evaluation protocol, not the package test suite.

## License

Apache 2.0. See [LICENSE](https://github.com/waseem07799/jeeva/blob/main/LICENSE).
