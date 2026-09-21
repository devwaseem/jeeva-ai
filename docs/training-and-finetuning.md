# Training and Fine-Tuning

`DecisionModel.fit` supports both causal language models and bidirectional encoders.
Training and inference use the same `DecisionSchema`; artifacts retain the runtime
architecture and fitted scorer.

## Install

```bash
pip install "jeeva-ai[training]"
```

For QLoRA training on MPS or CUDA:

```bash
pip install "jeeva-ai[qlora]"
```

## Build examples

Each `Example` contains a context and a partial or complete dictionary target.
The schema is explicit on `fit`.

```python
from jeeva import (
    Choice,
    DecisionModel,
    DecisionSchema,
    Example,
    Noul,
    TargetDistribution,
)


class TicketDecision(DecisionSchema):
    department = Choice(
        "Choose the department that should handle the request.",
        {
            "billing": "Billing, invoices, charges, and refunds.",
            "support": "Product failures and technical troubleshooting.",
            "sales": "Purchasing, plans, and upgrades.",
        },
    )
    urgent = Noul(
        "Does the request need immediate attention?",
        {
            True: "Immediate attention is required.",
            False: "Normal handling is sufficient.",
        },
    )


training = [
    Example(
        context="I was billed twice for the same month.",
        target={"department": "billing", "urgent": True},
    ),
    Example(
        context="Where can I download my invoice?",
        target={"department": "billing", "urgent": False},
    ),
    Example(
        context="Checkout has returned 500 since the latest deploy.",
        target={"department": "support", "urgent": True},
    ),
]
```

Hard and soft fields may be mixed explicitly:

```python
ambiguous = Example(
    context="I need help with a charge I do not recognize.",
    target={
        "department": TargetDistribution({"billing": 0.65, "support": 0.30, "sales": 0.05}),
        "urgent": False,
    },
)
```

Input validation happens before training. `Noul` hard labels must be Booleans,
`Choice` hard labels must be declared keys, `Score` hard labels must be finite
numbers in the configured range, and unknown fields are rejected. A
`TargetDistribution` must name every candidate exactly once with finite,
nonnegative probabilities summing to 1. A continuous Score target is linearly
interpolated between its adjacent anchors.

## Choose a runtime architecture

Causal models are the default and can decide zero-shot before fitting:

```python
base = DecisionModel("Qwen/Qwen2.5-1.5B-Instruct", device="mps")
```

Bidirectional encoders require an explicit architecture on the raw model:

```python
base = DecisionModel(
    "answerdotai/ModernBERT-base",
    architecture="encoder",
    device="mps",
)
```

Encoder inference packs every field and candidate into one suffix, scores the
contextualized candidate markers in one model pass, and supports candidate sets
larger than 26. Only document tokens are truncated; a schema whose complete
candidate suffix does not fit fails before execution. A raw encoder must be fitted
before `decide`, `decide_many`, or `warmup`.

## Large JSONL datasets

Pass training and evaluation paths directly to `fit`; the `DecisionSchema`
validates every target:

```python
base = DecisionModel("Qwen/Qwen2.5-1.5B-Instruct", device="mps")
trained = base.fit(
    "data/tickets-train.jsonl",
    schema=TicketDecision,
    validation="data/tickets-eval.jsonl",
    config=FitConfig(
        epochs=3,
        learning_rate=2e-4,
        scheduler="cosine",
    ),
)
```

The decoder uses letter readout through its model output head by default. Select
a learned metric strategy when candidates should not depend on single-token letters:

```python
static_metric = FitConfig(decoder_head="bilinear", head_learning_rate=1e-3)
contextual_metric = FitConfig(decoder_head="contextual", head_learning_rate=1e-3)
```

`"bilinear"` uses context-free candidate embeddings. `"contextual"` encodes the
field query and every candidate in separate branches appended to the same
document prefill, then compares those representations with the normalized
bilinear head. Candidate branches remain isolated from one another. Encoders
always use their candidate-marker head; `decoder_head` does not apply.

Use one JSON object per line. The canonical form nests labels under `target`:

```json
{"context":"I was billed twice.","target":{"department":"billing","urgent":true}}
{"context":"Where is my invoice?","target":{"department":"billing","urgent":false}}
```

Soft JSON targets use an explicit `$distribution` tag:

```json
{"context":"The request could be billing or support.","target":{"department":{"$distribution":{"billing":0.65,"support":0.30,"sales":0.05}},"urgent":false}}
```

Tagged `Noul` distributions use `Yes` and `No`; tagged `Score` distributions use
the canonical level labels shown by the schema.

Flat records are accepted for existing datasets:

```json
{"context":"I was billed twice.","department":"billing","urgent":true}
```

`context` is the canonical input key; `text`, `document`, `ticket`, and `input` are accepted aliases. A nested `target` is recommended because it cannot collide with input metadata.

Jeeva scans the file once to validate every row and build a compact byte-offset index. Training and evaluation then read individual records from disk on demand. The full dataset is never converted into a list of Python dictionaries or `Example` objects, so memory scales with the offset index rather than record content.

If `validation` is omitted, `validation_fraction` creates on-disk training and
validation views without copying records. Add a top-level `"group"` to keep
every record from a template, source, or subject in one split. Every `FitConfig`
gets a fresh random 32-bit `seed` by default, and that resolved seed is printed
and recorded in the training report and artifact. Set it explicitly when a run
must be reproducible:

```python
trained = base.fit(
    "data/tickets.jsonl",
    schema=TicketDecision,
    config=FitConfig(validation_fraction=0.2, seed=42),
)
```

Every fit uses one explicit schema, whether its examples are in JSONL or memory.

## Fit, save, and load

```python
base = DecisionModel(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device="mps",
    revision="0123456789abcdef",  # immutable Hub commit
)
trained = base.fit(training, schema=TicketDecision)
trained.save("artifacts/ticket-model")

loaded = DecisionModel("artifacts/ticket-model", device="mps")
result = loaded.decide("The API is down.", TicketDecision)
```

`fit` reuses the loaded backbone, updates the `DecisionModel` in place, and returns
that same object. This avoids holding separate inference, training, and reloaded
copies of the model in accelerator memory. Create separate `DecisionModel`
instances when training independent variants. QLoRA is the exception: changing
to a 4-bit representation requires releasing the loaded runtime and loading the
quantized model, but Jeeva does not keep both copies resident.

The save destination must not exist. Jeeva writes and validates a sibling staging
directory, then renames it into place so callers never observe a partial artifact. The
resulting directory is a
complete Jeeva artifact: it records the runtime architecture, scorer, weight
kind, tokenizer assets, fit configuration, known schema descriptors, training
report, calibration evidence, creation timestamp, requested and resolved model/tokenizer
IDs and revisions, offline-loading policy, and the exact Jeeva, Python, Torch,
Transformers, and applicable PEFT versions.

Artifact loading is fail-closed on the manifest format and schema protocol.
Recorded dependency versions are diagnostic provenance rather than equality
constraints: patching a dependency does not itself invalidate an artifact. Test
the artifact in the target environment before promotion.

Omitting `tokenizer_revision` makes it inherit `revision`; set it explicitly only when the
tokenizer is intentionally pinned elsewhere. `local_files_only=True` prevents network
resolution. Saved artifacts own their recorded revisions and reject constructor revision
overrides, so a promoted artifact cannot silently drift to another repository snapshot.

### Standalone merged export

Adapter-backed models can be exported as standalone full weights. Set
`merge=True` when deployment should not depend on PEFT:

```python
trained.save("artifacts/ticket-model-merged", merge=True)
loaded = DecisionModel("artifacts/ticket-model-merged", device="cuda")
```

Jeeva performs a safe temporary merge, saves only the resulting base-model weights, and unmerges the live `trained` object afterward. The merged artifact retains the Jeeva manifest, schemas, training report, and calibration evidence, but it does not contain a separate adapter.

All backbone, merged-model, and adapter weights must use Safetensors. Jeeva passes
`use_safetensors=True` to Transformers loaders, requires
`adapter_model.safetensors` for local PEFT adapters, and rejects pickle-backed
`.bin` checkpoints before model construction.

Choose the artifact deliberately:

| Keep the adapter artifact when… | Export merged weights when… |
|---|---|
| More fitting will continue from this adapter. | This is a final, standalone deployment artifact. |
| Adapter swapping, disabling, or inspection is useful. | The serving environment should not depend on PEFT. |
| Storage, upload time, or artifact versioning cost matters. | The serving stack requires an ordinary full-weight Hugging Face model. |
| The base model is available to the runtime. | The runtime cannot fetch or mount the original base model separately. |
| The model is quantized or merge compatibility is unknown. | Merge compatibility was tested for this exact model, dtype, and adapter. |

The recommended lifecycle is adapter-first: train, evaluate, and retain the adapter artifact; create a merged copy only at the final deployment boundary. If you may need both rollback-friendly continued training and standalone serving, keep both artifacts.

The tradeoffs are:

- full-model disk size instead of adapter-sized storage;
- additional export time and working memory;
- no adapter disabling, swapping, or exact adapter continuation from the merged directory;
- unsupported quantized or adapter configurations fail with `ArtifactError`;
- loading is standalone and does not require PEFT.

When both deployment and further adapter training matter, save two directories:

```python
trained.save("artifacts/ticket-model-adapter")
trained.save("artifacts/ticket-model-merged", merge=True)
```

## Validation and calibrated certainty

For production decisions, pass representative held-out examples:

```python
trained = base.fit(
    training,
    schema=TicketDecision,
    validation=validation_examples,
)
```

Jeeva derives field-level certainty thresholds for the requested
`target_precision`. `Choice` and `Noul` calibrate against categorical
correctness. `Score` calibrates against `abs(prediction - target) <= tolerance`
and therefore requires an explicit tolerance to produce a validated threshold.
The semantic fingerprint includes field names, primitive kinds, criteria,
candidates, Score levels and tolerance, and examples. It excludes `tau`, which
is a runtime escalation policy.

If `validation` is omitted, `validation_fraction` creates a deterministic split.
A field needs at least `minimum_calibration_examples` held-out labels before it
can be marked validated.

For an uncalibrated field, certainty uses an explicit primitive `tau` or the
default threshold:

```python
field = result.fields["department"]
field.threshold_source  # "manual", "calibrated", or "default"
field.tau
```

## Advanced tuning with `FitConfig`

`FitConfig` is the only public training configuration object:

```python
from jeeva import FitConfig

config = FitConfig(
    epochs=5,
    progress_updates_per_epoch=10,
    learning_rate=1e-4,
    decoder_head="letter",  # "letter", "bilinear", or "contextual"
    head_learning_rate=1e-3,  # metric or encoder marker head
    scheduler="cosine",  # "cosine", "linear", or "constant"
    minimum_learning_rate_ratio=0.05,
    weight_decay=0.01,
    maximum_gradient_norm=1.0,
    adaptation="auto",  # "auto", "head", "full", "lora", or causal "qlora"
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    batch_size=8,  # encoder document-field batch size
    max_length=512,  # prefix + packed candidate-block budget
    brier_weight=0.5,
    ordinal_weight=0.25,
    save_best=True,
    early_stopping_patience=2,
    validation_fraction=0.2,
    seed=42,  # omit for a fresh random 32-bit seed
    target_precision=0.95,
    minimum_calibration_examples=20,
)

trained = base.fit(
    training,
    schema=TicketDecision,
    validation=validation,
    config=config,
)
```

### Training objective

All fields use soft-target cross-entropy plus a weighted Brier term. `Score`
adds a weighted ranked-probability loss over cumulative level distributions;
numeric interval widths weight mistakes on unevenly spaced levels.
`brier_weight` and `ordinal_weight` configure these terms.

### Progress output

Training announces each epoch immediately, then prints at most
`progress_updates_per_epoch` newline-delimited updates with completed samples,
percentage, running composite loss, learning rate, elapsed time, and ETA. New
lines render reliably in notebooks and captured logs. Set the value to `0` to
disable intermediate updates; epoch start, epoch summary, and validation
announcements remain visible.

### Schedulers

- `"cosine"`: smooth decay to `minimum_learning_rate_ratio`; the general default.
- `"linear"`: linear decay to the minimum ratio.
- `"constant"`: keep the initial learning rate.

For causal decoders, `decoder_head="letter"` trains through single-token letter
logits from the model output head. `"bilinear"` and `"contextual"` select
normalized bilinear metric heads;
`head_learning_rate` controls that head. The static bilinear scorer dispatches
one field-query branch and compares it with cached context-free candidate
embeddings. The contextual scorer dispatches \(K+1\) rows for a field with
\(K\) candidates: one query branch and one isolated candidate branch per option,
all extending the shared document prefill. Encoders always use the
candidate-marker head, whose rate is also controlled by `head_learning_rate`.
Head-only encoder adaptation leaves the backbone frozen. The default head rate
is `1e-3`; active parameter groups follow the selected scheduler and minimum
ratio.

### Adaptation controls

`adaptation="auto"` selects LoRA for new causal and encoder models. For causal
text models on MPS or CUDA, `"qlora"` loads the frozen backbone in 4-bit NF4
with double quantization and trains ordinary LoRA parameters. Automatic dtype
selection uses FP16 compute. The learned bilinear/contextual head remains
floating point. Causal training already processes one labeled example per step;
keep source documents and candidate sets compact on memory-constrained Apple
Silicon. QLoRA reduces backbone-weight memory, not activation or branch
KV-cache memory.

QLoRA saves a normal PEFT adapter. Inference loads that adapter over an
unquantized base model and does not require bitsandbytes. Calling
`DecisionModel.save(..., merge=True)` therefore merges through the unquantized
runtime model; the internal live 4-bit trainer refuses direct merging.
Continuing a QLoRA artifact with `adaptation="auto"` keeps using QLoRA.

For encoders, select `"head"` to freeze the backbone and train only the nonlinear
candidate-marker scorer. Set `"full"` explicitly only when the dataset is large
enough to update the complete backbone. `lora_rank`, `lora_alpha`, and
`lora_dropout` control adapter capacity.

Encoder training randomly permutes both fields and candidates on every batch, then maps
logits back to stable semantic values. Packed candidates are isolated by the attention
mask and share the same position origin, so one candidate cannot use another candidate
or its packing slot as a shortcut. `Score` additionally uses ordinal loss weighted by
the numeric distance between anchors.

### Device and dtype

Device and dtype are not training configuration. They belong to the loaded model:

```python
base = DecisionModel("Qwen/Qwen2.5-1.5B-Instruct", device="cuda", dtype="bfloat16")
trained = base.fit(training, schema=TicketDecision, config=config)
```

This prevents fit-time settings from silently moving or recasting an already loaded model.

## Partial labels

An example may label only the known fields:

```python
training = [
    Example(ticket_text, {"department": "support", "urgent": True}),
    Example(invoice_text, {"department": "billing"}),
]

trained = base.fit(training, schema=TicketDecision)
```

Unlabeled lanes are masked from that example's loss. Unknown fields are errors.

## Continue fitting a saved adapter

Load a saved artifact and call `fit` again:

```python
current = DecisionModel("artifacts/ticket-model", device="cuda")
updated = current.fit(
    new_examples,
    schema=TicketDecision,
    validation=new_validation,
    config=config,
)
updated.save("artifacts/ticket-model-next")
```

The existing adapter is loaded as trainable rather than replaced by a fresh adapter.

## Reading the training report

```python
report = trained.training_report
report["hard_target_fields"]
report["soft_target_fields"]
```

The report is also written to the artifact manifest. Soft targets may train the
policy and contribute to encoder temperature/Brier calibration, but hard
correctness and target-precision thresholds use hard reference outcomes only.
The report's exact metrics reflect the training and validation data supplied to
the run; do not use training loss as a substitute for held-out task evaluation.
