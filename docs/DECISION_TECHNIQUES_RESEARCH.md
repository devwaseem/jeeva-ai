# Decision Technique Research Notes

This note records the design constraints behind Jeeva's current public API. Decoder
training exposes three scorer strategies; the saved artifact owns the resulting
inference behavior.

## Problem statement

Given context $x$ and a finite typed schema, choose one candidate for every field while avoiding autoregressive JSON generation. The system should:

- reuse document computation across fields;
- return only schema-valid values;
- work zero-shot with a compatible causal model;
- support local domain adaptation;
- expose scoring evidence without overstating confidence;
- persist enough metadata to load trained behavior unambiguously.

## Techniques considered

### Next-token finite choice

Map finite candidates to constrained answer tokens and inspect next-token logits after a field-specific query. This reuses the language model's instruction-following behavior and needs no learned classifier. Its constraints include tokenizer dependence and a bounded convenient answer alphabet.

This is the zero-shot path and the `"letter"` decoder training head, provided
the mapping is internal, deterministic, and uses single-token letters.

### Bilinear metric strategies

Project the query and candidate representations through a learned normalized
bilinear metric. This removes dependence on a small answer alphabet without
relying on raw cosine geometry. The head is trained with the decoder and stored
in its artifact.

The `"bilinear"` strategy uses context-free candidate embeddings. The
`"contextual"` strategy appends a query branch and one candidate branch per
option independently to the same document prefill. Candidate branches can use
the document and field instruction but cannot attend to sibling candidates;
their final hidden states meet only in the bilinear head. This adds branch work
while preserving a shared prefill and keeping execution policy artifact-owned.

### Architecture-specific learned heads

Both causal bilinear strategies can score representations for a changing finite
schema. Encoders instead use one candidate-marker head over packed candidate
blocks. All learned heads are packaged inside self-describing artifacts.

## Why there is no inference-time mode

An inference-time scorer switch creates several bad states:

- an artifact loaded with a scorer different from the one it trained;
- field-level and model-level precedence rules;
- schema metadata that changes numerical execution rather than semantics;
- results that cannot be reproduced from the artifact path alone.

The selected contract keeps the choice at causal training time:

```python
raw = DecisionModel("Qwen/Qwen2.5-1.5B-Instruct")
letter = raw.fit(examples, schema=Schema)
bilinear = raw.fit(examples, schema=Schema, config=FitConfig(decoder_head="bilinear"))
contextual = raw.fit(
    examples,
    schema=Schema,
    config=FitConfig(decoder_head="contextual"),
)
deployed = DecisionModel(saved_artifact_path)
```

The artifact then determines the inference implementation.

## One-schema training

Each fit names exactly one `DecisionSchema`. Dictionary labels may omit fields;
Jeeva masks those lanes so only provided labels contribute to loss. Requiring
one schema keeps validation, calibration, and artifact identity unambiguous.

## Confidence terminology

Candidate distributions are useful model scores. Their top-two margin is a useful ranking statistic. Neither deserves the name correctness probability without held-out evidence.

Jeeva nevertheless needs boolean certainty in zero-shot operation because ambiguity gates and arbiters consume it. Every field compares its margin with an active threshold resolved in this order:

1. manual primitive `tau`;
2. a held-out fitted threshold;
3. the engine default.

`is_certain` is always boolean. `threshold_source` records the selected source,
while `calibration` separately records held-out evidence coverage.

## Selected public design

```python
model = DecisionModel(source, device="auto", dtype="auto")
arbiter = SequenceArbiter()
result = model.decide(context, Schema, arbiter=arbiter)
results = model.decide_many(contexts, Schema, arbiter=arbiter)
trained = model.fit(examples, schema=Schema, validation=held_out, config=FitConfig(...))
trained.save(path)
```

The design keeps semantic choices public and numerical execution choices artifact-owned. It also makes the complete deployment unit a single loadable directory rather than a base model plus undocumented flags and sidecar objects.
