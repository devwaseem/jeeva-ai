# Jeeva Technical Architecture

## Abstract

Jeeva executes finite structured decisions directly over model states rather than
generating a JSON response. It supports decoder-only causal language models and
bidirectional encoders behind one typed public boundary. Causal runtimes prefill a
document once and reuse its KV cache across decision lanes. Encoder runtimes insert
one marker per field candidate and score all markers in one packed forward pass.
The public interface is a `DecisionModel` plus semantic `DecisionSchema` declarations;
architecture-specific scoring and training heads remain artifact-owned internals.

## Public boundary

```python
arbiter = SequenceArbiter()
result = model.decide(context, Schema, arbiter=arbiter)
trained = model.fit(examples, schema=Schema, validation=held_out, config=FitConfig(...))
trained.save(path)
loaded = DecisionModel(path, device="cuda")
```

The boundary deliberately exposes:

- model source, runtime architecture for raw models, device, and dtype at construction;
- `DecisionSchema` classes composed from `Noul`, `Choice`, and `Score`;
- an optional arbiter at inference;
- dictionary labels, one schema, and one `FitConfig` per fit;
- dictionary values, field evidence, calibration state, and telemetry in results;
- self-describing artifact directories for persistence.

It deliberately does not expose heads, primitive precedence, or fitted scorer
selection. A raw model defaults to causal zero-shot execution; encoder execution must
be requested explicitly and fitted before inference. A trained artifact declares its
architecture, scorer, weight kind, and encoder token budget in its manifest.

## Schema compilation

A public schema is a `DecisionSchema` subclass whose class attributes are immutable primitives:

- `Noul("instructions", {True: "meaning", False: "meaning"})` produces `True`
  or `False`.
- `Choice("instructions", {"value_a": "meaning", "value_b": "meaning"})`
  produces one declared stable string value.
- `Score("instructions", {0: "low", 1: "high"})` produces the expected numeric
  value over ordered semantic anchors.
- `examples` attach representative contexts to result values.
- optional `tau` expresses the application's escalation threshold.

Compilation preserves declaration order and produces keyed runtime lanes plus a
canonical descriptor. The descriptor includes field order, primitive kind,
instructions, stable values, semantic descriptions, examples, and Score
tolerance, then is serialized deterministically and hashed with SHA-256. Manual
`tau` is excluded, so operators may adjust escalation without changing semantic
compatibility.

## Architecture-specific context sharing

For context tokens $x_{1:n}$, the base transformer computes a KV cache once:

$$
K^{(\ell)}, V^{(\ell)} = \mathrm{Prefill}^{(\ell)}(x_{1:n})
$$

Each schema field creates a short decision query. Lanes share or replicate the prefill cache according to the internal cache layout, then execute as a batch. For $L$ lanes, the expensive document prefill remains independent of $L$; lane dispatch scales with short query work and candidate scoring rather than full-document repetition.

The causal runtime owns cache layout internally and favors contiguous,
kernel-friendly tensors on supported backends.

For an encoder, every document and schema field forms one tokenizer pair. The first
sequence contains the field instructions and document; the second packs every candidate
behind its own mask marker. Pair tokenization uses `truncation="only_first"`, preserving
the complete candidate set. Candidate tokens attend only to the shared prefix and their
own candidate block, and every candidate block restarts at the same position origin.
The marker representations pass through one shared nonlinear candidate-marker scorer.
Training permutes fields and candidates, then maps logits back to stable semantic values.

## Scoring

A raw causal model uses a zero-shot finite-candidate readout with
candidate-position balancing enabled by default. Applications that have
measured acceptable order stability and prioritize minimum dispatch latency or
memory can select the single declared order with
`DecisionModel(..., candidate_position_balancing=False)`.

The balanced scorer evaluates cyclic display orders so every semantic candidate
occupies every letter position once. Positional logits are remapped to semantic
candidates and averaged before normalization; this cancels fixed letter and
absolute-position preferences while preserving one shared document prefill.
Telemetry reports the physical branch count plus per-field trial count,
order-sensitivity spread, and worst-order winner margin. The latter is used as
the conservative certainty score.

Causal fitting selects one of three scorers: the model's output head by default,
a normalized bilinear head over context-free candidate embeddings, or the same
head over document-conditioned candidate branches. Encoder fitting always uses
a shared candidate-marker head over packed document–instruction–candidate
representations.
Every path produces a normalized distribution and observable statistics for each
field:

- selected or expected value;
- winning candidate probability;
- top-two margin;
- primitive-specific certainty score;
- entropy and normalized entropy;
- winning level and distribution standard deviation for `Score`.

These statistics are model evidence, not a universal probability of correctness.

## Arbitration

After the initial readout, Jeeva sends only uncertain fields to the optional
arbiter. Certain fields retain their gatekeeper result. Local sequence arbiters
can reuse a causal primary model's prefilled context. Encoder primaries require a
dedicated causal model for sequence/thinking arbitration. Dedicated-model and
OpenAI-compatible endpoint arbiters perform their own resolution work.

The final `FieldDecision` preserves `resolved_by` and optional `reasoning`, while telemetry records invoked fields and arbiter latency. If a replacement remains below its active field threshold, aggregate ambiguity remains true so the application can escalate to human review.

## Training

`DecisionModel.fit` compiles the one schema supplied for the run. Each label is
a dictionary; omitted fields are masked so only provided labels contribute to
loss. Unknown fields and invalid values fail validation. Continuous Score
targets interpolate between adjacent anchors.

Causal fitting defaults to LoRA adaptation through letter readout from the model
output head. The
`"bilinear"` scorer represents the field query with one document-conditioned
branch and compares it with context-free candidate embeddings. The
`"contextual"` scorer uses the same normalized bilinear head, but appends one
candidate token per option to the prompt and backpropagates through the document
prefill plus candidate-token hidden states. Each contextual candidate therefore
receives its own contextual embedding without paying for a second document prefill.
Encoder fitting supports head-only, LoRA, and full-backbone modes, with
`"auto"` selecting the architecture-compatible default.

The training path includes:

1. **Schema compilation** — Produces fixed candidate spaces and masks.
2. **Train/validation split** — Deterministic when validation data is not supplied.
3. **Adapter or head optimization** — Encoder models train either the head or
   LoRA adapters. Causal models train LoRA adapters, with optional QLoRA loading
   of the frozen backbone in 4-bit NF4 on MPS or CUDA. QLoRA uses double
   quantization and keeps adapters plus the decision head in floating point.
4. **Class-imbalance correction** — inverse-frequency class weights (clipped to
   a maximum ratio of 10) prevent rare labels from being ignored.
5. **Calibration fitting** — Fit per-field temperatures on held-out logits.

All learned paths optimize soft-target cross-entropy plus a weighted Brier term.
`Score` adds a weighted ranked-probability loss over the cumulative ordinal
distribution. Encoder batches pack all schema candidates into one pass.
Candidate order is part of a trained encoder's schema fingerprint and remains
stable through inference. Trained causal artifacts using either metric strategy
may accept a different runtime schema because candidate representations are
computed at runtime; letter artifacts retain the 26 single-token candidate
limit.
Calibration remains keyed to the schema used to produce it.

Adapter artifacts store PEFT weights plus portable metadata. QLoRA emits the
same ordinary adapter format and reloads it over an unquantized base model for
inference and optional merging; bitsandbytes is required only while fitting or
continuing QLoRA. `DecisionModel.save(..., merge=True)` can therefore produce a
standalone Hugging Face directory without attempting to merge a live 4-bit
model.

## Calibration

For field $f$, let $c_f$ be its primitive-specific certainty score. `Choice`
and `Noul` use the top-two probability margin. `Score` uses normalized inverse
distribution spread:

$$
c_f^{\mathrm{score}} =
1 - \operatorname{clamp}\left(
\frac{2\sigma_f}{v_{\max}-v_{\min}}, 0, 1
\right)
$$

Held-out predictions pair certainty with correctness. Score correctness is
$|\hat y-y| \le \mathrm{tolerance}$, so Score calibration requires explicit
tolerance. Jeeva selects a calibrated threshold $\tau_f$ under the requested
precision objective when sufficient labeled validation support exists.

Encoder validation additionally selects a per-field temperature that minimizes
held-out Brier score before certainty thresholds are fitted.

At inference:

$$
\mathrm{certain}_f = [c_f \ge \tau_f^*],
\qquad
\tau_f^* =
\begin{cases}
\tau_f^{\mathrm{manual}} & \text{when the primitive declares } \tau \\
\tau_f^{\mathrm{calibrated}} & \text{when schema-specific calibration exists} \\
\tau_0 & \text{otherwise}
\end{cases}
$$

Field and aggregate certainty are always boolean, allowing arbiters to run zero-shot. Each field reports both the active numeric `tau` and its `"manual"`, `"calibrated"`, or `"default"` source. Calibration status separately records held-out evidence coverage.

## Artifact format

A zero-shot causal artifact may contain only a Jeeva manifest referencing its base
model. Trained artifacts contain either full model weights or a PEFT adapter;
`save(..., merge=True)` converts an adapter artifact to standalone base-model
weights. Every trained form includes:

- tokenizer and processor assets when applicable;
- a versioned `jeeva.json` manifest with architecture, scorer, and weight kind;
- requested and resolved model/tokenizer IDs and revisions, local-only policy, and public fit configuration;
- known schema descriptors and training report;
- schema/field calibration evidence;
- the selected learned head: causal static/contextual bilinear or encoder candidate-marker, when applicable.

Adapter artifacts additionally contain PEFT configuration and adapter weights.
They are compact and retain exact adapter continuation. Full and merged artifacts
contain the complete backbone weight set and load without PEFT.

Merged export uses PEFT's checked temporary merge plus a base-only state dictionary. The live model is unmerged in a `finally` path after serialization, avoiding permanent mutation of the caller's model. Loading validates the format-specific files before constructing the internal engine. The user does not reconstruct training internals or choose a scorer.

Raw repository loads accept independent model and tokenizer revisions, with the tokenizer
defaulting to the model revision. Resolution metadata from Transformers is persisted in
the manifest. Artifact reload uses those resolved revisions and rejects caller overrides;
dependency versions remain diagnostic while the manifest format and schema protocol are
the hard compatibility gates.

## Runtime ownership

A `DecisionModel` owns loaded model state and therefore owns device and dtype. Schemas are immutable call contracts and can vary between calls. `fit` returns a new model owner so the receiver remains usable and mutation is explicit. `save` requires an absent destination, writes and validates a sibling staging directory, and exposes it with one atomic rename.

## Operational guidance

- Keep one long-lived model per process/device.
- Evaluate exact dictionary behavior on independent held-out data.
- Report accepted precision together with review rate.
- Use `is_ambiguous` for gating; inspect `threshold_source` and `calibration` before making precision claims.
- Version and promote whole artifact directories atomically.
- Run a real decision smoke check before shifting traffic.
