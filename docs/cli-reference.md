# CLI Reference

Jeeva exposes one command-line lifecycle over the public `DecisionModel` API:
`decide`, `fit`, `evaluate`, and `serve`.

All commands accept schemas only as data-only `.json` files. Executable Python schema
modules and inline JSON are intentionally unsupported by the CLI. A JSON schema maps field
names to `noul`, `choice`, or `score` definitions using the same `instructions`, `criteria`,
`tau`, and `tolerance` concepts as the Python API.

## Common model options

| Flag | Description |
| :--- | :--- |
| `-m`, `--model` | Required model ID or saved Jeeva artifact directory. |
| `--device` | Runtime device. Default: `auto`. |
| `--dtype` | Runtime dtype. Default: `auto`. |
| `--revision` | Model repository branch, tag, or immutable commit. |
| `--tokenizer-revision` | Tokenizer revision. Defaults to `--revision`. |
| `--local-files-only`, `--no-local-files-only` | Require cached/local files or allow remote resolution. |

For reproducible remote loads, prefer an immutable commit in `--revision`. Saved Jeeva
artifacts record both requested and resolved model/tokenizer revisions and reuse the
resolved revisions on reload; revision flags therefore apply to raw model IDs, not saved
artifact directories.
Commands that perform decisions also accept `--candidate-position-balancing` or
`--no-candidate-position-balancing` and the following arbiter options:

| Flag | Description |
| :--- | :--- |
| `--arbiter` | `none`, `sequence`, `thinking`, or `cloud`. Default: `none`. |
| `--arbiter-model` | Optional model ID for a local sequence or thinking arbiter. |
| `--cloud-model` | OpenAI-compatible cloud model ID. Default: `gpt-4o-mini`. |
| `--ambiguity-criterion` | `margin`, `entropy`, or `both` for local arbiters. |

## `jeeva decide`

Run one typed decision. Provide the context with `--context`, or omit it and pipe text to
standard input.

```bash
jeeva decide \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --schema schemas/ticket.json \
  --context "Payment failed on invoice 1029"

cat inquiry.txt | jeeva decide \
  --model artifacts/ticket-model \
  --schema schemas/ticket.json \
  --json
```

Human-readable output is the default. `--json` emits the complete `Decision`, including
per-field distributions, certainty evidence, calibration status, resolver, reasoning, and
telemetry.

## `jeeva fit`

Fit one schema and save the resulting self-describing artifact. Install the
training extra first:

```bash
pip install "jeeva-ai[training]"

jeeva fit \
  --model answerdotai/ModernBERT-base \
  --architecture encoder \
  --schema schemas/ticket.json \
  --data data/train.jsonl \
  --validation data/validation.jsonl \
  --output artifacts/ticket-model
```

For 4-bit causal QLoRA training on MPS or CUDA, install
`jeeva-ai[qlora]` and pass `--adaptation qlora`.

`--architecture` accepts `causal` or `encoder`; omit it when the source artifact already
records the architecture. `--merge` requests a merged standalone causal artifact.
`--json` emits the artifact path, architecture, and training report as JSON.

Every `FitConfig` control is exposed:

| Flag | Default |
| :--- | :--- |
| `--epochs` | `3` |
| `--progress-updates-per-epoch` | `10` |
| `--learning-rate` | architecture default |
| `--head-learning-rate` | architecture default |
| `--decoder-head` | `letter` |
| `--scheduler` | `cosine` |
| `--minimum-learning-rate-ratio` | `0.05` |
| `--weight-decay` | `0.01` |
| `--maximum-gradient-norm` | `1.0` |
| `--adaptation` | `auto` |
| `--lora-rank` | `8` |
| `--lora-alpha` | `2 × lora-rank` |
| `--lora-dropout` | `0.05` |
| `--batch-size` | `8` |
| `--max-length` | `512` |
| `--brier-weight` | `0.5` |
| `--ordinal-weight` | `0.25` |
| `--save-best` / `--no-save-best` | enabled |
| `--early-stopping-patience` | `2` |
| `--no-early-stopping` | disabled |
| `--validation-fraction` | `0.2` |
| `--seed` | random 32-bit value |
| `--target-precision` | `0.95` |
| `--minimum-calibration-examples` | `5` |

`--adaptation` accepts `auto`, `head`, `full`, `lora`, or `qlora`. QLoRA is
limited to causal text models on MPS or CUDA and uses a 4-bit NF4
bitsandbytes backbone. `auto` continues an existing QLoRA artifact with QLoRA;
otherwise it selects ordinary LoRA.

`--decoder-head` accepts `letter`, `bilinear`, or `contextual` for causal
training. `letter` uses single-token candidate logits from the model output
head; `bilinear` uses context-free candidate embeddings; `contextual` encodes
the query and every candidate in isolated branches over the shared document
prefill. Encoder training uses its candidate-marker scorer and rejects a
nondefault decoder-head selection.

## `jeeva evaluate`

Evaluate a model or artifact on labeled JSONL data through the same public decision path
used in production.

```bash
jeeva evaluate \
  --model artifacts/ticket-model \
  --schema schemas/ticket.json \
  --data data/test.jsonl \
  --target-precision 0.98 \
  --output-json reports/ticket-scorecard.json
```

The terminal scorecard is human-readable by default. `--json` writes it to standard output
as JSON; `--output-json` also stores an indented JSON scorecard at the requested path.
Arbiter options have the same semantics as `DecisionModel.decide(..., arbiter=...)`.

## `jeeva serve`

Install the serving extra, set a bearer token, and start one model/artifact with exactly one
startup-registered schema:

```bash
pip install "jeeva-ai[serve]"
export JEEVA_API_KEY="replace-with-a-secret"

jeeva serve \
  --model artifacts/ticket-model \
  --schema schemas/ticket.json \
  --host 127.0.0.1 \
  --port 8000
```

Authentication is fail-closed. By default the server refuses to start unless
`JEEVA_API_KEY` is set. `--api-key-env NAME` selects another environment variable. The
literal token is never accepted as a command-line argument. For deliberate local-only
experiments, `--allow-unauthenticated` disables authentication explicitly.

The service accepts:

- `POST /v1/decide` with `{"context": "..."}`
- `POST /v1/batch` with `{"contexts": ["...", "..."]}`
- `GET /health`

Decision endpoints require `Authorization: Bearer <token>`. Responses include typed values,
certainty, calibration, thresholds, and resolver metadata by default. Add the query
parameter `include_evidence=true` to include distributions, scoring evidence, reasoning,
and telemetry. The health response contains status, Jeeva version, and uptime; it does not
disclose model IDs or hardware.

The server rejects per-request schemas, per-request thresholds, oversized contexts and
batches, requests beyond inference capacity, and request bodies beyond the configured
limit. OpenAPI is disabled by default.

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8000` | Bind port. |
| `--api-key-env` | `JEEVA_API_KEY` | Environment variable containing the bearer token. |
| `--allow-unauthenticated` | disabled | Explicit local-only authentication bypass. |
| `--max-batch-size` | `32` | Maximum contexts per batch. |
| `--max-context-chars` | `100000` | Maximum characters in one context. |
| `--max-pending-requests` | `64` | Maximum concurrent inference slots. |
| `--max-body-bytes` | `1000000` | Maximum declared request-body size. |
| `--enable-docs` | disabled | Expose `/docs` and `/openapi.json`. |

Example request:

```bash
curl -sS http://127.0.0.1:8000/v1/decide \
  -H "Authorization: Bearer $JEEVA_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{"context":"Payment failed on invoice 1029"}'
```
