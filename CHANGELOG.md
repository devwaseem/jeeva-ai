# Changelog

All notable changes to Jeeva are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-27

### Added

- Canonical `DecisionModel` lifecycle for typed decisions, fitting, evaluation, artifact persistence, and serving.
- Causal letter, bilinear, and contextual readouts plus ModernBERT candidate-marker training.
- Empirical precision-targeted calibration with explicit insufficient-data and unattainable states.
- Optional local and cloud arbiters that preserve per-field resolution evidence and telemetry.
- JSON-schema CLI commands: `decide`, `fit`, `evaluate`, and `serve`.
- Authenticated single-model FastAPI serving with bounded requests and opt-in evidence.
- Self-describing format 11 artifacts with dependency provenance and requested/resolved model and tokenizer revisions.
- Separate training, QLoRA, serving, cloud, and multimodal dependency extras.

### Security

- Fail-closed artifact validation and atomic artifact saves.
- Required bearer authentication by default for the built-in server.
- Remote image URLs are rejected; local and in-memory images have count,
  per-image decoded-byte, and aggregate decoded-byte limits enforced before decode.
- Immutable-pinned CI and trusted-publishing workflows with protected PyPI promotion.
- Saved head dimensions, parameter budgets, backbone compatibility, and Safetensor
  shapes are validated before module allocation.
- Caller-varying prompts and option text are no longer retained in process-lifetime
  accelerator caches.
- Model and adapter loading is Safetensors-only; pickle-backed checkpoints are rejected.
- Schema field, candidate, prompt, option, and aggregate budgets prevent branch and
  position-balancing allocation amplification.

### Changed

- Learned decision-head weights use Safetensors rather than pickle-capable Torch serialization.
- Calibration terminology now describes empirical selective-classification evidence rather than unsupported conformal guarantees.
- The causal model-output scorer is now named `"letter"` consistently in configuration, CLI, telemetry, reports, and artifacts; format 10 artifacts are rejected after the clean vocabulary cutover.
- Automated tests use mocked or synthetic model components and never download or execute pretrained checkpoints.

### Fixed

- Encoder fitting now casts reused inference runtimes to the declared training precision, preventing non-finite MPS updates when `dtype="auto"` selected float16 for inference.
- The Snake example again excludes collision moves using exact game state while preserving the model's probability ranking across safe directions.

### Removed

- Legacy `in_context` readout mode, duplicate evaluator module, obsolete CLI paths, broken training example, stale benchmark scripts/results, and duplicate documentation guides.

[0.1.0]: https://github.com/waseem07799/jeeva/releases/tag/v0.1.0
