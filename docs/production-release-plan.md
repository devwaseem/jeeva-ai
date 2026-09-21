# Production Release Plan

Status: **NO-GO** until the blocking correctness and release-gate items are resolved.

## Working agreement

- Resolve the plan one decision and implementation slice at a time.
- Record decisions in this document as they are made.
- Do not create intermediate commits. Create the production release commit only after all agreed work passes the complete release verification.
- Stabilize the existing `DecisionModel` architecture; do not perform a pre-release monolith rewrite.

## Keep

- `DecisionModel` with `DecisionSchema`, `Noul`, `Choice`, and `Score` as the public boundary.
- Off-the-shelf causal zero-shot letter readout with candidate-position balancing and optional domain fitting.
- Fitted causal dense scoring.
- Fitted ModernBERT candidate-marker encoding, including the low-latency Snake workload.
- Arbiter interfaces, selective invocation, evidence, and telemetry after correcting `CloudArbiter` semantics.
- Self-describing versioned artifacts and schema fingerprints after adding provenance and compatibility controls.
- Separate fast, model-integration, and training test tiers with shared model fixtures.
- Apache-2.0 licensing, `py.typed`, OIDC publishing, request-size and SSRF protections, and fail-closed manifest validation.

## Drop or withhold from 0.1 unless repaired

- The built-in server's current “production-grade” positioning and potentially the entire `serve` command.
- Legacy CLI training, evaluation, and serving commands unless rewritten around `DecisionModel` artifacts.
- `train_example.py`, which calls the nonexistent `jeeva.train` API.
- Legacy benchmark scripts and untrustworthy published results that use removed APIs or lack reproducibility.
- The dead duplicate `src/jeeva/eval.py`; retain the `jeeva/eval/` package.
- Unsupported “conformal prediction” terminology unless a real conformal guarantee is implemented.
- Duplicated canonical guides: `fine-tuning.md` versus `training-and-finetuning.md`, and `typing-and-schemas.md` versus `schemas-and-types.md`.

## Blocking correctness work

### Calibration thresholds

`QuantileNotchCalibrator.fit` evaluates individually sorted rows instead of complete equal-margin groups. It can persist a threshold whose actual accepted-set precision is below `target_precision`. It also returns a misleading threshold when no threshold achieves the requested precision.

Required outcome:

- Evaluate complete equal-margin groups.
- Represent an unattainable target explicitly.
- Never persist a below-target threshold as calibrated.
- Define zero-coverage behavior.
- Test ties, unattainable targets, all-correct, all-incorrect, and boundary margins.
- Use empirical selective-classification terminology unless a formal conformal procedure is implemented.

### Cloud arbiter

`CloudArbiter` assigns provider confidence to `selected` and divides the remainder among alternatives. Below `1 / candidate_count`, another candidate becomes the argmax and Jeeva returns a value different from the provider's selection.

Required outcome: define whether provider selection or probabilities are authoritative, reject incompatible responses where appropriate, and never silently contradict the selected value.

### Integration gate

The model/training tier currently has 31 passing and 3 failing tests:

- `tests/test_dense_embedding.py:62-73`
- `tests/test_dx_production.py:118-126`
- `tests/test_in_context.py:43-73`

Resolve stale or brittle semantic expectations without weakening genuine contracts. Remove permutation-invariance claims if the runtime does not guarantee them.

## Public surface work

- Decide whether 0.1 is a Python-library release or also supports CLI training/evaluation and a built-in server.
- If retained, move every CLI lifecycle onto `DecisionModel`, `fit`, `save`, and self-describing artifacts.
- Prefer an application-owned FastAPI integration over a generic server that accepts dynamic schemas and exposes internal telemetry.
- Add end-to-end tests for every retained command.
- Remove the broken root training example.
- Replace legacy benchmarks with reproducible public-API benchmarks.
- Add a reproducible Snake encoder scorecard with model revision, data digest, seed, hardware, command, quality metrics, latency, and memory.

## Artifact and API hardening

- Add public causal `DecisionModel` integration coverage for zero-shot decisions, zero-shot save/reload, causal fitting, and trained save/reload.
- Record Jeeva, Python, Transformers, Torch, and PEFT versions as applicable.
- Record requested and resolved model/tokenizer IDs and revisions.
- Add explicit model revision, tokenizer revision, and `local_files_only` controls.
- Define and test the current artifact compatibility policy with a golden fixture.
- Save artifacts into a sibling temporary directory, validate them, then atomically rename.
- Narrow the broad `AutoProcessor` exception fallback so authentication, corruption, and remote failures remain visible.
- Add direct `numpy` dependency, review `pydantic` placement, and add a cloud extra or remove the implicit OpenAI SDK path.

## Documentation work

- Make encoder adaptation defaults agree with code: `adaptation="auto"` currently resolves to LoRA.
- Replace unsupported conformal claims with the chosen calibration terminology.
- Publish an evidence-based compatibility matrix instead of claiming generic decoder support beyond continuously tested models.
- Consolidate duplicate guides.
- Use stable absolute documentation links in PyPI-facing content.
- Add `SECURITY.md`, `CHANGELOG.md`, artifact compatibility policy, and a release checklist.
- Keep research, blog, and whitepaper material separate from canonical task-oriented product documentation.

## Release engineering work

- Make the working tree coherent before tagging; crucial public implementation files are currently untracked.
- Pin GitHub actions and `setup-uv` instead of using `latest`.
- Lint and format-check source, tests, and examples.
- Split production test dependencies from research/notebook dependencies.
- Build the wheel once and test that exact immutable artifact in a clean environment.
- Default manual publishing to TestPyPI.
- Protect production PyPI publishing with a GitHub environment and OIDC.
- Run tests, `twine check`, isolated install, version/import checks, and a representative public decision before promotion.
- Correct package and repository URLs.

## Audit evidence

- Ruff lint: passed.
- Ruff format check: failed for five files; only one is Python code.
- BasedPyright on `src/`: 0 errors and 0 warnings.
- Fast suite: 200 passed, 34 deselected, with two dependency deprecation warnings.
- Model/training integration: 31 passed, 3 failed.
- Wheel and sdist build: passed.
- `twine check`: passed.
- Isolated wheel import, version, and CLI help: passed.
- Lockfile check: passed.
- Base and optional dependency vulnerability audit: no known vulnerabilities.
- Release tree: 63 unstaged changes and 26 untracked files at audit time.

## Decision log

- **Release scope:** ship a full production product. The Python API, CLI lifecycle, and built-in server must all meet production release gates; none may be labeled preview or silently excluded.
- **Canonical lifecycle:** rewrite the CLI and built-in server around `DecisionModel`. CLI training, evaluation, inference, and saving must use the same public artifacts. The server loads a registered schema and artifact at startup rather than accepting arbitrary per-request schemas.
- **Calibration guarantee:** use empirical precision-targeted threshold calibration. Report held-out evidence, achieved precision, and coverage; make no conformal or future-data guarantee.
- **Cloud arbiter evidence:** require a complete probability map keyed by every candidate label. Values must be finite, nonnegative, sum to one, and have a unique maximum; Jeeva derives the selected value rather than trusting a separate `selected` field.
- **No live-model automation:** local and CI test execution must never download, execute,
  or fine-tune pretrained checkpoints. Public lifecycle contracts use mocked runtimes and
  trainers; numerical tests use local synthetic components. Model quality is outside the
  package release gate.
- **Publication trigger:** manual dispatch publishes only to TestPyPI. Production PyPI
  publication requires a published `vX.Y.Z` GitHub release whose tag matches the package
  version and approval through the protected `pypi` environment.
- **Compatibility publication:** publish a capability-contract matrix, not a tested
  checkpoint allowlist. Qwen 2.5, Qwen 3, Gemma, Llama, and ModernBERT are intended
  candidates only; applications must pin and qualify exact revisions because project
  policy forbids live checkpoint execution.
- **Readout surface:** remove legacy `in_context` mode. The supported causal readouts are zero-shot letter scoring and fitted dense scoring; encoder artifacts use candidate-marker scoring.
- **CLI vocabulary:** use `decide`, `fit`, `evaluate`, and `serve` to match the `DecisionModel` API; do not retain pre-1.0 command aliases.
- **CLI schemas:** accept data-only JSON schema files exclusively. Inline JSON and executable Python schema modules are not part of the production CLI contract.
- **CLI output:** human-readable output is the default, with an explicit JSON mode for automation.
- **Fit flags:** expose every `FitConfig` setting as a CLI flag rather than introducing a separate fit configuration file.
- **Performance publication:** publish no general performance report or headline numbers for 0.1. Keep Snake as a runnable, hardware-local comparison: an off-the-shelf causal model plays zero-shot, while a separately fitted ModernBERT encoder demonstrates the trained low-latency path. The example may print score, throughput, and latency percentiles, but documentation must not present those local measurements as universal claims.
- **Server authentication:** require bearer authentication by default. The token comes
  from an environment variable rather than a command-line value; unauthenticated operation
  requires an explicit local-only override.
- **Server disclosure:** return values, certainty, calibration, thresholds, and resolver
  metadata by default. Authenticated clients may request full distributions, scoring
  evidence, reasoning, and telemetry explicitly. Health responses do not disclose model IDs
  or hardware.
- **Server tenancy:** one server process owns exactly one startup-registered schema and one
  model/artifact. Per-request schemas and threshold overrides are not supported.
- **Root training example:** delete the obsolete `train_example.py` rather than maintaining
  a duplicate top-level example. Canonical fitting guidance remains in the CLI reference,
  training guide, and maintained example directories.
- **Snake comparison protocol:** compare the zero-shot causal model and fitted encoder on
  the exact same held-out JSONL corpus. Report the corpus digest, directional accuracy,
  throughput, and p50/p95 warm wall-clock latency; do not commit measured result files.
- **Fitted causal Snake path:** retain and repair the optional causal trainer even though it
  is not part of the two-model comparison. It must train the same one-field
  `DirectionDecision` consumed by the game.
- **Evaluator namespace:** delete `src/jeeva/eval.py`. The installed `jeeva.eval` namespace
  is exclusively the evaluator package; the colliding ticket demo is not relocated.
- **Canonical guides:** retain the comprehensive `training-and-finetuning.md` and
  `schemas-and-types.md` guides. Delete the short duplicate paths rather than shipping
  redirect stubs before 0.1.
- **Causal lifecycle tests:** use mocked runtime and trainer dependencies to exercise the
  public zero-shot and fitted causal lifecycles without loading live model weights. The
  contract gate covers `decide`, `fit`, `save`, manifest reload, and equivalent public
  decisions before and after reload.
- **Artifact provenance compatibility:** record exact creation-time Jeeva, Python, Torch,
  Transformers, and applicable PEFT versions. Treat these versions as diagnostic evidence;
  hard compatibility gates are the manifest format and schema protocol, not dependency
  version equality.
- **Model source pinning:** expose model revision, tokenizer revision, and local-only
  loading on `DecisionModel` and every CLI lifecycle. The tokenizer revision inherits the
  model revision when omitted. Artifacts persist requested and resolved IDs/revisions and
  reload from the resolved revisions; callers may override offline policy but not artifact
  revision identity.
- **Atomic save contract:** require the destination path to be absent. Write and validate
  a sibling staging directory, then rename it into place. Do not retain the weaker
  absent-or-empty contract or add in-place artifact replacement to the 0.1 API.
- **Processor detection:** load `AutoProcessor` only when the loaded model configuration
  explicitly declares vision/image components. Once declared, missing dependencies,
  authentication failures, corrupt assets, and remote failures are fatal; none may
  silently downgrade the model to text-only operation.
- **Dependency boundaries:** keep the runtime core lean but complete: NumPy is a direct
  core dependency; Pydantic belongs to the serving extra; the OpenAI SDK belongs to a
  dedicated cloud extra. Training, QLoRA, and multimodal dependencies remain isolated.

## Implementation log

- **Empirical threshold calibration complete:** thresholds now evaluate complete
  equal-margin groups, return an explicit `"unattainable"` result when the requested
  precision cannot be reached, and never persist a learned threshold in that case.
  Artifacts record target precision, samples, best observed precision/coverage, and
  `"insufficient_data"` or `"unattainable"` status. Runtime results remain
  `"unvalidated"` and fall back to manual or default thresholds.
- Renamed the internal calibration module and evaluator output away from unsupported
  conformal terminology. Evaluator precision targets no longer treat zero coverage as
  successful.
- **CloudArbiter correction complete:** provider prompts and validation now use the
  full-distribution contract. Missing, extra, nonnumeric, nonfinite, negative,
  non-normalized, non-string-keyed, and tied responses fail closed. Choice and Score
  resolution tests verify that returned values and evidence come from the same
  distribution.
- **Integration gates repaired:** dense cache coverage now checks equivalence between
  contiguous and memory-efficient cache paths instead of one model-quality answer;
  public semantic-value assertions use declared schema keys. Legacy `in_context`
  execution and its unsupported permutation-invariance claim were removed.
- Verification: the complete model/training integration tier passed with 32 tests and
  216 deselections.
- Verification: focused calibration and arbiter tests passed; the fast suite passed
  with 216 tests and 35 deselections; Ruff and BasedPyright passed for the changed code.
- **Formatting gate repaired:** Ruff formatted the full repository, including fenced
  Python examples; `ruff format --check .` and `ruff check .` now pass.
- **Canonical CLI complete:** `decide`, `fit`, `evaluate`, and `serve` now use
  `DecisionModel`; CLI schemas are JSON files only, output is human-readable by default,
  and every `FitConfig` setting has a corresponding flag.
- **Built-in server hardened:** the server uses public `DecisionModel` decisions, a single
  startup schema, required bearer authentication, constant-time token comparison, bounded
  bodies/contexts/batches/inference capacity, minimal default responses, opt-in evidence,
  nondisclosing health output, hidden internal errors, and disabled-by-default OpenAPI.
- Verification: canonical CLI, server-hardening, and non-model production DX tests passed
  with 23 tests and four model-integration deselections; focused BasedPyright passed.
- **Broken root example removed:** deleted `train_example.py`; no remaining code or
  documentation references call the nonexistent `jeeva.train` API.
- **Legacy benchmarks replaced:** removed internal-engine AG News, Banking77, and support
  routing scripts, their checked-in result files, and their benchmark-only test. Snake now
  provides one public-API comparison over identical held-out cases with dataset hashing and
  local-only score, throughput, and latency output.
- **Snake lifecycle repaired:** generated records, both trainers, comparison, and gameplay
  now share `DirectionDecision`; collision safety remains exact game logic. The causal
  trainer remains available as an optional fitted path.
- Verification: 11 Snake example tests passed; focused Ruff and BasedPyright passed; the
  comparison command's real CLI help path executed successfully.
- **Evaluator collision removed:** deleted the domain-specific `src/jeeva/eval.py`; all
  evaluator imports continue to resolve through the retained `src/jeeva/eval/` package.
- **Duplicate guides removed:** deleted `fine-tuning.md` and
  `typing-and-schemas.md`. The documentation index already points to the retained
  comprehensive guides, and no live documentation links target the deleted paths.
- **Public causal lifecycle covered:** zero-shot and fitted causal tests now traverse the
  public API through filesystem artifacts and reload them before comparing typed decisions.
  Runtime and trainer dependencies are mocked as requested; no live model is loaded.
- Verification: the complete model API contract suite passed with 39 tests.
- **Artifact provenance recorded:** format 11 manifests require a timezone-aware creation
  timestamp, exact creation-environment versions, and requested/resolved model and
  tokenizer source identity. Adapter artifacts also record PEFT. Loaders validate the
  provenance shape but intentionally do not reject version differences.
- **Compatibility fixture added:** a golden format 11 zero-shot manifest with historical
  dependency versions loads under the current code, while format 10 and current-format
  manifests missing provenance fail closed.
- **Immutable model sources complete:** Python and CLI model loads accept model revision,
  tokenizer revision, and local-only controls. The tokenizer inherits the model revision;
  artifacts record requested and resolved IDs/commits and reload using resolved commits.
- Verification: revision lifecycle, artifact override rejection, compatibility fixture,
  and CLI forwarding tests passed; focused BasedPyright passed for all changed source
  modules.
- **Atomic artifact saves complete:** `DecisionModel.save` now rejects every existing
  destination, writes into a sibling staging directory, validates the manifest and
  format-specific files, and renames only a complete artifact into place. Failed saves
  remove staging output without exposing the destination.
- **Processor fallback hardened:** text-only model configurations skip processor loading.
  Vision-declaring configurations require a valid image processor and propagate all load
  failures. The trainer uses the same configuration-driven policy.
- Verification: 47 engine, trainer-component, and multimodal tests passed; focused
  BasedPyright passed.
- **Dependency metadata corrected:** NumPy is declared directly, Pydantic moved to
  `serve`, and `cloud` now installs the OpenAI SDK. `CloudArbiter` names the required
  extra when SDK construction is requested without it.
- **CI policy corrected:** real-model and real-training jobs were removed. A collection
  guard skips every legacy integration marker even when explicitly selected, enforcing
  the no-live-model rule locally and in CI.
- **Immutable wheel gate complete:** CI builds the wheel and sdist once, runs Twine
  metadata validation, installs that exact wheel into isolated base and production-extra
  environments, verifies imports/version/CLI help without loading models, and uploads the
  tested distributions as one artifact.
- Verification: the local wheel and sdist passed Twine; the exact wheel passed isolated
  base and `training,serve,cloud,multimodal` installation and import smoke checks.
- **Publishing hardened:** every action and uv version is immutable-pinned. The publish
  workflow builds, validates, and isolated-smoke-tests one distribution set, then sends
  that same artifact to TestPyPI for manual dispatch or to PyPI only for a matching
  published release through OIDC-scoped environments.
- **Release documentation complete:** added a private-reporting security policy, a dated
  0.1.0 changelog, and an executable release checklist covering deterministic verification,
  immutable distributions, trusted publishing, and rollback. README links are absolute for
  correct PyPI rendering.
- **Compatibility contract published:** the matrix distinguishes automated mocked and
  synthetic evidence from intended model-family candidates, documents exact runtime
  prerequisites, and gives an application-owned checkpoint qualification procedure.
- **Safe head serialization:** causal metric-head artifacts now write and require
  Safetensors weights; the pickle-capable `torch.save`/`torch.load` path was removed.
- **Security scans remediated:** complete scoped scans reviewed all source, workflow,
  and package-metadata files. The first pass identified four findings: remote image
  fetching was removed to eliminate DNS-rebinding SSRF; image count and decoded-byte
  budgets now run before decode; saved head dimensions and exact Safetensor shapes are
  validated before allocation; and caller-varying accelerator caches were removed. A
  follow-up pass found two further issues: every model and adapter loader now requires
  Safetensors, and schema plus letter-position balancing enforce explicit field,
  candidate, text, and aggregate budgets before branch allocation. Focused regressions
  cover every remediation.
- **Rejected security candidates documented:** dataset local paths and adapter base-model
  pointers remain explicit local-operator and application artifact-policy boundaries.
  Remote dataset images flow through the now-disabled URL path, and no dynamic repository
  code-loading option is enabled.

## Planned order

1. Set the release scope and public support promise.
2. Decide the CLI/server surface, calibration terminology, compatibility promise, and benchmark evidence bar.
3. Fix calibration and `CloudArbiter` correctness.
4. Resolve integration failures and formatting.
5. Align or remove legacy public surfaces.
6. Harden public API tests, artifacts, provenance, dependencies, and documentation.
7. Make CI and publishing exercise the exact built wheel.
8. Complete and remediate a scoped security scan.
9. Run the complete release verification.
10. Create the single final production release commit.
