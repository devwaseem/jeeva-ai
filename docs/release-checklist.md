# Release Checklist

Use this checklist for every Jeeva release. Production publication is permitted only from a published GitHub release whose `vX.Y.Z` tag matches `project.version`.

## Repository and policy

- [ ] The release commit contains only reviewed product, test, documentation, and release-engineering changes.
- [ ] `pyproject.toml`, `src/jeeva/_version.py`, `CHANGELOG.md`, and the release tag use the same version.
- [ ] The changelog has a release date and no placeholder sections.
- [ ] `SECURITY.md`, license metadata, classifiers, project URLs, and `py.typed` are present.
- [ ] The compatibility matrix identifies exact tested contracts without claiming exhaustive model support.
- [ ] GitHub `testpypi` and `pypi` environments exist; `pypi` requires reviewer approval.
- [ ] TestPyPI and PyPI trusted publishers are configured for `.github/workflows/publish.yml` and their matching environments.

## Deterministic verification

No release command may download, execute, or fine-tune a pretrained checkpoint.

- [ ] `uv lock --check`
- [ ] `uv run ruff format --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run basedpyright src/`
- [ ] `uv run pytest -m "not model_integration and not training_integration"`
- [ ] An explicit integration-marker invocation reports skips rather than loading a model.
- [ ] The repository security scan has no unresolved release-blocking finding.

## Immutable distributions

- [ ] Remove old `dist/` output, then run `uv build` exactly once.
- [ ] Run `uvx --from twine==6.1.0 twine check dist/*`.
- [ ] Install the exact wheel into a clean base environment and verify `import jeeva`, installed metadata version equality, and `jeeva --help`.
- [ ] Install the same wheel with `training,serve,cloud,multimodal` extras into a second clean environment and verify those imports without loading a model.
- [ ] Inspect the wheel contents for credentials, local paths, checkpoints, caches, notebooks, and unintended data.

## Promotion

- [ ] Run the publish workflow manually; confirm it targets TestPyPI only.
- [ ] Install the TestPyPI candidate in a clean environment and repeat the no-model import/CLI smoke check.
- [ ] Create and review release notes from `CHANGELOG.md`.
- [ ] Create tag `vX.Y.Z` from the verified commit and publish the GitHub release.
- [ ] Approve the protected `pypi` environment only after the workflow confirms the tag/version match and repeats distribution verification.
- [ ] Verify the PyPI project page, hashes, install command, package version, and documentation links.
- [ ] Retain the tested distributions and previous release artifact for rollback and provenance.
