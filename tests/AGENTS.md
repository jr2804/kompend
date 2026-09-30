# DOX — tests

<!-- on entry: loaded when agent enters tests -->
<!-- parent: ../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Test suites for the KNOX project. Governs how tests are structured, where they
go, and what patterns are forbidden.

## Ownership

- `tests/unit/` — unit tests (fast, isolated, no real LLM calls)
- `tests/integration/` — integration tests (real pipeline runs, LLM calls optional)
- `tests/benchmarks/` — performance benchmarks
- `tests/data/` — test fixtures and sample documents
- `tests/fixtures/` — reusable `pytest` fixtures
- `tests/helpers/` — test utility functions

## Local Contracts

### File discovery — do NOT create new test files blindly

- **Find the existing test file first.** Every module under `src/knox/` should
  have a corresponding `tests/unit/test_<module>.py`. Never create a new test
  file before verifying the right one doesn't already exist.
- If you are writing tests for `src/knox/foo.py`, the test file is
  `tests/unit/test_foo.py` (not `tests/unit/test_foo_fixes.py`,
  `test_foo_v2.py`, or any variant).
- The only exception is when you are creating a **new module** that has no
  test file yet — then create `tests/unit/test_<module>.py` as a fresh file.
- Do not add tests to `tests/integration/` when the test is a unit test, and
  vice versa.

### Test-specific antipatterns

Universal code antipatterns and error-handling rules — see `.agents/CODE.md`.
They apply here too: no magic crops, no bare `except` to silence a failure.

1. **No test-creation-by-guessing.** Always `find` or `rg` for existing test
   files before creating. Creating a duplicate test file is worse than not
   writing tests.

2. **No `unittest.TestCase`.** All tests use `pytest` with plain functions,
   never classes inheriting from `TestCase`. Use `pytest.fixture` for setup.

3. **No hard-coded paths in assertions.** Import path constants from
   `knox.core.paths`. The string `".knox"` must never appear in test
   assertions.

4. **No real LLM/VLM/embedding calls in any test.** Mock external
   dependencies via `pytest.monkeypatch`. The canonical pattern is to
   patch ``knox.llm.models.build_model`` to return pydantic-ai's
   ``TestModel`` (which generates valid output for any result schema with
   no network or model load). Embedding must be replaced by an in-process
   provider (e.g. ``DeterministicTestEmbeddingProvider`` from
   ``tests/conftest.py``). For VLM, monkey-patch
   ``knox.agents.vlm.VlmAgent.run`` or the relevant codepath. **No real
   LLM, VLM, or embedding call may reach an external engine from any
   test, ever.** This applies to integration tests too. See
   ``tests/integration/test_pipeline_mock.py`` for the canonical mock
   pattern and ``tests/integration/test_real_corpus.py`` /
   ``test_arxiv_pipeline.py`` for the only permitted exception: network
   downloads for preprocessing data (no LLM/VLM/embed calls).

5. **No filesystem litter.** Use `tmp_path` fixture for all filesystem
   operations. Never write to the real project tree.

6. **No slow tests.** Unit tests should complete in <5s per file. If a test
   needs heavy setup, use `@pytest.fixture(scope="session")`.

### Structure

- Test files are named `test_<module>.py`.
- Test functions are `test_<descriptive_name>`.
- Use `pytest.mark.parametrize` for multiple inputs/outputs.
- Prefer `monkeypatch` over `mock.patch` when possible (built-in, no cleanup).

## Work Guidance

- When fixing a bug, add a regression test first (or alongside the fix).
- When adding a new module, create the test file in the same PR.
- Run `uv run pytest tests/unit/ -x --timeout=30` before committing.
- Key flags: `-x` (fail-fast), `--timeout` (prevent hangs), `-k` (filter),
  `-n auto` (parallel).

## Verification

- `uv run pytest tests/unit/ -x --timeout=30` — unit suite
- `uv run pytest tests/integration/ -x --timeout=120` — integration suite
- `uv run pytest tests/ -x --timeout=120 -n auto` — all tests in parallel
- `cd tests && find . -name "test_*.py" | sort` — before creating a new file,
  check this list first
