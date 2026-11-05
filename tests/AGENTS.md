# DOX — tests

<!-- on entry: loaded when agent enters tests -->
<!-- parent: ../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Test suites for the kompend project. Governs how tests are structured, where they
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

- **Find the existing test file first.** Every module under `src/kompend/` should
  have a corresponding `tests/unit/test_<module>.py`. Never create a new test
  file before verifying the right one doesn't already exist.
- If you are writing tests for `src/kompend/foo.py`, the test file is
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
   `kompend.core.paths`. The string `".kompend"` must never appear in test
   assertions.

4. **No real LLM/VLM/embedding calls in any test.** Mock external
   dependencies via `pytest.monkeypatch`. The canonical pattern is to
   patch ``kompend.llm.models.build_model`` to return pydantic-ai's
   ``TestModel`` (which generates valid output for any result schema with
   no network or model load). Embedding must be replaced by an in-process
   provider (e.g. ``DeterministicTestEmbeddingProvider`` from
   ``tests/conftest.py``). For VLM, monkey-patch
   ``kompend.agents.vlm.VlmAgent.run`` or the relevant codepath. **No real
   LLM, VLM, or embedding call may reach an external engine from any
   test, ever.** This applies to integration tests too. See
   ``tests/integration/test_pipeline_mock.py`` for the canonical mock
   pattern and ``tests/integration/test_real_corpus.py`` /
   ``test_arxiv_pipeline.py`` for the only permitted exception: network
   downloads for preprocessing data (no LLM/VLM/embed calls).

5. **No filesystem litter.** Use `tmp_path` fixture for all filesystem
   operations. Never write to the real project tree. The suite enforces this
   structurally: `tests/conftest.py` runs every test from a throwaway cwd
   (autouse `_isolate_cwd`), so a CLI invocation without `--wiki-dir` cannot
   grow a phantom `.kompend/` vault in the repository root. A test needing a
   specific directory chdirs itself; subprocess probes must pass the repo via
   `PYTHONPATH`, not `cwd`.

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
- Smoke tests spawn the real CLI: assert behaviour (the flag parses, the
  exit code, no traceback), never help-text wording. Cyclopts' meta app
  inherits root options into every subcommand's help, so "declared once"
  is a structural fact (`app.py`), not something `--help` can prove.
- A change that touches more than one stage of the image chain belongs in
  `tests/integration/test_image_pipeline.py` (figure → caption → page →
  vault), not in a per-stage test: the chain is what broke before, every
  stage was individually fine.
- Verify at the tier the change calls for (see **Verification tiers**); T2
  runs once, before commit — not three to four times per prompt.
- Key flags: `-x` (fail-fast), `-k` (filter), `-n auto` (pytest-xdist — opt-in
  for suite-scale runs; startup dominates single files), `-m 'not slow'`
  (skip tests tagged `slow`, the convention for any single test over 5 s).
  `timeout = 120` in pyproject (pytest-timeout) fails any hung test.

### Verification tiers (T0/T1/T2)

Agents never run the full unit suite unprompted mid-task. Verify at the
narrowest tier that covers the change, and state every command exactly as
it runs — no flags from plugins that are not installed.

- **T0 — changed code:** the test file(s) covering the change.
  `KOMP_LLM_TRANSIENT_BACKOFF=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/unit/test_<file>.py -q --no-header -p pytest_asyncio.plugin -p pytest_timeout`
- **T1 — blast radius:** T0 plus the test files that import the changed
  module (`rg -l "kompend\.<module>" tests/`), same command over that list.
- **T2 — full unit suite, once before commit:**
  `KOMP_LLM_TRANSIENT_BACKOFF=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/unit -q --no-header --tb=short -p pytest_asyncio.plugin -p pytest_timeout`.
  Integration tests only when the change touches them:
  `KOMP_LLM_TRANSIENT_BACKOFF=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/integration -q --no-header --tb=short -p pytest_asyncio.plugin -p pytest_timeout`.
- `KOMP_LLM_TRANSIENT_BACKOFF=0` (`src/kompend/core/env.py`) zeroes the
  transient-retry backoff, so a retry path under test does not really sleep.
- The `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` + `-p pytest_asyncio.plugin -p
  pytest_timeout` prefix skips the ~6 s/proc autoload tax (logfire, langsmith,
  anyio, platformdirs, cov otherwise join every run). `asyncio_mode = "auto"`
  is still honoured, but the plugin must be named `pytest_asyncio.plugin` —
  plain `-p pytest_asyncio` does not register it and every async test fails.
  `--no-cov` is invalid in that form (no cov plugin); drop the prefix for a
  coverage run (`uv run pytest … --cov=kompend`).

## Verification

- T0/T1/T2 commands above — the tiered policy is the default verification
- `cd tests && find . -name "test_*.py" | sort` — before creating a new file,
  check this list first
