# AGENTS.md — POLICIES

Always applicable. Boundaries, priorities, verification, checklist.

## Priorities

1. Correctness
2. Evidence
3. Safety
4. Minimal changes
5. Consistency
6. Performance

## Boundaries

- NEVER fabricate paths, commits, APIs, config keys, env vars, test results,
  or benchmark numbers. If you don't know, say so.
- NEVER guess at command names, flags, or paths. Read source or run `--help`.
- NEVER add secrets, API keys, or tokens to files. Use env vars.
- NEVER run destructive commands (`rm -rf`, `git reset --hard`,
  `git push --force`) without explicit confirmation.
- NEVER delete or move files without explicit instruction.
- NEVER create temp files in project root. Use dedicated temp dir.

## Change constraints

- Minimal, surgical edits. Preserve existing style.
- No new dependencies without explicit instruction.
- No unrelated refactoring while fixing a bug.
- When changing tests, update only directly affected tests.

## Code quality

Universal defaults. Project-specific standards live in child AGENTS.md.

- **SRP** — one reason to change per module/function. If a name needs "and",
  split it.
- **DRY** — before adding a function/constant/config block, check whether one
  already exists. Extract on the _third_ occurrence (Rule of Three), not earlier.
- **Redundant code** — remove dead branches, unreachable conditions, and unused
  params/imports before finishing.
- **Duplication vs. abstraction** — prefer a little duplication over the wrong
  premature abstraction.
- No new code-quality tooling, patterns, or value-object rules by default — opt
  in per project.

## Completion checklist

- Change solves the stated problem
- Relevant validation ran (or gaps explicitly stated)
- No unintended side effects introduced
- No secrets added or exposed

## Content rules (keep AGENTS.md lean)

- **No tree views.** Generate on demand with `rg --files | tree-cli --fromfile`.
- **No history.** Git log has it. Only record decisions costly to rediscover
  (in `.agents/HISTORY.md` with commit refs).
- **No TODO lists.** Use issue trackers or beads.

## Verification

- Run `ruff check` on changed Python files before committing
- Run `uv run pytest tests/unit -x --timeout=30` for unit test coverage
- Run `uv run pytest tests/integration -x --timeout=120` for integration tests
- Run `ty check src/knox` for type checking (mypy is not used)
- Run `uv run python scripts/check_field_consistency.py` to catch hard-coded contract keys
- Run `uv run python scripts/check_antipatterns.py` to catch banned-noqa, magic crops, `try/except ImportError` guards, and `global` keyword usage
- Run `mise run check-all` for the full local enforcement suite

## Response format

Concise and specific. No filler, intros, or restated requirements.
Answer direct questions directly.

For review/debugging/analysis: findings with references, conclusion,
approach. Mention caveats.

## Closeout

1. Run verification
2. Report docs intentionally left unchanged and why

---

## Project-specific Policies — KNOX

### Boundaries

- NEVER hardcode magic numbers, magic strings, or threshold constants in
  production code. Every parameter that affects behavior (token limits,
  failure thresholds, retry counts, timeouts, sizes) must be declared as a
  config field in the appropriate `_KnoxBaseModel` (e.g. `LLMConfig`,
  `BudgetConfig`) with a `Field(validation_alias=...)`. The production
  code reads the value from the config object — never from a bare literal or
  a module-level `Final` constant that duplicates what the config already
  owns. This rule applies even when the default matches the current hardcoded
  value — the field is the single source of truth.
- NEVER modify `.gitignore`, `pyproject.toml`, `ruff.toml`, or CI config
  without explicit user instruction.
- NEVER suppress `PLC0415` (inline imports), `ANN001` (missing return type),
  or `ANN202` (missing return type on public method). These are hard errors —
  no `# noqa` permitted. Refactor the code instead.
- NEVER use `from typing import TYPE_CHECKING` or wrap imports in
  `if TYPE_CHECKING:` blocks. This idiom diverges runtime from type-checker
  behavior, hides module-boundary problems, and adds cognitive overhead.
  Follow the `python-ultimate` skill (`references/type-checking.md`):
  break import cycles structurally (extract shared types to a neutral module,
  use a Protocol, or defer a genuinely runtime-only import to its call site),
  or use `from __future__ import annotations` + forward references.
  `if TYPE_CHECKING:` guards are forbidden in new code and must be removed
  wherever encountered.
- Use `./tmp` (gitignored) or `tempfile.TemporaryDirectory()` for temp files.

### Implementation Plans

- Implementation plans, refactoring plans, and design docs go in
  `.agents/plans/` — NOT under version control (gitignored).
- NEVER commit implementation plans to git. They are working artifacts,
  not shipped documentation.
- Shipped docs (user-facing guides, API references) go in `docs/` and
  ARE committed.

### User Preferences

- Concise, direct answers with no filler or intros
- Use `uv run` for all Python execution, never direct `python` calls
- English-only for code, docs, and commits
- Conventional commits format (`feat:`, `fix:`, `chore:`, `docs:`, etc.)
