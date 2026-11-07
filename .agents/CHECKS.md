# AGENTS.md — Checks

Verification commands, enforcement scripts, and new-code checklist.

## General Checks

### Pattern: Enforcement Scripts

A small set of scripts catch what linting misses: bare-string contract keys,
magic-number crops, banned-noqa patterns. `mise run check-all` runs them; no CI
job does (see **Wiring** below).

Two guarantees keep them honest: a scan that finds no files is an error (an
empty scan is not a pass), and an enforced contract that no longer exists in
the code (moved module, renamed enum) fails the script instead of warning.
Per-line waivers exist for sites the rule cannot judge — `# nocheck: <reason>`
in field-consistency, `# noqa: <code>` in antipatterns; the reason states why
this site is legitimate.

**Extension.** Adding a verification tool, enforcement script, or checklist item
updates the matching section below.

## Project-specific Checks — kompend

### Verification

- Run `ruff check` on changed Python files before committing
- pytest runs are tiered — never the full unit suite unprompted mid-task; T0 = the test file(s) covering the change: `KOMP_LLM_TRANSIENT_BACKOFF=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/unit/test_<file>.py -q --no-header -p pytest_asyncio.plugin -p pytest_timeout`, T1 = T0 plus the test files importing the changed module (`rg -l "kompend\.<module>" tests/`)
- T2 = full unit suite once, before commit: `KOMP_LLM_TRANSIENT_BACKOFF=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/unit -q --no-header --tb=short -p pytest_asyncio.plugin -p pytest_timeout`; integration suite only when the change touches it: `KOMP_LLM_TRANSIENT_BACKOFF=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/integration -q --no-header --tb=short -p pytest_asyncio.plugin -p pytest_timeout` (policy detail: `tests/AGENTS.md`)
- Run `ty check src/ tests/ scripts/` for type checking (mypy is not used); CI's `unit` job runs the same command
- Run `uv run python scripts/checks/all.py` — every gate; run one directly for its detail. The runner exits `0` when every gate judged and found nothing, `1` on a violation, `2` when a gate could not judge.
- Run `uv run pytest tests/smoke` after touching the CLI surface: it spawns the real `kompend` (init, config check, status, doctor, lint show, empty-vault compile) and fails on a traceback
- Verify `kompend.toml.example` stays in sync with `DEFAULTS` when adding config keys
- After changing CLI commands, MCP tools, config keys, or output format, check `skills/kompend/SKILL.md` for stale or missing info
- Run `uv run mkdocs build --strict` when changing documentation files

### New Code Checklist

Tick these BEFORE writing a function:

- **Limit value** — from `config.get(...)` / `DEFAULTS` / `PipelineConfig` field?
  Literals like `[:10]` silently drop data; use `DisplayConfig` / `BudgetConfig`.
- **Role / task / compiler / tier** — from `LLMRole` / `AgentTask` / `Compiler` /
  `Tier` enum? Bare strings like `role="compile.extract"` will be rejected.
- **Import** — at module level; `PLC0415` is banned and cannot be suppressed.
  Only a `try/except ImportError` probe for an explicitly optional dep is exempt.
- **Global state** — no `global` keyword and no module-level mutable singleton
  (`_cache = {}`, `_instance = None`, a one-key dict slot). Constant once
  computed → `@functools.cache` getter returning a frozen value; changing during
  the process → an object created and passed (or a `ContextVar` for a genuinely
  cross-thread switch). `# noqa: PLW0603` is never allowed.
- **Vocabulary** — is a closed set of names (backend, matcher, flavor, preset,
  tier, …) defined as a `StrEnum`? Dict keys, comparisons and defaults use
  members, never their bare string values (`STR-KEYED-REGISTRY` /
  `ENUM-VALUE-LITERAL` reject both).
- **Noqa suppression** — actually necessary with a justification comment?

### Enforcement scripts

| Script                                      | Catches                                                                                                                                                                 | Lives in         |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| `ruff check`                                | Most lint issues (PLC0415/E402 are warnings, not errors)                                                                                                                | ruff.toml        |
| `scripts/checks/check_field_consistency.py` | Bare-string contract keys                                                                                                                                               | scripts/checks/  |
| `scripts/checks/check_antipatterns.py`      | Banned-noqa, magic crops, `try/except ImportError` guards, string-keyed registries, bare enum-value literals, bare reason-code literals, pip-side installs in code/docs | scripts/checks/  |
| `scripts/checks/check_persisted_paths.py`   | An absolute filesystem path persisted in `.kompend/` (every text column of `kompend.db` plus loose JSON/JSONL)                                                          | scripts/checks/  |
| `scripts/checks/check_dox.py`               | DOX file size budgets, cross-file duplication                                                                                                                           | scripts/checks/  |
| `scripts/checks/check_enum_cache.py`        | A `StrEnum` reachable from a cached dataclass missing from `_CACHE_ENUMS`                                                                                               | scripts/checks/  |
| `scripts/checks/check_config_literals.py`   | A literal repeating a config default, or a dotted config key as a string                                                                                                | scripts/checks/  |
| `scripts/checks/check_cli_surface.py`       | A run-level flag (`--config`) declared on a command instead of the root app                                                                                             | scripts/checks/  |
| `scripts/checks/all.py`                     | Runs every gate in one command; `[ok]`/`[n/a]`/`[FAIL]`, exits `0`/`2`/`1`                                                                                              | scripts/checks/  |
| `scripts/analyze/analyze_codebase.py`       | Files/submodules exceeding net-line targets (>350 lines/file, >7 files/submodule)                                                                                       | scripts/analyze/ |

**Wiring.** The pre-commit hook lives at `.githooks` (`mise run install-hooks`):
it runs the forbidden ruff rules plus several gates, then `_delegate.sh`
forwards to hook files other tools own in `.beads/hooks` (beads, repowise) —
never edit those, they regenerate. `check_dox.py` covers the DOX files plus the
core `.agents/` state; `.agents/skills/**` and `.agents/history/**` are skipped.

### Gate exit codes

Above the generic rule ("a scan that finds no files is an error"), a gate must
assert four more failure faces and the exit-code convention:

- **Identity** — a scan of the _wrong_ source is not empty, so no coverage check
  sees it: reading one project with another's facts is confident and wrong.
  Resolve a fact from the caller or the scanner itself, never a guessing
  fallback — make the source non-negotiable, not the fallback smarter.
- **Facts** — a rule deciding by facts read elsewhere (declared dependencies, an
  import→distribution map, a schema) exits 2 when they are unavailable: an
  empty fact set is "unknown", not "none". `require_facts` sees a count, not a
  shape, so each caller needs a test that empties that fact and asserts exit 2.
- **Errors** — an unreadable or unparseable input exits 2; `except X: return []`
  inside a gate disarms it. A scan error is no finding and no pass.
- **Circular evidence** — a probe must not share the assumption it tests.
  `callable(info.default)` "showed" no config field used a factory while that
  same test WAS the bug (a factory leaves `PydanticUndefined`), confirming the
  defect instead of measuring it. Ask what the predicate reports if the
  assumption is false; "the same" means it tests nothing.
- **Codes** — `0` clean, `1` violations, `2` could not establish what it needs
  to judge. Never `0` for the last case.

**Guess direction.** Prefer the failure a human sees: the finding-producing value
over a `None`/sentinel that compares false, a missed finding over a false alarm.
A lookup table inverts this — omit the entry rather than guess one; a wrong entry
looks exactly like an honest probe, while a missing one exits 2 in a project file
_and_ in a PEP 723 script. Ask what a _wrong_ entry does: if it can only make the
scanner quieter (selecting what to check, resolving to nothing), the typo is the
silent case, and that table needs a test that every entry still matches reality.

**Migrations disarm gates.** A storage change empties a scanner's input without
touching it: `check_persisted_paths` read loose JSON under `.kompend/`, the state
moved into `kompend.db`, and it kept exiting 0 reading nothing. When a layout
moves, re-check every gate that reads it.

**A gate needs a subject, and the runner supplies it.** The same migration left
`check_persisted_paths` needing a VAULT, and the repository root is not one —
its `.kompend/` is gitignored, so a fresh checkout never has a store. A gate
that reads nothing must exit 2 (never 0), so the RUNNER gives it something:
`all.py` builds a throwaway vault holding a real store, and the gate judges the
repository's own DDL instead of declining.
