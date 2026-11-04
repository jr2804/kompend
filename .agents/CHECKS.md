# AGENTS.md — Checks

Verification commands, enforcement scripts, and new-code checklist.

## General Checks

### Pattern: Enforcement Scripts

A small set of scripts catch what linting misses: bare-string contract keys,
magic-number crops, banned-noqa patterns. Run them locally before pushing.
All exit non-zero on violations; `mise run check-all` runs them (no CI job
does — see **Wiring** below).

Two guarantees keep them honest: a scan that finds no files is an error (an
empty scan is not a pass), and an enforced contract that no longer exists in
the code (moved module, renamed enum) fails the script instead of warning.
Per-line waivers exist for sites the rule cannot judge — `# nocheck: <reason>`
in field-consistency, `# noqa: <code>` in antipatterns. The reason is part of
the waiver: it states why this site is legitimate.

### Extension

| Who | When | What |
|-----|------|------|
| Project maintainer | Adding a new verification tool | Verification section |
| Project maintainer | Adding a new enforcement script | Enforcement table |
| Project maintainer | Adding a new checklist item | New Code Checklist |

The new-code checklist itself lives once, in the project section below.

---

## Project-specific Checks — kompend

### Verification

- Run `ruff check` on changed Python files before committing
- Run `uv run pytest tests/unit -x --timeout=30` for unit test coverage
- Run `uv run pytest tests/integration -x --timeout=30` for integration tests (no real-engine calls — must be fast)
- Run `ty check src/kompend` for type checking (mypy is not used)
- Run `uv run python scripts/checks/all.py` - the single entry point for every gate
- Run `uv run pytest tests/smoke` after touching the CLI surface: it spawns the real `kompend` (init, config check, status, doctor, lint show, empty-vault compile) and fails on a traceback
- Verify `kompend.toml.example` stays in sync with `DEFAULTS` when adding config keys
- After changing `src/kompend/core/paths.py`, verify no hard-coded path strings
  exist in other `.py` files
- After changing CLI commands, MCP tools, config keys, or output format, check
  `skills/kompend/SKILL.md` for stale or missing info
- Run `uv run mkdocs build --strict` when changing documentation files

Individual gates (`check_antipatterns`, `check_enum_cache`, `check_config_literals`,
`check_cli_surface`, `check_field_consistency`, `check_persisted_paths`) are
listed with what each catches under **Enforcement scripts**; run one directly to
see its detail, or `all.py` for the whole set.

### New Code Checklist

Tick these BEFORE writing a function:

- **Limit value** — from `config.get(...)` / `DEFAULTS` / `PipelineConfig` field?
  Literals like `[:10]` silently drop data. Use `DisplayConfig` for display
  limits or `BudgetConfig` for LLM input/output budgets.
- **Role / task / compiler / tier** — from `LLMRole` / `AgentTask` / `Compiler` /
  `Tier` enum? Bare strings like `role="compile.extract"` will be rejected.
- **Import** — at module level? Function-body imports (`PLC0415`) are rejected
  except for `try/except ImportError` probes for explicitly optional deps.
- **Global state** — no `global` keyword and no module-level mutable
  singleton (`_cache = {}`, `_instance = None`, a one-key dict slot). Constant
  once computed → `@functools.cache` getter returning a frozen value; changes
  during the process → an object created and passed (or a `ContextVar` for a
  genuinely cross-thread switch). `# noqa: PLW0603` is never allowed.
- **Vocabulary** — is a closed set of names (backend, matcher, flavor, preset,
  tier, …) defined as a `StrEnum`? Dict keys, comparisons and defaults use
  members, never their bare string values (`STR-KEYED-REGISTRY` /
  `ENUM-VALUE-LITERAL` reject both).
- **Noqa suppression** — actually necessary with a justification comment?
  `PLC0415` and other banned-noqa rules cannot be suppressed.

### Enforcement scripts

| Script | Catches | Lives in |
|---|---|---|
| `ruff check` | Most lint issues (PLC0415/E402 are warnings, not errors) | ruff.toml |
| `scripts/checks/check_field_consistency.py` | Bare-string contract keys | scripts/checks/ |
| `scripts/checks/check_antipatterns.py` | Banned-noqa, magic crops, `try/except ImportError` guards, string-keyed registries, bare enum-value literals, pip-side installs in code/docs | scripts/checks/ |
| `scripts/checks/check_persisted_paths.py` | An absolute filesystem path persisted in `.kompend/` (every text column of `kompend.db` plus loose JSON/JSONL) | scripts/checks/ |
| `scripts/checks/check_dox.py` | DOX file size budgets, cross-file duplication | scripts/checks/ |
| `scripts/checks/check_enum_cache.py` | A `StrEnum` reachable from a cached dataclass missing from `_CACHE_ENUMS` | scripts/checks/ |
| `scripts/checks/check_config_literals.py` | A literal repeating a config default, or a dotted config key as a string | scripts/checks/ |
| `scripts/checks/check_cli_surface.py` | A run-level flag (`--config`) declared on a command instead of the root app | scripts/checks/ |
| `scripts/checks/all.py` | Runs every gate in one command; stops at the first failure | scripts/checks/ |
| `scripts/analyze/analyze_codebase.py` | Files/submodules exceeding net-line targets (>350 lines/file, >7 files/submodule) | scripts/analyze/ |

**Wiring.** Git honours one `core.hooksPath`; this repo points it at
`.githooks` (`mise run install-hooks`). `.githooks/pre-commit` runs the
forbidden ruff rules plus the three gates, then `_delegate.sh` forwards to the
hook files other tools own in `.beads/hooks` (beads, repowise) — never edit
those, the tools regenerate them. `check_dox.py`'s scope is the DOX files plus
the core `.agents/` state files; `.agents/skills/**` and `.agents/history/**` are
not project DOX and are skipped. No CI job runs these; `mise run check-all` runs
the two AST gates.

### Gate exit codes

The generic rule above ("a scan that finds no files is an error") has three
more faces. A gate asserts each before reporting success:

- **Facts** — a rule deciding by facts read elsewhere (declared dependencies, an
  import→distribution map, a schema) exits 2 when they are unavailable. An
  empty fact set is "unknown", not "none".
- **Errors** — an unreadable or unparseable input exits 2; `except X: return []`
  inside a gate disarms it silently. A scan error is no finding and no pass.
- **Codes** — `0` clean, `1` violations, `2` could not establish what it needs
  to judge. Never `0` for the last case.

**Migrations disarm gates.** A storage change empties a scanner's input without
touching the scanner: `check_persisted_paths` read loose JSON under `.kompend/`,
the state moved into `kompend.db`, and the gate kept exiting 0 reading nothing.
When a schema or artifact layout moves, re-check every gate that reads it.
