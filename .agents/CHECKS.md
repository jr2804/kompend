# AGENTS.md — Checks

Verification commands, enforcement scripts, and new-code checklist.

## General Checks

### Pattern: New Code Checklist

Before writing a function, tick these. If you can't, stop and fix the foundation
first — the alternatives are all antipatterns:

- **Limit values** — from config/DEFAULTS/field, not a literal slice?
- **Enum values** — from StrEnum members, not bare strings?
- **Imports** — at module level, not inside function bodies?
- **Global state** — no `global` keyword or module-level mutable singletons?
- **Noqa suppressions** — actually necessary, with a justification comment?

### Pattern: Enforcement Scripts

A small set of scripts catch what linting misses: bare-string contract keys,
magic-number crops, banned-noqa patterns. Run them locally before pushing.
All exit non-zero on violations; all run in CI.

### Extension

| Who | When | What |
|-----|------|------|
| Project maintainer | Adding a new verification tool | Verification section |
| Project maintainer | Adding a new enforcement script | Enforcement table |
| Project maintainer | Adding a new checklist item | Checklist |

---

## Project-specific Checks — KNOX

### Verification

- Run `ruff check` on changed Python files before committing
- Run `uv run pytest tests/unit -x --timeout=30` for unit test coverage
- Run `uv run pytest tests/integration -x --timeout=120` for integration tests
- Run `ty check src/knox` for type checking (mypy is not used)
- Run `uv run python scripts/check_field_consistency.py` to catch hard-coded contract keys
- Run `uv run python scripts/check_antipatterns.py` to catch banned-noqa, magic crops, `try/except ImportError` guards, and `global` keyword usage
- Verify `knox.toml.example` stays in sync with `DEFAULTS` when adding config keys
- After changing `src/knox/core/paths.py`, verify no hard-coded path strings
  exist in other `.py` files
- After changing CLI commands, MCP tools, config keys, or output format, check
  `skills/knox/SKILL.md` for stale or missing info
- Run `uv run mkdocs build --strict` when changing documentation files

### New Code Checklist

Tick these BEFORE writing a function:

- **Limit value** — from `config.get(...)` / `DEFAULTS` / `PipelineConfig` field?
  Literals like `[:10]` silently drop data. Use `DisplayConfig` for display
  limits or `BudgetConfig` for LLM input/output budgets.
- **Role / task / compiler / tier** — from `LLMRole` / `AgentTask` / `Compiler` /
  `Tier` enum? Bare strings like `role="compile.extract"` will be rejected.
- **Import** — at module level? Function-body imports (`PLC0415`) are rejected
  except for `try/except ImportError` probes for explicitly optional deps.
- **Global state** — no `global` keyword or module-level mutable singleton
  (`_cache = {}`, `_instance = None`).
- **Noqa suppression** — actually necessary with a justification comment?
  `PLC0415` and other banned-noqa rules cannot be suppressed.

### Enforcement scripts

| Script | Catches | Lives in |
|---|---|---|
| `ruff check` | Most lint issues (PLC0415/E402 are warnings, not errors) | ruff.toml |
| `scripts/check_field_consistency.py` | Bare-string contract keys | scripts/ |
| `scripts/check_antipatterns.py` | Banned-noqa, magic crops, `try/except ImportError` guards, `global` | scripts/ |
| `scripts/analyze_codebase.py` | Files/submodules exceeding net-line targets (>350 lines/file, >7 files/submodule) | scripts/ |

All three run in CI. Run locally: `mise run check-all`.
Run the codebase analyzer: `uv run python scripts/analyze_codebase.py`.
