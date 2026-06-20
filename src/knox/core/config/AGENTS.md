# DOX — knox.core.config

## Purpose

Configuration resolution engine for KNOX. Governs how config values are
discovered, merged, overridden, and accessed. All project configuration flows
through this package — no other module reads config files or env vars directly.

## Ownership

- `src/knox/core/config/` — config defaults, env var mapping, TOML loading,
  precedence logic, public API (`resolve_config`, `ResolvedConfig`)
- `knox.toml.example` (at project root) — reference config, the schema contract

## Local Contracts

### Architecture

```text
src/knox/core/config/
├── __init__.py    # Public API re-exports
├── _core.py       # Config resolution engine (merge, precedence, discovery)
├── keys.py        # Config key constants
├── _configs.py    # Config dataclass definitions
└── AGENTS.md      # This file
```

Public API lives in `__init__.py`. Internal helpers in `_core.py` are prefixed
with `_` and should not be imported outside this package or tests.

### Precedence order (memorize this)

```text
CLI args  >  env vars (KNOX_*)  >  project knox.toml  >  global config.toml  >  defaults
```

When adding a new config key, **all 5 layers must work**. Test each.

### Adding a new config key — checklist

1. **Add to `DEFAULTS` dict** in `_core.py` — every key must have a default.
2. **Add to `_ENV_MAP`** if the key makes sense as an env var override. Use the
   `KNOX_` prefix and the dotted config path as the dict key. Example:

   ```python
   "compile.cache.ttl": "KNOX_CACHE_TTL",
   ```

3. **Add to `_build_cli_overrides()`** in `commands.py` if there is a CLI flag.
   Map the CLI parameter name to the same dotted key used in `DEFAULTS`.
4. **Update `knox.toml.example`** — add the new key with a comment explaining
   valid values. Keep the same section structure as `DEFAULTS`.
5. **Wire into the consumer** (e.g. `PipelineConfig`, `LintRunner`) via
   `cfg.get("dotted.key", fallback)`.
6. **Write tests** — at minimum: default value, project config override, env var
   override, CLI override, full precedence chain.

### Conventions

- **TOML only** — no YAML, no JSON config files. Python 3.11+ has `tomllib`.
- **Dotted keys** — all config access uses dot-separated paths:
  `cfg.get("compile.llm.default_model")`. Never reach into `cfg.raw` directly.
- **`resolve_config()` is the single entry point** — callers should never
  call `_load_toml` or `_read_env_vars` directly.
- **No mutable global state** — `resolve_config()` returns a fresh
  `ResolvedConfig` each call. Do not cache it at module level.
- **New sections go in both places** — if you add `[compile.cache]` to the
  TOML schema, add `compile.cache` to `DEFAULTS` with all its sub-keys.
- **Boolean env vars** — `true/1/yes` → `True`, `false/0/no` → `False`.
  Integers are auto-coerced. Comma-separated strings become lists.
  All other values stay as strings.
- **Empty string = unset = None** — an empty string (`""`) is treated
  identically to `None` or a missing key. Consumers that support fallback
  chains (e.g. `dedup.model` → `compile.embedding.model`) must treat all
  three as "not provided" using `value or fallback`, never
  `if value is not None`.
- **`knox.toml.example` is the schema contract** — it must always reflect the
  current keys, defaults, and valid options. Treat it like a living spec.
- **`paths.py` is a leaf module** — never import `knox.core.config` from it.
  Dependency direction is one-way: `config` imports path constants and
  `VaultPaths` from `paths`; `paths` imports nothing from config. Config-driven
  vault paths are built in `PipelineConfig.paths` (`pipeline_defs.py`), which
  sits above both. A `config` import in `paths.py` recreates a hard import
  cycle — do not paper over it with `TYPE_CHECKING` or inline imports.

### File locations

| File | Purpose | Tracked in git |
|------|---------|----------------|
| `knox.toml.example` | Reference config with all keys commented out | Yes |
| `knox.toml` | User's actual project config | No (gitignored) |
| `~/.config/knox/config.toml` | User's global config | No |

## Verification

- Use `monkeypatch.chdir(tmp_path)` in tests that call `resolve_config()` —
  the real project root has a `knox.toml` that would leak into tests.
- Test precedence by setting up multiple layers and asserting the winner.
- Test provenance via `cfg.source_of("key")` — it must report the winning layer.
- After changing `DEFAULTS` or `_ENV_MAP`, verify each layer override still
  wins in the correct order.
- `knox.toml.example` must stay in sync with `DEFAULTS` — run a diff check.

---

## Child DOX Index

*This module has no child AGENTS.md files.*
