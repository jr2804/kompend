# DOX — kompend.core.config

<!-- on entry: loaded when agent enters src/kompend/core/config -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Configuration resolution engine for kompend. Governs how config values are
discovered, merged, overridden, and accessed. All project configuration flows
through this package — no other module reads config files or env vars directly.

## Ownership

- `src/kompend/core/config/` — config defaults, env var mapping, TOML loading,
  precedence logic, public API (`resolve_config`, `ResolvedConfig`)
- `kompend.toml.example` (at project root) — reference config, the schema contract

## Local Contracts

### Architecture

- `__init__.py` — Public API re-exports
- `_core.py` — Config resolution entry point (merge, precedence, discovery)
- `generate.py` — Code-driven TOML generation from DEFAULTS
- `keys.py` — Config dotted-key constants (re-exports `EnvVar`)
- `../env.py` — `EnvVar` StrEnum, the **only** place a `"KOMP_*"` env-var name may be
  defined. Lives outside this package: `paths.py` reads env vars before the config
  package finishes loading. Enforced by `scripts/checks/check_antipatterns.py`
  (`ENV-VAR-LITERAL` rule).
- `_configs.py` — Config dataclass definitions
- `engine/` — Split resolution engine (discrete stages):
  - `_env.py` — Env var loading (``KOMP_*``)
  - `_loading.py` — TOML file loading
  - `_dedupe.py` — Duplicate-key tolerant TOML rewriting (keeps first occurrence)
  - `_overlay.py` — Config overlay merging
  - `_precedence.py` — Precedence ordering and final resolution
  - `_schema.py` — Config schema validation
  - `_check.py` — Config file diff vs schema defaults (backing ``kompend config check``)
  - `__init__.py` — Public re-exports for the engine
- `AGENTS.md` — This file

Public API lives in `__init__.py`. Internal helpers are prefixed with `_`
and should not be imported outside this package or tests.

`generate.py` provides `generate_scaffold_config()` (used by `kompend init` to
write `.kompend/config.toml` from `DEFAULTS` — no Jinja2 template involved) and
`extract_toml_keys()` / `get_default_keys()` (used by the sync test to verify
`kompend.toml.example` documents every key in `DEFAULTS`). Scaffold output
annotates enum-like keys with the remaining valid values as a trailing
comment (`key = "value"  # other, values`); the value domains come from
`enum_choices()` — schema StrEnum leaves plus `_STR_ENUM_CHOICES` for
str-typed keys with a closed domain.

`_loading.load_toml()` tolerates duplicated key assignments: tomllib's
duplicate errors ("Cannot overwrite a value", "Cannot declare … twice")
trigger `_dedupe.strip_duplicate_keys()`, which drops later
re-assignments (first occurrence wins) and logs a warning naming each
dropped key and line number. Only *adjacent* `[table]` re-declarations
are merged; non-adjacent re-opens and any other TOML error still fall
through to the hard-failure path (empty config + warning).

### Precedence order (memorize this)

```text
CLI args  >  env vars (KOMP_*)  >  project kompend.toml  >  global config.toml  >  defaults
```

When adding a new config key, **all 5 layers must work**. Test each.

### Project config discovery (Layer 3)

`resolve_config` locates the project config file by walking **up** from a
start dir. The start dir is:

1. **`--config <file>`** → that exact file (replaces discovery entirely).
2. **`--wiki-dir <vault>` / `KOMP_WIKI_DIR`** → discover from the vault.
   Operating on a vault means reading *that* vault's config, not the cwd's.
   This is what makes `kompend --wiki-dir vault ...` self-contained.
3. Otherwise → the **current working directory** (walk up the tree).

Helper: `_wiki_dir_for_discovery()` (`_core.py`). CLI/output-shape/tuning
values are config-only — override per-run with `--set key=val` (see CLI DOX).

### Adding a new config key — checklist

1. **Add to `DEFAULTS` dict** in `_core.py` — every key must have a default.
2. **Add the env alias** — `validation_alias=EnvVar.<NAME>` on the field in
   `_configs.py` (auto-derived into `ENV_MAP`), or add the name to the
   `EnvVar` enum in `../env.py` if it is a CLI-only override. Example:

   ```python
   ttl: int = Field(default=3600, validation_alias=EnvVar.CACHE_TTL)
   ```

3. **Do NOT add a CLI flag for output-shape or tuning keys.** Those are
   config-only (set via `kompend config set` or `--set`). Only input-set,
   pipeline-control, model-swap, and bool-toggle flags belong on the CLI —
   see `cli/args/AGENTS.md`.
4. **Update `kompend.toml.example`** — add the new key with a comment explaining
   valid values. Keep the same section structure as `DEFAULTS`.
   The sync test (`TestKompendDotExampleSync`) will fail if a key in `DEFAULTS`
   is missing from `kompend.toml.example`.
5. **Declare enum-like value domains** — a key whose values form a closed set
   must be discoverable by `enum_choices()` in `generate.py`: StrEnum-typed
   fields are picked up automatically; str-typed keys need an entry in
   `_STR_ENUM_CHOICES`. The domain drives the scaffold's trailing value
   comments and must match `kompend.toml.example`.
6. **Wire into the consumer** (e.g. `PipelineConfig`, `LintRunner`) via
   `cfg.get("dotted.key", fallback)` — or, for a group of related keys, hand
   the config group to the consumer (`display=cfg.kompend.display`) instead of
   copying single values around.
7. **Write tests** — at minimum: default value, project config override, env var
   override, CLI override, full precedence chain.
8. **A budget key shapes the work, it never excludes it** — a threshold warns,
   a window size splits a request (see the budget rule in `.agents/CODE.md`
   and `compilation/AGENTS.md`). There is no output-ceiling key: an answer is
   bounded by the model's context window, and the planners derive their answer
   room from it (`ContextWindow.answer_reserve`).

### Conventions

- **TOML only** — no YAML, no JSON config files. Python 3.11+ has `tomllib`.
- **Dotted keys** — all config access uses dot-separated paths:
  `cfg.get("compile.llm.default_model")`. Never reach into `cfg.raw` directly.
- **`DEFAULTS` is a nested dict** — `DEFAULTS.get("display.x", fallback)` always
  returns the fallback, because `DEFAULTS` is `KompendConfig().model_dump()`
  (sections, not dotted keys). Read a value from the *resolved* config
  (`cfg.get("display.x")`, `cfg.kompend.display.x`) and pass the config group
  to the consumer; never read a dotted key out of `DEFAULTS`.
- **A key either has an effect or does not exist** — a key with no reader is
  removed from the schema, the `EnvVar` enum, `keys.py` and
  `kompend.toml.example`, and registered in `REMOVED_KEYS`
  (`engine/_schema.py`) so old vaults get a deprecation warning instead of a
  crash. A key with a reader but no wiring gets wired.
- **`resolve_config()` is the single entry point** — callers should never
  call `_load_toml` or `_read_env_vars` directly.
- **No mutable global state** — `resolve_config()` returns a fresh
  `ResolvedConfig` each call. Do not cache it at module level.
- **New sections go in both places** — if you add `[compile.cache]` to the
  TOML schema, add `compile.cache` to `DEFAULTS` with all its sub-keys.
- **Boolean env vars** — `true/1/yes` → `True`, `false/0/no` → `False`.
  Integers are auto-coerced. Comma-separated strings become lists.
  All other values stay as strings.
- **Env-list set is schema-derived** — which env vars split on commas is
  computed from list-typed fields, never hand-listed. Direct env reads at the
  discovery boundary must use the config-map alias (e.g. `KOMP_WIKI_DIR`),
  never a hand-written variant.
- **Empty string = unset = None** — an empty string (`""`) is treated
  identically to `None` or a missing key. Consumers that support fallback
  chains (e.g. `dedup.model` → `compile.embedding.model`) must treat all
  three as "not provided" using `value or fallback`, never
  `if value is not None`.
- **`kompend.toml.example` is the schema contract** — it must always reflect the
  current keys, defaults, and valid options. Treat it like a living spec.
- **`_check.py` walks the live pydantic model, not `DEFAULTS`** — optional
  nested models (e.g. `compile.llm.rlm`) store `None` in the `DEFAULTS` dump,
  so `flat_keys(DEFAULTS)` silently misses their leaves. `_schema_leaf_keys()`
  / `_schema_defaults()` recurse `KompendConfig` field types to surface them.
  Only report a nested-model leaf when the user actually wrote its parent
  section (`_parent_key_present`), so an untouched optional model contributes
  no rows.
- **User-named namespaces are matched, not enumerated** — a `dict[str, Model]`
  field (`providers`, `compile.skills`, `providers.<name>.models`) becomes a
  `providers.*.base_url` pattern: the user picks that segment, the model's fields
  validate the rest, so an unknown sub-key is still an error. A free-form table
  (`categories`, `frontmatter`, `hooks`) is an open namespace — its keys are data.
  One cached walk (`_schema_walk`) feeds keys, defaults and open namespaces, so the
  three views cannot drift. A dotted name inside the namespace
  (`models."hf.co/qwen3.6"`) needs `resolve_dotted()`, because `get_dot()` splits
  on every `.` and cannot see the key.
- **`paths.py` is a leaf module** — never import `kompend.core.config` from it.
  Dependency direction is one-way: `config` imports path constants and
  `VaultPaths` from `paths`; `paths` imports nothing from config. Config-driven
  vault paths are built in `PipelineConfig.paths` (`pipeline_defs.py`), which
  sits above both. A `config` import in `paths.py` recreates a hard import
  cycle — do not paper over it with `TYPE_CHECKING` or inline imports.

### File locations

| File | Purpose | Tracked in git |
|------|---------|----------------|
| `kompend.toml.example` | Reference config with all keys commented out | Yes |
| `kompend.toml` | User's actual project config | No (gitignored) |
| `~/.config/kompend/config.toml` | User's global config | No |

## Verification

- Use `monkeypatch.chdir(tmp_path)` in tests that call `resolve_config()` —
  the real project root has a `kompend.toml` that would leak into tests.
- Test precedence by setting up multiple layers and asserting the winner.
- Test provenance via `cfg.source_of("key")` — it must report the winning layer.
- After changing `DEFAULTS` or `_ENV_MAP`, verify each layer override still
  wins in the correct order.
- `kompend.toml.example` must stay in sync with `DEFAULTS` — enforced by
  `TestKompendDotExampleSync` in `tests/unit/test_config.py`.
