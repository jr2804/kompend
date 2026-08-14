# DOX — knox.cli

<!-- on entry: loaded when agent enters src/knox/cli -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

The CLI layer is a thin shell over the KNOX core library. It parses user input,
calls the right core function, and formats the output. Nothing more.

`knox` must be usable as a Python library without importing `knox.cli`.

## Ownership

- `src/knox/cli/` — all CLI command definitions, argument declarations, output
  formatting, and thin wrappers that delegate to core
- CLI-specific orchestration (file resolution, URL downloads, source copies)

## Local Contracts

### What belongs in `knox.cli`

- Typer command definitions (`@app.command`)
- `Annotated` argument/option declarations (`args/` submodule)
- Input validation at the string level (file exists? flag parsing?)
- Output formatting (print, `--json`, progress bars, `typer.Exit`)
- CLI-specific orchestration (resolve files, download URLs, copy to sources dir)
- Thin wrapper helpers that delegate to core with a console print after

### Lazy command registration

Command modules are NOT imported eagerly at startup. `app.py` maintains a
`_LAZY_MODULES` dict mapping command names to `(module_name, attribute_name)`.
Only the module for the invoked command is imported (or all for `--help`).

**Why:** eager imports pulled heavy deps (pydantic-ai, docling, MCP SDK)
into every `knox` invocation. With lazy registration, `knox --version` takes
~1.5s instead of ~23s.

**Two-layer lazy:** `app.py` lazy-imports command modules on first use, AND
`cmds/__init__.py` uses `__getattr__` to avoid eagerly importing sibling
modules when any single command is accessed.

**Adding a new command:**

1. Add `"name": ("knox.cli.cmds.module", "app")` to `_LAZY_MODULES` in `app.py`
2. Add `"name"` to `_MODULE_NAMES` and `__all__` in `cmds/__init__.py`
3. Tests: `conftest.py` calls `register_all_commands()` at session start
4. No other changes needed — the module is imported on first invocation

### What does NOT belong in `knox.cli`

| Prohibited | Belongs in |
|---|---|
| Business logic (parsing, chunking, compiling, linting) | `knox.core`, `knox.lint`, `knox.compilation` |
| Data models and constants | `knox.core.models` |
| Config defaults and resolution logic | `knox.core.config` |
| LLM calls or LLM client construction | `knox.llm` |
| Parser registry / router construction | `knox.preprocessing.parsers` |
| Hard-coded magic numbers (chunk sizes, thresholds) | `knox.core.config` DEFAULTS |

### The test

If you can remove `src/knox/cli/` and `import knox` still works for
programmatic use (pipeline, models, config, lint), then the boundary is clean.

### File responsibilities

| File/Directory | Purpose |
|---|---|
| `app.py` | Typer app instance, version callback, **lazy command registration** |
| `args/` | Argument type definitions organized by domain (see `args/AGENTS.md`) |
| `cmds/` | Command implementations organized by domain (see `cmds/AGENTS.md`) |
| `_sources.py` | File resolution, glob expansion, copy-to-sources, URL download. |
| `_ignore.py` | Ignore/gitignore pattern matching for source scanning. |
| `_pipeline_helpers.py` | Pipeline orchestration helpers (progress callback, input resolution, review queue, provenance, lint-fix persistence). |
| `_progress.py` | Rich progress bar wrappers. |
| `_shared.py` | Shared utilities (config resolution, CLI overrides, hook loading). |

### Argument declarations (`args/` submodule)

See `src/knox/cli/args/AGENTS.md` for detailed argument organization rules.

**Summary:**

- All `typer.Option` / `typer.Argument` live in `args/` as `Annotated` types.
- Arguments organized by domain into 9 modules: `common.py`, `compile.py`, `parse.py`, `query.py`, `pipeline.py`, `config.py`, `init.py`, `memory.py`, `other.py`.
- Commands import the types, never call `typer.Option()` inline (B008).
- Default values are module-level constants (`DEFAULT_CHUNK_SIZE`, etc.).
- **These constants are the CLI's own defaults, not the source of truth.** The
  source of truth is `knox.core.config.DEFAULTS`. The CLI constants exist only
  for Typer's `--help` text and as fallback values passed to `resolve_config()`.

### Command functions (`cmds/` submodule)

See `src/knox/cli/cmds/AGENTS.md` for detailed command organization rules.

**Summary:**

- One file per command group organized by domain. Groups are cohesive sets of related commands.
- 14 command modules: `capture.py`, `config.py`, `doctor.py`, `init.py`, `lint.py`, `memory.py`, `organize.py`, `pack.py`, `pipeline.py`, `quality.py`, `query.py`, `review.py`, `serve.py`, `status.py`.
- Each command function is thin: parse args → resolve config → call core → print.
- Never put a `for` loop that iterates over wiki nodes and does real work.
  That's business logic — move it to a core module.
- `try/except` at the top level is fine for user-facing error messages.
- Use `get_console().print(msg)` for stdout output, `get_err_console().print(msg)` for stderr. Import from `knox.logging`.
- `--json` flag: serialize and print. No other behavior change.

### CLI overrides → config

Every CLI flag that maps to a config key must:

1. Be listed in `build_cli_overrides()` with the dotted config key.
2. The command function passes the override to `resolve_config()`.
3. The resolved value comes from `cfg.get("dotted.key")`, never from the raw
   CLI parameter directly (unless it's purely a CLI concept like `--json`).
4. Use tri-state `bool | None` + `--flag/--no-flag` for config-backed toggles
   (see `args/AGENTS.md`: Bool flag pattern). Override with `if val is not None:`
   so `--no-X` is honored — never bare `if val:` (swallows False).

### Config-only values (no CLI flag)

Output-shape values (`format`, `flavor`, `frontmatter`, `kind`, `compiler`,
`image_mode`) and tuning numbers (`budget.*`, `llm.timeout`, `embedding.batch_size`,
`memory_half_life_days`, `memory_decay_threshold`) have **no CLI flag** — they
are set-once vault properties. Change them with `knox config set <key> <val>`
(persistent) or override one run with `--set <key>=<val>` (ephemeral). Never
re-add a flag for these — use the `--set` escape hatch instead.

`--set` is parsed by `parse_cli_set()` (in `cmds/pipeline.py`): each `key=value`
is validated via `validate_config_key()` and coerced via `coerce_config_value()`,
then merged last by `build_cli_overrides(cli_set=...)` so it wins over every
other flag.

### Vault config discovery

`knox --wiki-dir <vault> ...` reads *that* vault's config (`.knox/config.toml`),
not the cwd's. See `core/config/AGENTS.md`: Project config discovery. Seed a
vault with a chosen config at creation time via `knox init <dir> --config <toml>`
(the file is copied verbatim into the vault).

### Progress bars

**Recurring trap:** a progress bar looks "stuck," gets
"fixed" with a plumbing change, and still looks stuck — because the real
cause was a slow/hung backend operation, not the bar. This has happened
repeatedly. Break the cycle with this diagnostic order:

1. **Confirm the total is correct first.** A bar that shows "Captioning
   images (171)" means the `reset_for_phase(total, ...)` call worked — the
   plumbing is wired. A wrong total (e.g. 0) is a real plumbing bug; a
   correct total is not.
2. **Confirm the elapsed timer is advancing.** If `0:12:11` keeps climbing,
   the bar is live and rendering — the loop is just blocked. A frozen
   timer means the UI thread died (a different bug).
3. **Then conclude: the bar is correct, the *work* is slow.** 0% with a
   correct total + advancing timer = the first unit is still running. A
   single VLM/LLM call that hangs until timeout (e.g. `glm-ocr` looping
   on a degenerate token) will pin the bar at 0% for the full timeout
   duration × retry count.

Do NOT "fix" a bar that passes steps 1–2. Investigate the backend call
that isn't completing. The fix lives in `src/knox/llm/`, the resolver,
or the model config (`timeout`, `fallback_max_retries`), not in
`_progress.py` or `_pipeline_helpers.py`.

**Signal-path contract:** a phase advances only when the core tier calls
`self._emit(phase, current, total, detail)`. The CLI handler in
`_pipeline_helpers._make_progress_callback` maps `(current, total, detail)`
to `tracker.reset_for_phase` / `tracker.update_progress`. When adding a
sub-phase (like the VLM captioning burst inside resolve), emit a sentinel
`(0, n, "<phase_name>")` to switch the bar, then `(i+1, n, "<detail>")`
per unit. The handler must recognize both the sentinel `detail` string and
the per-unit one.

**Determinate vs indeterminate:** bars with a known `total` (parse,
resolve, image-captioning) fill as units complete. Bars without one
(compile, output) are indeterminate and convey activity via description
updates, not percentage. Never fake a percentage on an indeterminate task.

### No direct core imports in thin wrappers

Helper functions like `_send_to_review_queue`, `_annotate_provenance`,
`_verify_claims` are acceptable as thin wrappers — they import the core
function, call it, and print a status line. The test: if you strip the
`get_console().print()` line, the remaining code should be a single function call
to a core module.

## Known violations (do not replicate)

These patterns remain in the current codebase and should be refactored when
touched, not extended:

| Violation | Where | Why it's wrong |
|---|---|---|
| `doctor.py` has its own parser registration logic | `doctor.py:58-74` | Should reuse `build_default_router()` from the parser registry |
| `status.py` imports from `knox.cli._sources` | `status.py:10` | Cross-imports within CLI are fine, but extension enumeration should come from parser registry |

### Fixed violations (do not re-introduce)

| Fixed | Was |
|---|---|
| `_build_router()` → `build_default_router()` in `knox.preprocessing.parsers._registry` | Parser construction in CLI |
| `_load_wiki()` → `load_cached_wiki()` in `knox.core.models` | JSON deserialization in CLI |
| `_build_vector_index()` → `build_vector_index()` in `knox.rag` | RAG business logic in CLI |
| `doctor.py` now uses `LLMClient.try_complete()` | Direct LLM client call |
| `status.py` now uses `find_project_config()` | Hardcoded `.knox/config.yaml` path |
| `SUPPORTED_EXTENSIONS` → `get_supported_extensions()` querying parser registry | Duplicated extension set |

## Work Guidance

- Never expand the CLI's role beyond input/output formatting and thin delegation.
- When adding a new command: add `Annotated` types to `args/` submodule, add the
  `@app.command` function to the appropriate `cmds/` module file (create one
  if no cohesive group exists), add argument → config mapping in
  `_build_cli_overrides()`.
- Use `get_console()` / `get_err_console()` from `knox.logging` for all output — not `typer.echo()` or `print()`.

## Verification

- `uv run python -m knox.cli --help` — app starts and shows commands
- `uv run pytest tests/unit/ -k "cli" -x --timeout=30` — CLI tests pass
- `import knox` works without importing `knox.cli`

---

## Child DOX Index

| Path | Scope |
|------|-------|
| `src/knox/cli/args/AGENTS.md` | Argument definitions — Annotated types, domain organization, reusable type patterns |
| `src/knox/cli/cmds/AGENTS.md` | Command implementations — @app.command registration, thin wrapper patterns, sub-apps (config, init, memory) |
