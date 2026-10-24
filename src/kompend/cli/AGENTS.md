# DOX — kompend.cli

<!-- on entry: loaded when agent enters src/kompend/cli -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

The CLI layer is a thin shell over the kompend core library. It parses user input,
calls the right core function, and formats the output. Nothing more.

`kompend` must be usable as a Python library without importing `kompend.cli`.

## Ownership

- `src/kompend/cli/` — all CLI command definitions, argument declarations, output
  formatting, and thin wrappers that delegate to core
- CLI-specific orchestration (file resolution, URL downloads, source copies)

## Local Contracts

- **A run-level input is declared once, on the root app.** `--wiki-dir` and
  `--config` live in `_callback` (`ctx.obj`); commands read them via
  `vault_paths_from_ctx` / `config_file_from_ctx`. Never add a per-command copy.

### What belongs in `kompend.cli`

- Typer command definitions (`@app.command`)
- `Annotated` argument/option declarations (`args/` submodule)
- Input validation at the string level (file exists? flag parsing?)
- Output formatting (print, `--json`, progress bars, `typer.Exit`)
- CLI-specific orchestration (resolve files, download URLs, copy to sources dir)
- Thin wrapper helpers that delegate to core with a console print after

### Lazy command registration

Command modules are NOT imported eagerly at startup. `_registry.py` declares each
command exactly once — `CommandName`, the `CommandModule` registering it, and its
`HelpPanel` — and `_group.LazyGroup` imports a module only when one of its commands
is used. All three are `StrEnum` members: a bare string is the defect the config
schema bans, and help lists every command from the registry at no import cost.

**Why:** eager imports pulled heavy deps (pydantic-ai, parsecraft, MCP SDK)
into every `kompend` invocation. With lazy registration, `kompend --version` takes
~1.5s instead of ~23s.

**Adding a new command:**

1. Add the name to `CommandName`, its module to `CommandModule` if new, and one
   `CommandSpec` entry to `COMMANDS` in `_registry.py`
2. Add the module to `_MODULE_NAMES` and `__all__` in `cmds/__init__.py`
3. Tests: `conftest.py` calls `register_all_commands()` at session start
4. No other changes needed — the module is imported on first invocation

### What does NOT belong in `kompend.cli`

| Prohibited | Belongs in |
|---|---|
| Business logic (parsing, chunking, compiling, linting) | `kompend.core`, `kompend.lint`, `kompend.compilation` |
| Data models and constants | `kompend.core.models` |
| Config defaults and resolution logic | `kompend.core.config` |
| LLM calls or LLM client construction | `kompend.llm` |
| Parser registry / router construction | `kompend.preprocessing.parsers` |
| Hard-coded magic numbers (chunk sizes, thresholds) | `kompend.core.config` DEFAULTS |

### The test

If you can remove `src/kompend/cli/` and `import kompend` still works for
programmatic use (pipeline, models, config, lint), then the boundary is clean.

### File responsibilities

| File/Directory | Purpose |
|---|---|
| `app.py` | Typer app instance, global callback (`--wiki-dir`, `--config`, `--verbose`, `--version`), command registration |
| `_registry.py` | The command registry: `CommandName` / `CommandModule` / `HelpPanel` StrEnums, `COMMANDS`, `PANEL_ORDER` |
| `_group.py` | `LazyGroup` — lists every command from the registry, imports each module on first use |
| `args/` | Argument type definitions organized by domain (see `args/AGENTS.md`) |
| `cmds/` | Command implementations organized by domain (see `cmds/AGENTS.md`) |
| `_sources.py` | File resolution, glob expansion, copy-to-sources, URL download. |
| `_ignore.py` | Ignore/gitignore pattern matching for source scanning. |
| `_pipeline_helpers.py` | Pipeline orchestration helpers (progress callback, input resolution, review queue, provenance, lint-fix persistence). |
| `_distill_helpers.py` | Distill wiring shared by `kompend distill` and `kompend compile --distill` (pending-note count/notice, agent selection, result lines). |
| `_esc_watch.py` | Double-ESC cancel watcher — polls stdin (msvcrt/termios), two ESC within `CANCEL_WINDOW_SECONDS` requests the cancel token. |
| `_progress.py` | Rich progress bar wrappers; owns the watcher lifecycle (`cancel=` token in, watcher started with the bar, stopped on finish/stop). |
| `_shared.py` | Shared utilities: config resolution + `config_file_from_ctx`, CLI overrides, hook loading. |

### Argument declarations (`args/` submodule)

See `src/kompend/cli/args/AGENTS.md` for detailed argument organization rules.

**Summary:**

- All `typer.Option` / `typer.Argument` live in `args/` as `Annotated` types.
- Arguments organized by domain into 9 modules: `common.py`, `compile.py`, `parse.py`, `query.py`, `pipeline.py`, `config.py`, `init.py`, `memory.py`, `other.py`.
- Commands import the types, never call `typer.Option()` inline (B008).
- Default values are module-level constants (`DEFAULT_CHUNK_SIZE`, etc.).
- **These constants are the CLI's own defaults, not the source of truth.** The
  source of truth is `kompend.core.config.DEFAULTS`. The CLI constants exist only
  for Typer's `--help` text and as fallback values passed to `resolve_config()`.

### Command functions (`cmds/` submodule)

See `src/kompend/cli/cmds/AGENTS.md` for detailed command organization rules.

**Summary:**

- One file per command group organized by domain. Groups are cohesive sets of related commands.
- Each command function is thin: parse args → resolve config → call core → print.
- Never put a `for` loop that iterates over wiki nodes and does real work.
  That's business logic — move it to a core module.
- `try/except` at the top level is fine for user-facing error messages.
- Use `get_console().print(msg)` for stdout output, `get_err_console().print(msg)` for stderr. Import from `kompend.logging`.
- `--json` flag: serialize and print. No other behavior change.
- `--verbose` / `KOMP_VERBOSE` gates tracebacks globally via
  `kompend.core.verbosity`; the recoverable-log rule that accompanies it lives in
  `core/pipeline/AGENTS.md`.

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
are set-once vault properties. Change them with `kompend config set <key> <val>`
(persistent) or override one run with `--set <key>=<val>` (ephemeral). Never
re-add a flag for these — use the `--set` escape hatch instead.

`--set` is parsed by `parse_cli_set()` (in `cmds/pipeline.py`): each `key=value`
is validated via `validate_config_key()` and coerced via `coerce_config_value()`,
then merged last by `build_cli_overrides(cli_set=...)` so it wins over every
other flag.

### Vault config discovery

`kompend --wiki-dir <vault> ...` reads *that* vault's config (`.kompend/config.toml`),
not the cwd's. See `core/config/AGENTS.md`: Project config discovery. Seed a
vault with a chosen config at creation time via `kompend init <dir> --config <toml>`
(the file is copied verbatim into the vault).

### Fail-fast vault validation

Every CLI/MCP entry point resolves paths through `vault_paths()`
(`core/paths.py`), which **raises LookupError** when the resolved directory has
no vault marker (`kompend.toml` / `.kompend/config.toml`). Silent operation on
arbitrary directories masks misconfiguration (config, vectors, and LLM would
all degrade invisibly). Only config-introspection commands (`kompend config
get/list/path/check/set`) opt out via `require_vault=False` — they read
defaults, not vault data.

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
   single VLM/LLM call that hangs until timeout (a parser or model looping
   on a degenerate token) will pin the bar at 0% for the full timeout
   duration × retry count.

Do NOT "fix" a bar that passes steps 1–2. Investigate the backend call
that isn't completing. The fix lives in `src/kompend/llm/`, the resolver,
or the model config (`timeout`, `fallback_chain`), not in
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
summary, resolve, image-captioning) fill as units complete. Bars without one
(compile, output) are indeterminate and convey activity via description
updates, not percentage. Never fake a percentage on an indeterminate task.

### Cooperative cancel (double-ESC)

`compile`/`ingest` create a `core.cancel.CancelState` and thread it into
`build_pipeline` / `ingest_documents` and `PipelineProgress.create`; with a
bar shown, `start()` runs the `_esc_watch.py` watcher (two ESC within
`CANCEL_WINDOW_SECONDS` ⇒ cancel request; no bar ⇒ no watcher). Boundary
contract: `core/pipeline/AGENTS.md`; `app.py` exits 130 with a resume hint.

### No direct core imports in thin wrappers

Helper functions like `_send_to_review_queue`, `_annotate_provenance`,
`_verify_claims` are acceptable as thin wrappers — they import the core
function, call it, and print a status line. The test: if you strip the
`get_console().print()` line, the remaining code should be a single function call
to a core module.

## Known violations (do not replicate)

These patterns remain in the codebase — refactor when touched, don't extend:

| Violation | Where | Why it's wrong |
|---|---|---|
| `status.py` imports from `kompend.cli._sources` | `status.py:10` | Cross-imports within CLI are fine, but extension enumeration should come from parser registry |

Fixed violations (router/wiki/vector-index construction, config path,
extension set) are archived in `.agents/history/2026-10-23-rationale-archive.md`.

## Work Guidance

- Never expand the CLI's role beyond input/output formatting and thin delegation.
- When adding a new command: add `Annotated` types to `args/` submodule, add the
  `@app.command` function to the appropriate `cmds/` module file (create one
  if no cohesive group exists), add argument → config mapping in
  `_build_cli_overrides()`.
- Use `get_console()` / `get_err_console()` from `kompend.logging` for all output — not `typer.echo()` or `print()`.

## Verification

- `uv run python -m kompend.cli --help` — app starts and shows commands
- `uv run pytest tests/unit/ -k "cli" -x --timeout=30` — CLI tests pass
- `import kompend` works without importing `kompend.cli`

---

## Child DOX Index

| Path | Scope |
|------|-------|
| `src/kompend/cli/args/AGENTS.md` | Argument definitions — Annotated types, domain organization, reusable type patterns |
| `src/kompend/cli/cmds/AGENTS.md` | Command implementations — @app.command registration, thin wrapper patterns, sub-apps (config, init, memory) |
