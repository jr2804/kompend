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

- Cyclopts command definitions (`app.command(...)`, one per `CommandSpec`)
- `Annotated` argument/option declarations (`args/` submodule)
- Input validation at the string level (file exists? flag parsing?)
- Output formatting (print, `--json`, progress bars, `typer.Exit`)
- CLI-specific orchestration (resolve files, download URLs, copy to sources dir)
- Thin wrapper helpers that delegate to core with a console print after

### Lazy command registration

Command modules are NOT imported eagerly at startup. `tree/registry.py` declares
each command exactly once — `CommandName`, the `CommandModule` owning it, its
`HelpPanel`, and the `short_help` line `--help` shows — and `app.py` registers
each as `app.command("<module>:<entry>", ...)`, which Cyclopts resolves only when
that command runs. Every vocabulary is a `StrEnum` member: a bare string is the
defect the config schema bans. Help renders the full panel tree from
`CommandSpec`, without importing a command module.

The directory is a package per concern, not a flat pile of helpers:
`runtime/` (process), `tree/` (declaration), `sources/` (file IO), `workflow/`
(a pipeline run), `query/` (one query end to end), plus `args/` and `cmds/`.
A module that fits none of those is a sign the concern is new — name it.

**Why:** eager imports pulled heavy deps (pydantic-ai, parsecraft, MCP SDK)
into every `kompend` invocation. With lazy registration and registry-rendered
help, `--help` reads only `_registry.py`, and `--version` never loads the config
engine (pydantic, `pydantic_settings`). `telemetry.py` owns `import logfire`;
the callback configures it after the `--version` exit, and `kompend/__init__.py`
disables logfire's pydantic plugin (`PYDANTIC_DISABLE_PLUGINS`), which pydantic's
plugin loader would otherwise import when the first model class is defined. The
run log loads on the first real command via `OnDemandModule.RUNLOG`.

**Adding a new command:**

1. Add the name to `CommandName`, its module to `CommandModule` if new, and one
   `CommandSpec` entry to `COMMANDS` in `tree/registry.py`
2. Add the module to `_MODULE_NAMES` and `__all__` in `cmds/__init__.py`
3. Tests: `conftest.py` calls `register_all_commands()` at session start; nothing
   else changes - the module imports on first invocation

### What does NOT belong in `kompend.cli`

| Prohibited | Belongs in |
|---|---|
| Business logic (parsing, chunking, compiling, linting) | `kompend.core`, `kompend.lint`, `kompend.compilation` |
| Data models and constants | `kompend.core.models` |
| Config defaults and resolution logic | `kompend.core.config` |
| LLM calls or LLM client construction | `kompend.llm` |
| Parser construction | `kompend.preprocessing.parsers` |
| Hard-coded magic numbers (chunk sizes, thresholds) | `kompend.core.config` DEFAULTS |

### The test

Removing `src/kompend/cli/` must leave `import kompend` working for programmatic
use; then the boundary is clean.

### File responsibilities

| File/Directory | Purpose |
|---|---|
| `app.py` | Typer app instance, global callback (`--wiki-dir`, `--config`, `--verbose`, `--version`), command registration |
| `launch.py` | Console entry point: bounds the numeric thread pools, makes the console decode UTF-8, answers a bare `--version` before the command tree loads, installs the console proxy + failure hooks + fault handler, delegates everything else to `app.main` |
| `context.py` | `RunContext` + its `ContextVar`: the `--wiki-dir` / `--config` a run was started with |
| `settings.py` | What a command derives from that context: `vault_paths_from_ctx`, `config_file_from_ctx`, `build_cli_overrides`, `resolve_lint_output` |
| `errors.py` | `usage_error` (exit 2) and `format_exception_for_cli` (traceback only under `--verbose`) |
| `runtime/thread_pools.py` | `cap_thread_pools()` — `KOMP_BLAS_THREADS`, default `min(8, cores)`, applied before any native library is imported |
| `runtime/console.py` | The console boundary: `HardenedStream` (a handle that dies mid-run cannot kill the run), `ensure_utf8_code_page()`. See the stream contract |
| `runtime/faults.py` | Never-silent reporting: `faulthandler` to `%TEMP%\kompend-faults.log`, `threading.excepthook` / `sys.unraisablehook` routed through the log |
| `runtime/progress.py` | Rich progress bar wrappers; owns the watcher lifecycle (`cancel=` token in, watcher started with the bar, stopped on finish/stop) |
| `runtime/esc_watch.py` | Double-ESC cancel watcher — polls stdin (msvcrt/termios), two ESC within `CANCEL_WINDOW_SECONDS` requests the cancel token |
| `tree/registry.py` | The command registry: `CommandName` / `CommandModule` / `HelpPanel` StrEnums, `COMMANDS` with each `CommandSpec` (module, entry, panel, `short_help`), `PANEL_ORDER`, `OnDemandModule` |
| `sources/resolve.py` | File resolution, glob expansion, copy-to-sources, URL download |
| `sources/io.py` | Reading and writing those source files (archives, encodings, atomic copies) |
| `workflow/pipeline.py` | Pipeline orchestration: progress callback, input resolution |
| `workflow/post_compile.py` | What runs after a compile: review queue, provenance, lint, vector index, overview |
| `workflow/distill.py` | Distill wiring shared by `kompend distill` and `kompend compile --distill` (pending-note count/notice, agent selection, result lines) |
| `query/search.py` | Retrieval for `kompend query`: vector store, keyword/semantic/adaptive search |
| `query/synthesize.py` | Unified query-session synthesis and query memory |
| `query/render.py` | Raw and structured rendering, wiki links, figure lines |
| `args/` | Argument type definitions organized by domain (see `args/AGENTS.md`) |
| `cmds/` | Command implementations organized by domain (see `cmds/AGENTS.md`) |

### Argument declarations (`args/` submodule)

See `src/kompend/cli/args/AGENTS.md` for detailed argument organization rules.

**Summary:**

- All `Parameter` declarations live in `args/` as `Annotated` types, one module
  per command group; commands import the types, never build `Parameter(...)`.
- **These constants are the CLI's own defaults, not the source of truth.**
  That is `kompend.core.config.DEFAULTS`; the CLI constants exist only for
  `--help` text and as fallback values passed to `resolve_config()`.

### Command functions (`cmds/` submodule)

See `src/kompend/cli/cmds/AGENTS.md` for detailed command organization rules.

**Summary:**

- One file per command group organized by domain.
- Each command function is thin: parse args → resolve config → call core → print.
  Never put a `for` loop that iterates over wiki nodes and does real work.
- `try/except` at the top level is fine for user-facing error messages.
- Use `get_console().print(msg)` for stdout output, `get_err_console().print(msg)` for stderr. Import from `kompend.logging`.
- `--json` flag: serialize and print. No other behavior change.
- `--verbose` / `KOMP_VERBOSE` gates tracebacks via
  `kompend.core.verbosity`, whose value is a `ContextVar` — cross-thread readers
  (the parse pool) need the submission wrapped in `copy_context().run(...)`,
  which `stages/parse.py` does. The recoverable-log rule that accompanies it
  lives in `core/pipeline/AGENTS.md`.

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
(persistent) or `--set <key>=<val>` (ephemeral). Never re-add a flag for these.

`--set` is parsed by `parse_cli_set()` (`cmds/pipeline.py`): each `key=value`
is validated via `validate_config_key()`, coerced via `coerce_config_value()`, then
merged last by `build_cli_overrides(cli_set=...)` so it wins over every flag.

### Vault config discovery

`kompend --wiki-dir <vault> ...` reads *that* vault's config
(`.kompend/config.toml`), not the cwd's. See `core/config/AGENTS.md`: Project
config discovery; `kompend init <dir> --seed-config <toml>` seeds one.

### Fail-fast vault validation

Every CLI/MCP entry point resolves paths through `vault_paths()`
(`core/paths.py`), which **raises LookupError** when the directory has no vault
marker (`kompend.toml` / `.kompend/config.toml`). Silent operation on arbitrary
directories masks misconfiguration; only `kompend config get/list/path/check/set`
opt out via `require_vault=False`.

### Progress bars

**Recurring trap:** a bar that looks "stuck" is usually slow/hung backend work,
not plumbing; diagnose the total and the elapsed timer, then fix the work.
**Visibility contract:** auto mode enables the bar when either stream is a TTY
and forces a terminal console when stderr is not (Windows/PowerShell).

**Stream contract:** the `Progress` bar uses `redirect_stdout=False`,
`redirect_stderr=False` — rich's `Live` swaps the std streams for proxies by
default and restores them unconditionally on stop, which races the refresh thread
into `OSError(9): The handle is invalid`. Logging goes through the console object,
so the swap buys nothing. `runtime/console.harden_console_streams()` proxies both streams
at entry for the other cause: a capture window rebinds descriptor 2 around a native
conversion, and the buffered stream bound to the old handle never recovers — every
later write raises, the refresh thread dies, and the run ends as CPython's silent
exit status 120. The proxy reopens a fresh stream over the current descriptor
(bounded by `_REPAIR_LIMIT`) and absorbs only a descriptor that is really gone;
any other `OSError` (full disk, a pipe a consumer reads) still propagates. Never
remove that wrapper to "simplify" output.

**Never-silent contract:** a dying thread, an unraisable exception and a native
abort must each leave a trace. `runtime/faults.py` installs the hooks and enables
`faulthandler` against `%TEMP%\kompend-faults.log`; the first failure per
(thread, exception) pair is reported at ERROR with its traceback, repeats at DEBUG.
Never report through `print`/`Console` — the console is exactly what fails.

**Signal-path contract:** a phase advances only when the core tier calls
`self._emit(phase, current, total, detail)`; the CLI handler in
`workflow/pipeline.make_progress_callback` maps that triple to
`tracker.reset_for_phase` / `tracker.update_progress`. A sub-phase (like the VLM
captioning burst inside resolve) emits a sentinel `(0, n, "<phase_name>")` then
`(i+1, n, "<detail>")` per unit; the handler must recognize both.

**Determinate vs indeterminate:** bars with a known `total` (parse, summary,
resolve, image-captioning) fill as units complete; the rest (compile, output)
are indeterminate and convey activity via description, never a faked percentage.

### Cooperative cancel (double-ESC)

`compile`/`ingest` create a `core.cancel.CancelState` and thread it into
`build_pipeline` / `ingest_documents` and `PipelineProgress.create`; with a bar
shown, `start()` runs the `runtime/esc_watch.py` watcher (no bar, no watcher).
Boundary contract: `core/pipeline/AGENTS.md`; `app.py` exits 130 with a hint.

### No direct core imports in thin wrappers

Thin wrappers (`send_to_review_queue`, `annotate_provenance`, `verify_claims`) are
acceptable: strip their `get_console().print()` line and one core call remains.

## Known violations (do not replicate)

Fixed violations (parser/wiki/vector-index construction, config path, extension
set) are archived in `.agents/history/2026-10-23-rationale-archive.md`.

## Work Guidance

- Never expand the CLI's role beyond input/output formatting and thin delegation.
- New command: `Annotated` types in `args/`, the `@app.command` function in the
  matching `cmds/` module, the flag → config mapping in `build_cli_overrides()`.
- Use `get_console()` / `get_err_console()` from `kompend.logging` for all output.

## Verification

- `uv run python -m kompend.cli --help` — app starts and shows commands
- `KOMP_LLM_TRANSIENT_BACKOFF=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/unit/ -k "cli" -x --no-header -p pytest_asyncio.plugin -p pytest_timeout` — CLI tests pass (T0/T1 tier)
- `import kompend` works without importing `kompend.cli`

## Child DOX Index

| Path | Scope |
|------|-------|
| `src/kompend/cli/args/AGENTS.md` | Argument definitions — Annotated types, domain organization, reusable type patterns |
| `src/kompend/cli/cmds/AGENTS.md` | Command implementations — @app.command registration, thin wrapper patterns, sub-apps (config, init, memory) |
