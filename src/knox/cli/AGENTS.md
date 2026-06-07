# AGENTS.md — knox.cli

Rules for AI coding assistants working on the KNOX CLI layer.

## Design principle

> **`knox` must be usable as a Python library without importing `knox.cli`.**
> The CLI is a thin shell over the core library. It parses user input, calls
> the right core function, and formats the output. Nothing more.

### What belongs in `knox.cli`

- Typer command definitions (`@app.command`)
- `Annotated` argument/option declarations (`args.py`)
- Input validation at the string level (file exists? flag parsing?)
- Output formatting (print, `--json`, progress bars, `typer.Exit`)
- CLI-specific orchestration (resolve files, download URLs, copy to sources dir)
- Thin wrapper helpers that delegate to core with a `typer.echo` after

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

If you can remove `src/knox/cli/` and `import knox` still works for programmatic
use (pipeline, models, config, lint), then the boundary is clean.

## File responsibilities

| File | Purpose |
|---|---|
| `app.py` | Typer app instance, version callback, command auto-import |
| `args.py` | `Annotated` type aliases for every CLI option/argument. No logic. |
| `commands.py` | `@app.command` functions. Parse args → call core → format output. |
| `_sources.py` | File resolution, glob expansion, copy-to-sources, URL download. |
| `_ignore.py` | Ignore/gitignore pattern matching for source scanning. |
| `_progress.py` | Rich progress bar wrappers. |
| `doctor.py` | Health checks (dirs, parsers, LLM connectivity). |
| `status.py` | Wiki statistics collection and display. |

## Conventions

### Argument declarations (`args.py`)

- All `typer.Option` / `typer.Argument` live in `args.py` as `Annotated` types.
- Commands import the types, never call `typer.Option()` inline (B008).
- Default values are module-level constants (`DEFAULT_CHUNK_SIZE`, etc.).
- **These constants are the CLI's own defaults, not the source of truth.** The
  source of truth is `knox.core.config.DEFAULTS`. The CLI constants exist only
  for Typer's `--help` text and as fallback values passed to `resolve_config()`.

### Command functions (`commands.py`)

- Keep them short: parse → resolve config → call core → print.
- Never put a `for` loop that iterates over wiki nodes and does real work.
  That's business logic — move it to a core module.
- `try/except` at the top level is fine for user-facing error messages.
- Use `typer.echo(msg, err=True)` for errors, `typer.echo(msg)` for output.
- `--json` flag: serialize and print. No other behavior change.

### CLI overrides → config

Every CLI flag that maps to a config key must:

1. Be listed in `_build_cli_overrides()` with the dotted config key.
2. The command function passes the override to `resolve_config()`.
3. The resolved value comes from `cfg.get("dotted.key")`, never from the raw
   CLI parameter directly (unless it's purely a CLI concept like `--json`).

### No direct core imports in thin wrappers

Helper functions like `_send_to_review_queue`, `_annotate_provenance`,
`_verify_claims` are acceptable as thin wrappers — they import the core
function, call it, and print a status line. The test: if you strip the
`typer.echo()` line, the remaining code should be a single function call
to a core module.

## Known violations (do not replicate)

These patterns exist in the current codebase and should be refactored, not
extended:

| Violation | Where | Why it's wrong |
|---|---|---|
| `_build_router()` constructs and registers parsers | `commands.py:807` | Parser registry is core business logic; CLI should call a factory from `knox.preprocessing` |
| `_load_wiki()` deserializes JSON into `CompiledWiki` | `commands.py:834` | Serialization is a core concern; should live in `knox.core` or `knox.output` |
| `_build_vector_index()` directly uses `ZvecWikiStore` | `commands.py:849` | Vector index building is RAG business logic; should be a core function |
| `SUPPORTED_EXTENSIONS` defined in `_sources.py` | `_sources.py:16` | Extension registry should come from the parser layer |
| `doctor.py` calls `litellm.completion()` directly | `doctor.py:92` | All LLM calls must go through `knox.llm.client.LLMClient` |
| `status.py` hardcodes config path `.knox/config.yaml` | `status.py:67` | Should use `knox.core.config.find_project_config()` |
| `_build_cli_overrides()` lives in `commands.py` | `commands.py:731` | Could move to a dedicated `_config.py` in the CLI package |

When touching any of these files, prefer moving the logic toward core rather
than adding more in the CLI.
