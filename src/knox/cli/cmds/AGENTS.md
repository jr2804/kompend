# DOX — knox.cli.cmds

<!-- on entry: loaded when agent enters src/knox/cli/cmds -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Centralize all CLI command definitions (`@app.command` functions) into domain-organized modules for improved discoverability and maintainability.

Every command entry point lives here — commands are the only way users invoke KNOX. Organizing them by domain reduces cognitive load and makes it clear where to find implementation for a given feature.

## Ownership

- `src/knox/cli/cmds/` — All `@app.command` and `@app.callback` function definitions
- Command group sub-apps (config, init, memory use `typer.Typer()` internally)

## Local Contracts

### File organization

Commands organized by domain/functional grouping. Top-level modules are
individual entry points; ``knowledge/`` and ``pipeline/`` are group sub-apps:

| Module(s) | Purpose | Entry Point(s) |
|-----------|---------|-----------------|
| `agent.py` | Agent CLI (ACP interactive) | `/knox agent` |
| `agents.py` | Agent management | `/knox agents` |
| `capture.py` | Document capture | `/knox capture` |
| `config.py` | Configuration management | `/knox config` (sub-app: get, set, validate, show) |
| `doctor.py` | Health/diagnostic checks | `/knox doctor` |
| `init.py` | Project initialization | `/knox init` (sub-app: bare, scaffold, bootstrap) |
| `lint.py` | Linting and validation | `/knox lint` (sub-app: run, history, prune) |
| `log.py` | Run log viewer | `/knox log` |
| `memory.py` | Memory/RAG operations | `/knox memory` (sub-app: query, add, feedback) |
| `organize.py` | Wiki organization | `/knox organize` |
| `pack.py` | Archive packing | `/knox pack` |
| `pipeline.py` | Tiered compilation pipeline | `/knox pipeline` |
| `quality.py` | Quality diagnostics | `/knox quality` |
| `query.py` | Vector search / RAG queries | `/knox query` |
| `review.py` | Review queue management | `/knox review` |
| `serve.py` | Local HTTP server | `/knox serve` |
| `sources.py` | Source management | `/knox sources` |
| `status.py` | Wiki statistics | `/knox status` |
| `knowledge/` | Knowledge sub-app | capture, learn, distill, memory, organize, pack, query |
| `pipeline/` | Pipeline sub-app | compile, config, sources, status |

### Command function patterns

Every command follows the thin-wrapper pattern:

```python
from knox.cli.app import app
from knox.cli.args import CompileFiles, TargetFormat
from knox.cli._shared import resolve_project_paths
from knox.logging import get_console, get_err_console


@app.command()
def compile(
    files: CompileFiles,
    output_format: TargetFormat = TargetFormat.Markdown,
) -> None:
    """Compile sources into a wiki."""
    try:
        paths = resolve_project_paths(files)
        result = knox.core.compile(paths, output_format)
        get_console().print(result)
    except Exception as e:
        get_err_console().print(f"Error: {e}")
        raise typer.Exit(1)
```

**Structure:**

1. Import `Annotated` types from `knox.cli.args`
2. Call core/business logic once
3. Format and print output
4. Catch and report errors with `get_err_console().print(...)`
5. Never loop over wiki nodes or do data transformation — that's business logic

### Import rules

- Import command arguments from `knox.cli.args`
- Import helpers from `knox.cli._sources`, `knox.cli._shared`, `knox.cli._pipeline_helpers`
- Import the `app` instance from `knox.cli.app`
- Import core business logic from `knox.core`, `knox.lint`, `knox.llm`, etc. as needed
- Never define business logic in a command file — import and delegate

### Naming conventions

- Command functions are lowercase: `compile()`, `lint()`, `query()`
- Helper functions prefixed with `_` (private): `_build_review_queue()`, `_annotate_provenance()`
- Sub-app functions follow the same pattern (e.g., `config.py` defines `config` and its sub-commands)

### Sub-apps (config, init, memory)

Each sub-app has its own Typer instance and set of sub-commands:

**config.py:**

```python
config_app = typer.Typer()


@config_app.command()
def get(key: str) -> None: ...


@config_app.command()
def set(key: str, value: str) -> None: ...


app.add_typer(config_app, name="config")
```

Same structure for init.py and memory.py — each is a thin wrapper with sub-commands.

## Known patterns

### CLI overrides → config mapping

Every CLI flag that affects config must be mapped in `_build_cli_overrides()` in `_shared.py`.

Example:

- Flag: `--llm-model` (from args.py)
- Config key: `llm.model`
- Resolved in command: `cfg = resolve_config(overrides={"llm.model": llm_model})`

### Error handling

- Catch user-facing errors and `get_err_console().print(...)` with actionable message
- Do NOT catch and silently ignore — propagate failures
- For expected errors (missing file), print user message then `raise typer.Exit(1)`
- For unexpected errors, include stack trace in logs, brief message to user

### JSON output

- Check for `--json` flag; if set, serialize output and print, no other behavior changes
- Use `json.dumps()` with default encoder
- Never mix human-readable and JSON output in same command

## Verification

- All modules import successfully: `uv run python -c "from knox.cli.cmds import *"`
- No circular imports with `app.py`: `uv run python -c "from knox.cli import app, cmds"`
- All commands register with app: `uv run python -m knox.cli --help` shows all commands
- Backward compatibility maintained: Old import paths (`from knox.cli.lint_cmds import lint`) still work via `__init__.py` re-exports
- `__init__.py` uses `__getattr__` for lazy module loading — do NOT add eager imports; add new modules to `_MODULE_NAMES` frozenset instead

## Child DOX Index

*This module has no child AGENTS.md files.*
