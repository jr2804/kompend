# DOX — kompend.cli.cmds

<!-- on entry: loaded when agent enters src/kompend/cli/cmds -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Centralize all CLI command definitions (`@app.command` functions) into domain-organized modules for improved discoverability and maintainability.

Every command entry point lives here — commands are the only way users invoke kompend. Organizing them by domain reduces cognitive load and makes it clear where to find implementation for a given feature.

## Ownership

- `src/kompend/cli/cmds/` — All `@app.command` and `@app.callback` function definitions
- Command group sub-apps (config, init, memory use `typer.Typer()` internally)

## Local Contracts

### File organization

Commands organized by domain/functional grouping. Top-level modules are
individual entry points; ``knowledge/``, ``quality/``, and ``pipeline/``
are subpackages:

| Module(s) | Purpose | Entry Point(s) |
|-----------|---------|-----------------|
| `agent.py` | Agent CLI (ACP interactive) | `/kompend agent` |
| `agents.py` | Agent management | `/kompend agents` |
| `init.py` | Project initialization | `/kompend init` (sub-app: bare, scaffold, bootstrap) |
| `serve.py` | MCP + web server | `/kompend mcp`, `/kompend serve` |
| `knowledge/` | Capture, query, memory, organize, pack, attachments, references | `/kompend capture`, `/kompend learn`, `/kompend distill`, `/kompend write`, `/kompend query`, `/kompend memory`, `/kompend organize`, `/kompend pack-list`, `/kompend pack-install`, `/kompend attachments`, `/kompend references` |
| `pipeline/` | Parse, compile, ingest, index, config, sources | `/kompend parse`, `/kompend compile`, `/kompend ingest`, `/kompend index`, `/kompend config`, `/kompend sources` |
| `quality/` | Doctor, lint, review, log, status, provenance, list-parsers | `/kompend doctor`, `/kompend lint`, `/kompend review`, `/kompend log`, `/kompend status`, `/kompend provenance`, `/kompend list-parsers` |

### Command function patterns

Every command follows the thin-wrapper pattern:

```python
from kompend.cli.app import app
from kompend.cli.args import CompileFiles, TargetFormat
from kompend.cli._shared import resolve_project_paths
from kompend.logging import get_console, get_err_console


@app.command()
def compile(
    files: CompileFiles,
    output_format: TargetFormat = TargetFormat.Markdown,
) -> None:
    """Compile sources into a wiki."""
    try:
        paths = resolve_project_paths(files)
        result = kompend.core.compile(paths, output_format)
        get_console().print(result)
    except Exception as e:
        get_err_console().print(f"Error: {e}")
        raise typer.Exit(1)
```

**Structure:**

1. Import `Annotated` types from `kompend.cli.args`
2. Call core/business logic once
3. Format and print output
4. Catch and report errors with `get_err_console().print(...)`
5. Never loop over wiki nodes or do data transformation — that's business logic

### Import rules

- Import command arguments from `kompend.cli.args`
- Import helpers from `kompend.cli._sources`, `kompend.cli._shared`, `kompend.cli._pipeline_helpers`
- Import the `app` instance from `kompend.cli.app`
- Import core business logic from `kompend.core`, `kompend.lint`, `kompend.llm`, etc. as needed
- Never define business logic in a command file — import and delegate

### Naming conventions

- Command functions are lowercase: `compile()`, `lint()`, `query()`
- Helper functions prefixed with `_` (private): `_build_review_queue()`, `_annotate_provenance()`
- Sub-app functions follow the same pattern (e.g., `pipeline/config.py` defines `config` and its sub-commands)

### Sub-apps (config, memory, attachments, references, sources, lint)

Each sub-app has its own Typer instance and set of sub-commands:

**pipeline/config.py:**

```python
config_app = typer.Typer()


@config_app.command()
def get(key: str) -> None: ...


@config_app.command()
def set(key: str, value: str) -> None: ...


@config_app.command()
def check(paths: list[Path]) -> None: ...


app.add_typer(config_app, name="config")
```

Same structure for `knowledge/memory.py`, `knowledge/attachments.py`, `knowledge/references.py`, `pipeline/sources.py`, and `quality/lint.py`.

## Known patterns

### CLI overrides → config mapping

Every CLI flag that affects config must be mapped in `_build_cli_overrides()` in `_shared.py`.

Example:

- Flag: `--llm-default-model` (from `args/other.py`)
- Config key: `llm.default_model`
- Resolved in command: `cfg = resolve_config(cli_overrides=...)` via `build_cli_overrides()` in `_shared.py`

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

- All modules import successfully: `uv run python -c "from kompend.cli.cmds import *"`
- No circular imports with `app.py`: `uv run python -c "from kompend.cli import app, cmds"`
- All commands register with app: `uv run python -m kompend.cli --help` shows all commands
- `__init__.py` uses `__getattr__` for lazy module loading — do NOT add eager imports; add new modules to `_LAZY_MODULES` in `app.py` and `_MODULE_NAMES` in `__init__.py`
