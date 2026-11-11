# DOX — kompend.cli.args

<!-- on entry: loaded when agent enters src/kompend/cli/args -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Centralize all CLI argument definitions (`Annotated` type aliases) into domain-organized modules for improved discoverability and maintainability.

Every Cyclopts option, argument, and flag definition lives here — never inline in a command function.

## Ownership

- `src/kompend/cli/args/` — All `Annotated` types for CLI options, arguments, and flags
- No business logic; no imports of core modules except for type hints

## Local Contracts

### File organization

Arguments organized by domain/command group:

| Module | Purpose | Commands |
|--------|---------|----------|
| `common.py` | Cross-command arguments | used by multiple command groups |
| `compile.py` | Compile command arguments | `/kompend compile` incl. LLM failover options |
| `parse.py` | Parse command arguments | `/kompend parse` |
| `query.py` | Query command arguments | `/kompend query` and RAG operations |
| `pipeline.py` | Pipeline/tiered compilation arguments | `/kompend pipeline`, sources export |
| `knowledge.py` | Knowledge command arguments | attachments, references, capture, pack, organize, agents |
| `quality.py` | Quality command arguments | lint, review, provenance, doctor, status |
| `serving.py` | Serving command arguments | serve, mcp, agent |
| `config.py` | Config subcommand arguments | `/kompend config` sub-app |
| `init.py` | Init subcommand arguments | `/kompend init` sub-app |
| `gitignore.py` | Vault `.gitignore` command arguments | `/kompend gitignore` |
| `memory.py` | Memory command group arguments | `/kompend memory` sub-app |

### Naming conventions

- All type aliases are `PascalCase` with semantic names: `CompileFiles`, `OutputFormat`, `Verbose`
- Default values are module-level `UPPERCASE_CONSTANTS`
- Annotated types reference these constants by name
- No magic literals; all defaults in constants

### Import rules

- Import only `cyclopts` and `Annotated` from external packages
- Never import core modules (e.g., `kompend.core`, `kompend.lint`, `kompend.llm`)
- Cross-module imports within args/ are acceptable (e.g., `from kompend.cli.args.common import Verbose`)
- Command modules import from this package: `from kompend.cli.args import QueryText, ConfigPath`

### Type alias pattern

All argument definitions follow this pattern:

```python
from typing import Annotated
from cyclopts import Parameter

DEFAULT_CHUNK_SIZE = 512

ChunkSize = Annotated[
    int,
    Parameter(
        name="--chunk-size",
        help="Size in tokens for document chunks.",
    ),
]
```

- **Annotated** wraps the Python type (int, str, bool, etc.)
- **Parameter** specifies CLI behaviour (`name=`, `help=`, `env_var=`, `group=`, `validator=`)
- **Default value** is the command parameter's default, never a literal in the alias
- **Help text** is mandatory and describes what the flag does

Cyclopts has no `typer.Option(default, ...)`: the default lives on the command
function. Aliases only describe the *shape* of the flag.

### Positional vs keyword-only

Cyclopts infers positionality from the signature, not the alias:

- A command that takes a value is positional (`ParseFiles`, `MemoryQuery`, `PackName`).
- Everything declared with a `--name` (or `-x`) in `Parameter(name=...)` must sit
  **after `*`** in the signature, or Cyclopts will bind it positionally:

```python
def parse_cmd(files: ParseFiles, *, recursive: RecursiveScan = None) -> None: ...
```

### Bool flag pattern

Two flavors, chosen by whether the flag has a config default:

**Config-backed feature toggle → tri-state `bool | None` with `--flag/--no-flag`:**

```python
Verify = Annotated[
    bool | None,
    Parameter(
        name="--verify",
        help="... Default: on; --no-verify skips.",
    ),
]
```

- Name the alias with the **positive** spelling; Cyclopts generates the
  `--no-<name>` twin for `bool` and `bool | None`.
- Default `None` means "not passed" — the resolved value comes from
  `kompend.core.config.DEFAULTS`, never from the flag.
- Both directions override config: `--verify` → True, `--no-verify` → False.
- `build_cli_overrides` (in `_shared.py`) uses `if val is not None:` so a
  `False` from `--no-X` is honored, not swallowed.
- Exception: negative-idiomatic flags keep the negative form primary —
  `--no-privacy/--privacy` (declare `name="--privacy"`).

**One-shot action → plain `bool` with `--flag` (no `--no-*`):**

`verbose`, `json`, `fix`, `dry-run`, `open`, `git`, `bare`, `list-tiers`,
`elements`, `raw`, `adaptive`, `remember`. These are off by default and
  represent a single action this run, not a persistent toggle. Declare
  `negative=""` on the `Parameter` to suppress the generated `--no-*` twin
  (Cyclopts otherwise adds it for every bool, including list-valued options,
  which would surface as a meaningless `--empty-<name>`).

### Reuse via re-export

`__init__.py` re-exports all type aliases with `from module import X as X` for:

- Backward compatibility (old imports continue working)
- Central discoverability (one place to see all argument types)
- Easier IDE navigation

## Known patterns

### Common arguments (shared across multiple commands)

These live in `common.py` and are imported by multiple command modules:

- `WikiDir` — Project root directory
- `RecursiveScan` — Tri-state `--recursive/--no-recursive` for source discovery (`parse`, `compile`, `ingest`, `write`); `None` follows the `paths.recursive` config value
- `ConfigPath` — Path to config file
- `Verbose` — Enable verbose output
- `JsonOutput` — Serialize output as JSON
- `Progress` — Show progress bar

### Help panels

Options rendered into named panels in `--help` via `group=`.
Canonical panel names (compile command):

- `Project` — wiki dir, config path, `--set` overrides
- `Sources` — URLs, ignore patterns, in-place source handling
- `Pipeline` — tier entry, cache, workers, force, progress, memory
- `Output` — synthesize, index/embed, lint, review, verify, provenance, privacy
- `LLM` — model selection and failover chain

Options shared with other commands keep the same panel; a panel renders only
when the command actually has options in it.

### Domain-specific argument clusters

**Compile arguments** (`compile.py`):

- `OnlineUrls` — Fetch and embed external URLs
- `UrlFile` — File with URLs to download
- `Ignore` — Glob ignore patterns
- `CompileDistill` — Distill pending notes before compiling
- `EmbedModel` — Embedding model name
- `SynthesizeFlag` / `IndexStore` / `CompileLint` — output toggles

**Query arguments** (`query.py`):

- `QueryText` — Search text
- `QueryTopK` — Number of results
- `QueryFormatOption` — Output format
- `QuerySearchMode` — Vector, BM25, or hybrid search

**Pipeline arguments** (`pipeline.py`):

- `StartAtTier` — Which compilation tier to begin at
- `Workers` — Number of parallel workers
- `ReparseFlag` — Re-parse sources instead of reusing the cached parse
- `JsonProgress` — Machine-readable progress output

## Verification

- All `.py` files in `args/` import successfully: `uv run python -c "from kompend.cli import args; print(dir(args))"`
- No unused type aliases in `__init__.py`
- No circular imports: `uv run python -c "import kompend.cli.args"`
- All exported names match `__all__`: `uv run python -c "from kompend.cli.args import *"`
