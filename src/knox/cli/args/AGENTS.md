# DOX — knox.cli.args

## Purpose

Centralize all CLI argument definitions (`Annotated` type aliases) into domain-organized modules for improved discoverability and maintainability.

Every Typer option, argument, and flag definition lives here — never inline in a command function.

## Ownership

- `src/knox/cli/args/` — All `Annotated` types for CLI options, arguments, and flags
- No business logic; no imports of core modules except for type hints

## Local Contracts

### File organization

Arguments organized by domain/command group:

| Module | Purpose | Commands |
|--------|---------|----------|
| `common.py` | Cross-command arguments | used by multiple command groups |
| `compile.py` | Compile command arguments | `/knox compile` and tiered compilation |
| `parse.py` | Parse command arguments | `/knox parse` |
| `query.py` | Query command arguments | `/knox query` and RAG operations |
| `pipeline.py` | Pipeline/tiered compilation arguments | `/knox pipeline` |
| `config.py` | Config subcommand arguments | `/knox config` sub-app |
| `init.py` | Init subcommand arguments | `/knox init` sub-app |
| `memory.py` | Memory command group arguments | `/knox memory` sub-app |
| `other.py` | Remaining commands | `/knox lint`, `/knox review`, `/knox serve`, `/knox organize`, `/knox pack`, `/knox quality`, `/knox capture`, `/knox doctor`, `/knox status` |

### Naming conventions

- All type aliases are `PascalCase` with semantic names: `CompileFiles`, `OutputFormat`, `Verbose`
- Default values are module-level `UPPERCASE_CONSTANTS`
- Annotated types reference these constants by name
- No magic literals; all defaults in constants

### Import rules

- Import only `typer` and `Annotated` from external packages
- Never import core modules (e.g., `knox.core`, `knox.lint`, `knox.llm`)
- Cross-module imports within args/ are acceptable (e.g., `from knox.cli.args.common import Verbose`)
- Command modules import from this package: `from knox.cli.args import QueryText, ConfigPath`

### Type alias pattern

All argument definitions follow this pattern:

```python
from typing import Annotated
import typer

DEFAULT_CHUNK_SIZE = 512

ChunkSize = Annotated[
    int,
    typer.Option(
        DEFAULT_CHUNK_SIZE,
        "--chunk-size",
        help="Size in tokens for document chunks.",
    ),
]
```

- **Annotated** wraps the Python type (int, str, bool, etc.)
- **Option/Argument** specifies CLI behavior
- **Default value** is a module constant, never a literal
- **Help text** is mandatory and describes what the flag does

### Reuse via re-export

`__init__.py` re-exports all type aliases with `from module import X as X` for:
- Backward compatibility (old imports continue working)
- Central discoverability (one place to see all argument types)
- Easier IDE navigation

## Known patterns

### Common arguments (shared across multiple commands)

These live in `common.py` and are imported by multiple command modules:

- `WikiDir` — Project root directory
- `Recursive` — Process subdirectories recursively
- `ConfigPath` — Path to config file
- `Verbose` — Enable verbose output
- `JsonOutput` — Serialize output as JSON
- `Progress` — Show progress bar

### Domain-specific argument clusters

**Compile arguments** (`compile.py`):
- `CompileFiles` — Input files/directories
- `TargetFormat` — Output format (md, json, yaml)
- `ImageMode` — Embed, extract, or reference images
- `OnlineUrls` — Fetch and embed external URLs
- `EmbedModel` — Embedding model name
- `EmbedBatchSize` — Batch size for embedding

**Query arguments** (`query.py`):
- `QueryText` — Search text
- `QueryTopK` — Number of results
- `QueryFormat` — Output format
- `QuerySearchMode` — Vector, BM25, or hybrid search

**Pipeline arguments** (`pipeline.py`):
- `StartAtTier` — Which compilation tier to begin at
- `Workers` — Number of parallel workers
- `NoCacheFlag` — Skip cache and recompile

## Verification

- All `.py` files in `args/` import successfully: `uv run python -c "from knox.cli import args; print(dir(args))"`
- No unused type aliases in `__init__.py`
- No circular imports: `uv run python -c "import knox.cli.args"`
- All exported names match `__all__`: `uv run python -c "from knox.cli.args import *"`

## Child DOX Index

*This module has no child AGENTS.md files.*
