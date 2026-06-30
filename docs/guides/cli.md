# CLI Reference

KNOX provides a Typer-based CLI with the following commands:

| Command | Description |
|---------|-------------|
| `knox init` | Scaffold a new wiki project |
| `knox config` | Inspect and manage configuration |
| `knox parse` | Parse documents and show their structure |
| `knox compile` | Compile documents into a knowledge wiki |
| `knox query` | Query the compiled wiki with natural language |
| `knox doctor` | Validate configuration and connectivity |
| `knox status` | Show per-wiki statistics |
| `knox list-parsers` | List all registered parsers |
| `knox lint run` | Run quality linters on the compiled wiki |
| `knox lint history` | Show past lint runs and score trends |
| `knox lint prune` | Remove old lint history entries |
| `knox review` | Manage the review queue |
| `knox provenance` | Show provenance for a wiki page |
| `knox capture` | Extract knowledge items from free text |
| `knox learn` | Store a fact verbatim as a source |
| `knox crystallize` | Synthesize captured notes into wiki pages |
| `knox write` | Write or overwrite a wiki page from source |
| `knox pack-list` | List available content packs |
| `knox pack-install` | Install a content pack |
| `knox mcp` | Run the MCP server |
| `knox serve` | Serve the web UI |

## Global Options

| Flag | Description |
|------|-------------|
| `--version`, `-v` | Show version and exit |
| `--wiki-dir`, `-w` | Set project root — `sources/`, `wiki/`, `.knox/` resolved from here |
| `--help` | Show help message and exit |

`--wiki-dir` can also be set via the `KNOX_WIKI_DIR` environment variable.

## `knox init`

Scaffold a new wiki project with the standard directory structure.

```text
knox init [OPTIONS] [DIR]
```

**Arguments:**

| Name | Description |
|------|-------------|
| `DIR` | Target directory (default: current directory) |

**Options:**

| Flag | Description |
|------|-------------|
| `--bare` | Skip authored templates (`NOTES.md`, `README.md`) |
| `--force` | Overwrite existing project config |
| `--git` | Initialize a git repo with an initial commit |

Creates the following structure:

```text
<dir>/
├── sources/           # Source documents (PDF, DOCX, ...)
├── wiki/              # Compiled wiki output
├── .knox/
│   ├── config.toml    # Project configuration
│   └── cache/         # Runtime cache
├── NOTES.md           # Free-form notes (injected into LLM templates)
├── README.md          # Project readme
└── .gitignore         # Ignores wiki/, .knox/cache/, state files
```

## `knox config`

Inspect and manage configuration files.

```text
knox config COMMAND [OPTIONS]
```

**Subcommands:**

| Command | Description |
|------|-------------|
| `get <key>` | Print resolved value for a dotted config key (e.g. `compile.format`) |
| `set <key> <value>` | Write a key to project config (or `--global` for global config) |
| `list` | Print all resolved config keys with provenance |
| `path` | Print the config file path (or `--global`) |

`set` validates keys against the schema, coerces types, writes atomically, and creates a `.bak` backup.

## `knox parse`

Parse documents and display their structural elements.

```text
knox parse [OPTIONS] FILES...
```

**Arguments:**

| Name | Description |
|------|-------------|
| `FILES...` | Files, glob patterns, or directories to parse (required) |

**Options:**

| Flag | Description |
|------|-------------|
| `--recursive/--no-recursive`, `-r/-R` | Search source directories recursively |
| `--elements`, `-e` | Print extracted elements |

## `knox compile`

Compile documents into a knowledge wiki.

```text
knox compile [OPTIONS] FILES...
```

**Arguments:**

| Name | Description |
|------|-------------|
| `FILES...` | Files, glob patterns, or directories to compile |

**Options:**

| Flag | Default | Description |
|------|---------|------------|
| Output shape (`--format`, `--image-mode`, `--flavor`, `--compiler`, `--kind`) and tuning values (budgets, timeouts, decay params) are **config-only** — set them with `knox config set compile.<key> <val>` or override per-run with `--set compile.<key>=<val>`. |  |  |
| `--index/--no-index` | on | Index compiled wiki in zvec for semantic search |
| `--embed-model` | — | Embedding model for zvec indexing |
| `--vlm-model` | — | Vision model for captioning images during resolution |
| `--verify/--no-verify` | on | Run grounding verification on each claim |
| `--synthesize/--no-synthesize` | on | Generate LLM-synthesized overview.md |
| `--ignore` | — | Ignore pattern (glob-style, repeatable) |
| `--progress/--no-progress` | auto | Show rich progress bars (auto-detected for TTY) |

Source and output directories are resolved from `--wiki-dir` (or `KNOX_WIKI_DIR`), defaulting to the current directory. See [Configuration](./configuration.md) for details.

### Ignore patterns

Files matching ignore patterns are excluded from compilation. Patterns use
`.gitignore`-style glob syntax:

```bash
knox compile *.pdf --ignore "*.bak" --ignore "drafts/**"
```

Ignore patterns can also be configured in `.knox/config.toml`:

```toml
[compile]
ignore = ["*.tmp", "drafts/*", "**/private/*.pdf"]
```

By default, `.gitignore` in the source directory is respected. Set
`ignore_gitignore: false` in config to disable.

### Progress bars

When stdout is a TTY, both `knox ingest` and `knox compile` show rich progress
bars:

- **Across-file**: `Parsing sources [45/50] ████████████████░░░░ 90% 0:02:34`
- **Within-file** (compile only): Pipeline steps per file (Download, Parse,
  Chunk, Resolve, Compile, Write)

When dedup is enabled, `knox ingest` runs two progress phases — a parse pass
followed by a separate embedding pass — so GPU-bound parsers and the embedding
model don't compete for GPU memory.

Use `--no-progress` to disable, or `--progress` to force enable.

## `knox doctor`

Validate configuration and connectivity.

```text
knox doctor [OPTIONS]
```

**Options:**

| Flag | Description |
|------|-------------|
| `--verbose` | Show detailed output (including LLM connectivity check) |
| `--json` | Output in JSON format for machine readability |

Checks performed:

1. **Source directory** — exists and is a directory
2. **Output directory** — exists and is writable
3. **Parsers** — all registered parser dependencies load correctly
4. **Git repository** — check if inside a git repo
5. **Disk space** — warn if < 500 MB free
6. **Environment variables** — KNOX_LLM_MODEL, KNOX_LLM_PROVIDER, OPENAI_API_KEY
7. **LLM connectivity** — quick ping to configured LLM (only with `--verbose`)

Exit code: `0` if all checks pass, `1` if any check fails.

## `knox status`

Show per-wiki statistics.

```text
knox status [OPTIONS]
```

**Options:**

| Flag | Description |
|------|-------------|
| `--json` | Output in JSON format |

Displays:

- Wiki name and config file path
- Source count and total size
- Article count and total word count
- Last compile timestamp
- Health score (if eval is set up)

## `knox list-parsers`

List all registered parsers (built-in, external, and HTTP).

```text
knox list-parsers [OPTIONS]
```

**Options:**

| Flag | Description |
|------|-------------|
| `--json` | Output in JSON format |

## `knox query`

Ask a natural-language question against the compiled wiki.

```text
knox query [OPTIONS] QUESTION
```

`knox query` loads the cached wiki and uses hybrid search (FTS + vector + token overlap + raw sources fused with reciprocal rank fusion) when an index has been built with `knox compile --index`.

| Flag | Default | Description |
|------|---------|-------------|
| `--top-k` | `5` | Number of context chunks to retrieve |
| `--scope` | `hybrid` | Search scope: `wiki`, `raw`, or `hybrid` |
| `--format` | `text` | Output format: `text` or `json` |
| `--raw` | — | Return raw context chunks without LLM synthesis |
| `--no-classify` | — | Skip intent classification |

## `knox capture` and `knox learn`

Quickly persist notes, facts, or conversation snippets as source files in `sources/_captured/`.

```bash
# LLM extraction into one or more knowledge items
knox capture "We decided to use QKV projections with RoPE"

# Store a fact verbatim
knox learn "The transformer uses d_model=512 and h=8" --tag attention
```

## `knox crystallize`

Synthesize captured notes from `sources/_captured/` into coherent wiki pages. Notes are grouped by tag and merged (LLM-powered or deterministic fallback). Output goes to `sources/_crystallized/`.

```bash
# Merge all captured notes into synthesized pages
knox crystallize

# Then compile to incorporate crystallized pages
knox compile
```

## `knox organize`

Re-categorize the files in `sources/` into a hierarchical directory tree
generated by the LLM. Requires a prior `knox compile` so per-source summaries
are available in `.knox/cache/extract/`. The LLM is given the summaries and
returns a JSON category tree which is then validated and applied.

```bash
# Preview the proposed tree without moving files
knox organize --dry-run

# Cap the tree at 2 directory levels, require ≥3 sub-categories
knox organize --max-depth 2 --min-subcategories 3

# Disable single-file collapse (keep deep sub-trees with one file each)
knox organize --no-collapse-single-file
```

When applied, `knox organize` first flattens nested files in `sources/` (with
`-2`, `-3` suffixing on collisions), materializes the new directory tree,
moves each file into its assigned category, and rewrites wiki links that
reference the old paths. Updated locations:

- Inline `^[filename:start-end]` citations in `wiki/**/*.md`
- `.knox/provenance/*.json` per-node citation records
- `metadata.source_file` in `.knox/last_wiki.json`
- `source_path` in `.knox/compile_state.json`

A report is written to `.knox/last_organize.json`. Files without an
extraction summary are left in place at the top level.

## Environment Variables

All environment variables use the `KNOX_` prefix and map to dotted config keys.

| Variable | Config Key | Description |
|----------|------------|-------------|
| `KNOX_WIKI_DIR` | `paths.wiki_dir` | Project root directory |
| `KNOX_SOURCES_DIR` | `paths.sources_dir` | Sources directory (relative to wiki_dir) |
| `KNOX_OUTPUT_DIR` | `paths.output_dir` | Output directory (relative to wiki_dir) |
| `KNOX_RECURSIVE` | `compile.recursive` | Default recursive mode (true/false) |
| `KNOX_LLM_MODEL` | `compile.llm.default_model` | Default LLM model |
| `KNOX_VLM_MODEL` | `compile.llm.vlm_model` | Vision model for image captioning |
| `KNOX_NOTES_FILE` | `notes.file` | User notes filename |
| `KNOX_RESOLVE_RELATIONS` | `compile.resolve_relations` | Enable typed relation extraction |
| `KNOX_RELATION_CONFIDENCE_THRESHOLD` | `compile.relation_confidence_threshold` | Minimum confidence for relations |

## Event Hooks

KNOX fires hooks at key pipeline events. Register callbacks in `knox.toml` under `[hooks]` using `module:attribute` syntax:

```toml
[hooks]
post_compile = ["mypkg.hooks:auto_lint"]
on_capture = ["mypkg.hooks:notify"]
```

| Event | Fired by | Context data |
|-------|----------|-------------|
| `pre_compile` | `compile` | `source_dir`, `output_dir`, `mode` |
| `post_compile` | `compile` | `wiki`, `output_dir`, `node_count` |
| `on_capture` | `capture`, `learn` | `paths`, `sources_dir` |
| `on_query` | `query` | `query`, `results`, `scope` |
| `on_lint` | `lint` | `report`, `wiki` |

Callbacks receive a `HookContext` with `.event` and `.data`. Exceptions in callbacks are swallowed and logged so they never break the pipeline.
