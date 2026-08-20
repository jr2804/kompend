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
| `knox lint gaps` | Analyze lint results for knowledge gaps and source recommendations |
| `knox lint remediate` | LLM-generate missing pages for lint findings |
| `knox lint prune` | Remove old lint history entries |
| `knox review` | Manage the review queue |
| `knox sources list` | List ingested source files with their tags (`--tags` filter) |
| `knox sources tag` | Add a tag to an ingested source file |
| `knox sources untag` | Remove a tag from an ingested source file |
| `knox sources export` | Render an ingested source as clean markdown |
| `knox trace [SLUG]` | List pages with slugs, or trace a page to its sources |
| `knox capture` | LLM-extract knowledge items from free text (`--tag`) |
| `knox learn` | Store a fact verbatim as a source, no LLM (`--tag`) |
| `knox distill` | Merge captured notes into wiki pages |
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
| `get <key>` | Print resolved value for a dotted config key (e.g. `compile.output.format`) |
| `set <key> <value>` | Write a key to project config (or `--global` for global config) |
| `list` | Print all resolved config keys with provenance |
| `path` | Print the config file path (or `--global`) |
| `check [FILES]` | Diff config files against defaults (see [Configuration](./configuration.md)) |

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

## `knox index`

Rebuild the zvec vector index for semantic search from the cached wiki.

```text
knox index [OPTIONS]
```

This is a **recovery command**. A normal `knox compile` already builds the
index automatically (unless `--no-index`); you only need `knox index` when
the index is missing, stale, or you changed `compile.embedding.model` or
`search.embed_attachments` and want to re-embed without a full LLM recompile.

It reads the cached wiki from `.knox/last_wiki.json` and rebuilds the zvec
store in place. It does **not** re-ingest attachments or recompile the wiki —
for that use `knox compile --force --start-at-tier chunk`.

**Options:**

| Flag | Default | Description |
|------|---------|------------|
| `--embed-model` | `compile.embedding.model` | Embedding model for zvec indexing (overrides config) |
| `--set key=val` | — | Override any config key for this run |

The `search.embed_attachments` setting (and per-attachment `--embed`/`--no-embed`
flags at registration) controls whether attachment-derived nodes are indexed.

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
6. **LLM context window** — auto-discovered size and source (litellm, Ollama, models.dev, or fallback)
7. **VLM context window** — same for the configured VLM model
8. **Budget consistency** — validates that `gen_output_max` and `extract_output_max` fit within the context window, and that sufficient input budget remains after reserving output tokens
9. **LLM connectivity** — quick ping to configured LLM
10. **VLM connectivity** — quick ping to configured VLM
11. **Embedding connectivity** — quick ping to configured embedding model

Exit code: `0` if no check fails, `1` if any check fails. Warnings are reported but do not fail the run; a missing LLM (`connectivity-llm`) is a hard failure.

Example output::

```text
  [OK] sources-directory: Sources dir OK: /path/to/sources
  [OK] output-directory: Output dir OK: /path/to/wiki
  [OK] parsers: 3 parser(s) registered
  [OK] git-repo: Inside a git repository
  [OK] disk-space: 102400 MB free
  [OK] context-window-llm: ollama/gemma4:e2b — 128000 in / 128000 out tokens (source: fallback)
  [OK] context-window-vlm: No vlm model configured — skipped
  [OK] budget-consistency: Budgets OK: 128000 ctx, 32768 gen_out, 16384 extract_out, ~95104 input
  [OK] connectivity-llm: llm OK (ollama/gemma4:e2b)
  [WARN] connectivity-vlm: No vlm model configured — skipped
  [WARN] connectivity-embed: No embed model configured — skipped

  9/11 checks passed, 2 warning(s)
```

Exit code: `0` if no check fails (warnings do not fail), `1` if any check fails.

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
| `--format` | `text` | Output format: `text`, `json`, `toon`, or `ison` |
| `--raw` | — | Return raw context chunks without LLM synthesis |
| `--no-classify` | — | Skip intent classification |
| `--verbosity` | `standard` | Answer verbosity: `brief`, `standard`, or `comprehensive` |

## `knox capture` and `knox learn`

Quickly persist notes, facts, or conversation snippets as **source material**
for future compilation. Files land in `sources/_captured/` — they are *inputs*
that the next `knox compile` will parse, chunk, and resolve like any other
source. See [Wiki structure: input vs output](wiki-structure.md#input-vs-output-where-files-land)
for why captures go under `sources/` rather than `wiki/`.

```bash
# LLM extraction into one or more knowledge items
knox capture "We decided to use QKV projections with RoPE"

# Store a fact verbatim
knox learn "The transformer uses d_model=512 and h=8" --tag attention
```

## `knox distill`

Synthesize captured notes from `sources/_captured/` into coherent wiki pages.
Notes are grouped by tag and merged (LLM-powered or deterministic fallback).
Output goes to `sources/_distilled/` — like `capture`/`learn`, distilled pages
are still *inputs* that the next `knox compile` will fold into the compiled
wiki.

```bash
# Merge all captured notes into synthesized pages
knox distill

# Then compile to incorporate distilled pages
knox compile
```

## `knox write`

Write or overwrite a single **finished wiki page** from one or more sources.
Unlike `capture`/`learn`/`distill` (which produce *input* material under
`sources/`), `write` produces a *publication artifact*: output lands directly
in `wiki/write/`, outside the compile pipeline. The page is synthesized from
the cached parse of each `--source` file (continuous prose via LLM, with a
concatenation fallback) and written with default frontmatter
(`title`, `kind=page`, `compiled_at`, `sources`).

```bash
# Write a page from one source
knox write "Attention Mechanism" --source sources/paper.pdf

# Write from multiple sources with extra frontmatter
knox write "Transformers" --source a.pdf --source b.docx \
  --frontmatter kind=concept status=draft

# Scaffold a placeholder page (no sources -> body is a pending marker)
knox write "TODO: fill this in"
```

Because `write` output is not a source, it is **not** re-parsed on subsequent
compiles — it persists as-is under `wiki/write/` until overwritten by another
`write`.

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

- Inline `[filename:start-end]` citations in `wiki/**/*.md`
- `.knox/provenance/*.json` per-node citation records
- `metadata.source_file` in `.knox/last_wiki.json`
- `source_file` in `.knox/compile_state.json`

A report is written to `.knox/last_organize.json`. Files without an
extraction summary are left in place at the top level.

## `knox sources`

Manage ingested source files: list them, tag them, and export them as clean
markdown. Tags are the same
`metadata.tags` field written by `--tag` at ingestion time; changes here are
persisted in the per-file parse cache under `.knox/cache/` and propagate to
wiki nodes on the next `knox compile`.

```bash
# List every ingested source with its tags
knox sources list

# List only sources carrying a tag (same AND/OR group semantics as query --tags)
knox sources list --tags paper,attention --tags transformer

# Add / remove a tag on an already-ingested source (no re-ingest needed)
knox sources tag sources/paper.pdf attention
knox sources untag sources/paper.pdf attention

# Export a source as clean markdown, reconstructed from the parse cache
# ("as ingested" — not from the original file)
knox sources export sources/paper.pdf out.md
```

### `knox sources export`

The source is rebuilt from the Tier-1 parse cache in `.knox/cache/`, so the
result is a faithful markdown rendering of what KNOX ingested. Figures are
copied to `<output stem>_assets/` next to the output file by default, or
embedded as data URIs with `--inline`. Attachments stay in the vault and are
listed in an `## Attachments` table at the end with relative links.

```bash
# Default: images to out_assets/, annotations + summary as markdown sections
knox sources export sources/paper.pdf out.md

# Embed images inline; skip extracted knowledge
knox sources export sources/paper.pdf out.md --inline --annotations off --summary off

# Write extracted knowledge to out_assets/extracted.json instead of sections
knox sources export sources/paper.pdf out.md --annotations asset --summary asset
```

`--annotations` / `--summary` take `off | inline | asset` and override the
`export.annotations` / `export.summary` config keys (default `inline`). The
`source_id` argument matches the `source_file` recorded at ingestion —
`knox sources list` shows the exact value to use.

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
