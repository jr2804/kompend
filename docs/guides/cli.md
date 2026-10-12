# CLI Reference

KNOX provides a Typer-based CLI with the following commands:

| Command | Description |
|---------|-------------|
| `knox init` | Scaffold a new wiki project |
| `knox config` | Inspect and manage configuration |
| `knox parse` | Parse documents and show their structure |
| `knox compile` | Compile documents into a knowledge wiki |
| `knox ingest` | Copy files into sources without compiling |
| `knox query` | Query the compiled wiki with natural language |
| `knox doctor` | Validate configuration and connectivity |
| `knox status` | Show per-wiki statistics |
| `knox list-parsers` | List all registered parsers |
| `knox lint all` | Full lint pipeline: check → format → semantic → gaps → remediate (single weighted progress bar; LLM steps skipped without a model) |
| `knox lint check` | Deterministic structural rules, auto-fix on |
| `knox lint semantic` | LLM-powered rules (factual-consistency, clarity, coverage-gap, contradictions) |
| `knox lint history` | Show past lint runs and score trends |
| `knox lint gaps` | Analyze lint results for knowledge gaps and source recommendations |
| `knox lint remediate` | LLM-generate missing pages for lint findings |
| `knox lint prune` | Remove old lint history entries |
| `knox review` | Manage the review queue |
| `knox sources list` | List ingested source files with their tags (`--tags` filter) |
| `knox sources tag` | Add a tag to an ingested source file |
| `knox sources untag` | Remove a tag from an ingested source file |
| `knox sources export` | Render an ingested source as clean markdown |
| `knox provenance [SLUG]` | List pages with slugs, or trace a page to its sources |
| `knox capture` | LLM-extract knowledge items from free text (`--tag`, `--attr`) |
| `knox learn` | Store a fact verbatim as a source, no LLM (`--tag`, `--attr`) |
| `knox distill` | Merge captured notes into wiki pages (updates in place, archives originals) |
| `knox consolidate` | Find and archive redundant captured/learned/distilled notes |
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
| `--verbose` | Show full tracebacks for errors (fatal crashes and recoverable parse failures). Default is a one-line error. |
| `--help` | Show help message and exit |

`--wiki-dir` can also be set via the `KNOX_WIKI_DIR` environment variable, and
`--verbose` via `KNOX_VERBOSE=1`.

Errors are **quiet by default**: a failed parse logs a single
`WARNING: Failed to parse <file>: <error>` line and the run continues, and a
fatal error prints one `error: <Type>: <message>` line. `--verbose` (or
`KNOX_VERBOSE=1`) restores full tracebacks for both.

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
│   ├── knox.db        # Single non-vector store
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
| `check [FILES]` | Diff config files against defaults (omits to check the project config; `-w` supported) |

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

## `knox ingest`

Copy files (or URLs) into `sources/` without compiling. Like `capture`/`learn`,
ingest accepts `--tag` (repeatable) and `--attr KEY=VALUE` (repeatable) —
written to the parse-cache metadata and propagated to wiki nodes on the next
`knox compile`. See [Input pipeline](input-pipeline.md) for the full
capture/learn/distill/consolidate lifecycle.

```bash
knox ingest paper.pdf --tag audio --tag standards --attr version=2
knox ingest --url https://example.com/report.pdf --attr status=draft
```

### Forced re-ingest (parser setting change)

`knox ingest --no-cache` forces a re-parse of every selected source. Two
things make this safe and sufficient for a parser-related change:

- **Tags and attributes are preserved.** `knox sources tag` / `attr add` write
  into the parse-cache entry; a re-parse carries them forward instead of
  overwriting them from fresh parser output. CLI `--tag` / `--attr` are merged
  on top of the preserved values.
- **Attachments are untouched.** They live in the attachment manifest in
  `.knox/knox.db` and `attachments/<slug>/`; `ingest` never writes those.

A parser-related change invalidates automatically: the parse cache keys
on `ParserRouter.fingerprint()` and the chunk cache keys on the same
fingerprint. The fingerprint folds in the `[ingest] parsecraft_*` keys
(`parsecraft_judge`, `parsecraft_judge_model`, `parsecraft_preference`)
and the adapter's `max_passes`, so a change to any of those — and not to
the backend name, which is no longer selectable — invalidates the cache.
So after editing an `[ingest]` knob, a plain `knox ingest` re-parses and
the next `knox compile` re-chunks; no `--no-cache` needed. Use `--no-cache`
only to force a re-parse when nothing changed.

```bash
# change a parser knob: edit [ingest] parsecraft_*, then
knox ingest                 # re-parses the affected sources automatically
knox compile                # re-chunks from the new parser output

# force a re-parse regardless
knox ingest --no-cache
```

The `[ingest]` block now holds only the parsecraft parameters; the
backend selector was retired (the parser chain is parsecraft and only
parsecraft, with per-page routing inside it). A vault that still sets
the old `ingest.parser` key loads and is warned by `knox config check`;
the key lives in `REMOVED_KEYS`. The live shape is:

```toml
[ingest]
parsecraft_judge         = "none"        # none | ollama | zen | typesafe-ai
parsecraft_judge_model   = ""            # model token for the configured provider
parsecraft_preference    = "balanced"    # tiebreak: speed | balanced | quality (speed == balanced; quality inverts)
```

`knox[parsecraft]` adds parsecraft and its sub-backends; `knox[mineru]`
adds MinerU as a parsecraft sub-backend; `knox[systemone]` adds the
judge tier's SDK; marker is parsecraft's bring-your-own dependency. See
[ParsecraftParser](document-extraction.md#parsecraftparser-external-adapter)
for the fallback interaction and the diagnostics parsecraft carries into
KNOX metadata.

## `knox index`

Rebuild the zvec vector index for semantic search from the cached wiki.

```text
knox index [OPTIONS]
```

This is a **recovery command**. A normal `knox compile` already builds the
index automatically (unless `--no-index`); you only need `knox index` when
the index is missing, stale, or you changed `compile.embedding.model` or
`attachments.embed` and want to re-embed without a full LLM recompile.

It reads the compiled-wiki snapshot from `.knox/knox.db` and rebuilds the zvec
store in place. It does **not** re-ingest attachments or recompile the wiki —
for that use `knox compile --force --start-at-tier chunk`.

**Options:**

| Flag | Default | Description |
|------|---------|------------|
| `--embed-model` | `compile.embedding.model` | Embedding model for zvec indexing (overrides config) |
| `--set key=val` | — | Override any config key for this run |

The `attachments.embed` setting (and per-attachment `--embed`/`--no-embed`
flags at registration) controls whether attachment-derived nodes are indexed.
The key was renamed from `search.embed_attachments`; a config still setting the old name is migrated with a warning (see `docs/guides/attachments.md`).

## `knox attachments`

Manage per-source supplementary files. `add`, `remove`, and `list` are covered
in the [Attachments guide](attachments.md); two maintenance commands need no
compile:

| Command | Description |
|---------|-------------|
| `knox attachments verify` | Read-only check: orphan directories (slug matches no source) and manifest rows with no file on disk. Exits `1` when anything is found, `0` when clean. |
| `knox attachments adopt` | Register orphaned files under `attachments/<slug>/` — the same adoption the compile hook runs. Idempotent. |

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
6. **LLM context window** — auto-discovered size and source (provider override, Ollama, models.dev, or fallback)
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

List all registered parsers (built-in, external, and HTTP). External
backends appear only when their package extra is installed — `parsecraft`
shows up after `uv add knox[parsecraft]`, and is otherwise absent (not an
error, just not registered).

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
| `--tags <group>` | — | Filter results by tag. Comma-separated values within one flag are AND'd; multiple `--tags` flags are OR'd (e.g. `--tags audio,paper --tags video`) |
| `--attr KEY=VALUE` | — | Filter results by source attribute. Repeatable; multiple flags are AND'd. Bare `KEY` matches any value for that key (e.g. `--attr version=2 --attr status`) |

## `knox capture` and `knox learn`

Quickly persist notes, facts, or conversation snippets as **source material**
for future compilation. Files land in `sources/_captured/` (capture) and
`sources/_learned/` (learn) — they are *inputs* that the next `knox compile`
will parse, chunk, and resolve like any other source. See
[Input pipeline](input-pipeline.md) for the full capture/learn/distill/
consolidate story, and
[Wiki structure: input vs output](wiki-structure.md#input-vs-output-where-files-land)
for why captures go under `sources/` rather than `wiki/`.

```bash
# LLM extraction into one or more knowledge items (needs substantive input)
knox capture "We decided to use QKV projections with RoPE for the attention mechanism"

# Store a fact verbatim (short notes are fine here)
knox learn "The transformer uses d_model=512 and h=8" --tag attention

# Both accept --attr KEY=VALUE (repeatable) for filterable facts
knox capture "..." --attr version=2 --attr status=confirmed
knox learn "..." --attr version=2
```

`capture` runs an LLM extraction over the text, so it rejects inputs shorter
than 80 characters — use `learn` for short verbatim facts.

## `knox distill`

Synthesize captured notes from `sources/_captured/` and `sources/_learned/`
into coherent wiki pages. Notes are grouped by tag and merged (LLM-powered or
deterministic fallback). Output goes to `sources/_distilled/` — like
`capture`/`learn`, distilled pages are still *inputs* that the next
`knox compile` will fold into the compiled wiki.

Repeated runs **update** the existing page for each tag group in place (no
accumulating timestamped duplicates), and by default the distilled source
notes are archived to `sources/_archive/<kind>/`. Pass `--no-archive` to
keep them in place.

```bash
# Merge all captured notes into synthesized pages (archives originals)
knox distill

# Keep the source notes in place
knox distill --no-archive

# Then compile to incorporate distilled pages
knox compile
```

## `knox consolidate`

Find and archive redundant notes across `sources/_captured/`,
`sources/_learned/` and `sources/_distilled/`. Near-duplicates are detected
with rapidfuzz token similarity, plus an embedding tier when an embedding
model is configured. For each duplicate group the newest or longest note is
kept; the rest are archived to `sources/_archive/<kind>/`.

```bash
# Preview what would be archived
knox consolidate --dry-run

# Archive duplicates (default)
knox consolidate

# Report only, keep everything in place
knox consolidate --no-archive
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
are available in `.knox/knox.db`. The LLM is given the summaries and
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
- the `provenance` table in `.knox/knox.db` (per-node citation records)
- `metadata.source_file` in the compiled-wiki snapshot in `.knox/knox.db`
- `source_file` in the `tier_state` table of `.knox/knox.db`

A report is written to `.knox/last_organize.json`. Files without an
extraction summary are left in place at the top level.

## `knox sources`

Manage ingested source files: list them, tag them, set attributes, and export them as clean
markdown. Tags are the same
`metadata.tags` field written by `--tag` at ingestion time; changes here are
persisted in the parse cache in `.knox/knox.db` and propagate to
wiki nodes on the next `knox compile`.

Attributes are `KEY=VALUE` facts per source (`version=2`, `status=draft`) —
same parse-cache persistence and compile propagation as tags. Unlike tags,
attribute values are collected **per source**: a page built from sources with
`version=1` and `version=2` carries `{version: ["1", "2"]}` in its frontmatter
and matches either value in a query filter. Tags stay flat strings for
hierarchy-style structuring (`ml/transformers`).

```bash
# List every ingested source with its tags
knox sources list

# List only sources carrying a tag (same AND/OR group semantics as query --tags)
knox sources list --tags paper,attention --tags transformer

# Add / remove a tag on an already-ingested source (no re-ingest needed)
knox sources tag sources/paper.pdf attention
knox sources untag sources/paper.pdf attention
# Set / remove a key=value attribute on an ingested source
knox sources attr sources/paper.pdf version=2
knox sources attr-rm sources/paper.pdf version

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
| `KNOX_LLM_DEFAULT_MODEL` | `compile.llm.default_model` | Default LLM model |
| `KNOX_VLM_MODEL` | `compile.llm.vlm_model` | Vision model for image captioning |
| `KNOX_NOTES_FILE` | `notes.file` | User notes filename |
| `KNOX_RESOLVE_RELATIONS` | `compile.resolve_relations` | Enable typed relation extraction |
| `KNOX_RELATION_CONFIDENCE_THRESHOLD` | `compile.relation_confidence_threshold` | Minimum confidence for relations |
| `KNOX_LLM_AGENTIC_OUTPUT_RETRIES` | `compile.llm.agentic_output_retries` | Output retry budget for agentic agents (default 4) |
| `KNOX_LLM_THROUGHPUT_WINDOW_TOKENS` | `compile.llm.throughput_window_tokens` | Rolling window (output tokens) for the short-term tok/s rate (default 1000) |

## Agentic Mode

KNOX uses **agentic multi-turn workflows** for LLM operations that benefit
from on-demand fetching or cross-item synthesis in one session. Instead of a
single-shot call per item, an agentic agent runs one session per task and
iterates via tools — fetching data on demand, batching work, and synthesizing
across sources.

Agentic mode activates when `compile.llm.default_model` is set and the
operation routes through the agentic factory. For the operations it covers,
the agentic variant is the default for the affected CLI commands and MCP
tools, for example:

- `knox distill` → `AgenticDistillAgent` (one session over all tag groups)
- `knox write` → `AgenticWriteAgent` (loads sources on demand)
- `knox lint verify` → `AgenticVerificationAgent` (one session over claims)

Not every operation is agentic: a set is **single-shot by design** (reference
extraction/resolution, capture, categorization, lint rules, WiCER
sub-calls, VLM captioning, doctor, query expansion/CRAG/HyDE). See the
[decision record](../architecture/decisions.md#decision-which-llm-actions-stay-single-shot-by-design)
before proposing to convert one.

The `compile.llm.agentic_output_retries` config key controls the output
retry budget for agentic agents separately from tool retries (see
`compile.llm.output_retries`).
