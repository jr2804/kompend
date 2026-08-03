---
name: knox
description: >-
  KNOX (Knowledge Organizer for eXtended documents) — compile documents (PDF,
  DOCX, HTML, Markdown, EPUB, images) into interlinked, queryable wikis with
  vector search and memory. Use when building knowledge bases, querying compiled
  wikis, capturing conversation knowledge, managing query memory, or automating
  wiki compilation.
---

# KNOX — Knowledge Builder

KNOX turns source documents into a structured, interlinked wiki with
semantic search, memory, and provenance tracking. Exposes a **CLI** and an
**MCP server**.

## Quick Start

```bash
# Run directly (no install needed)
uvx --from git+https://codeberg.org/jr2804/knox knox init          # scaffold project
uvx --from git+https://codeberg.org/jr2804/knox knox ingest *.pdf  # import sources
uvx --from git+https://codeberg.org/jr2804/knox knox compile ./sources/*.pdf  # build wiki
uvx --from git+https://codeberg.org/jr2804/knox knox query "..."   # ask questions

# Or install once
uv tool install --git https://codeberg.org/jr2804/knox
knox mcp                                                          # start MCP server
```

Sources can be placed directly in `./sources/` or imported via
`knox ingest <files>` (copies into sources) or `knox compile <files>`
(compiles directly, stores in sources for caching).

## Installation

Not on PyPI yet. Add to your project:

```toml
# pyproject.toml
[project.dependencies]
knox = { git = "https://codeberg.org/jr2804/knox" }
```

```bash
uv sync
```

Or install globally:

```bash
uv tool install --git https://codeberg.org/jr2804/knox
```

## Directory Layout

```text
.                       ← wiki project root
├── knox.toml           ← project config (optional, overriden by .knox/config.toml)
├── .knox/              ← internal state (cache, vectors, search db, memory)
├── sources/            ← source documents (PDF, DOCX, markdown, URLs)
│   ├── _captured/      ← knowledge items extracted from conversation
│   ├── _distilled/     ← LLM-synthesized wiki pages
│   └── _memory/        ← system context (auto-generated from memory)
└── wiki/               ← compiled output
    ├── wiki.md         ← main wiki document (first compile)
    ├── index.md        ← table of contents page (re-compile)
    └── pages/          ← individual page files (when output flavor is "pages")
```

## CLI Reference

### Project Lifecycle

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `knox init` | Scaffold new wiki project | `--bare`, `--force`, `--git`, `--config <toml>` (install a chosen config) |
| `knox compile` | Compile sources into wiki | `--start-at-tier`, `--no-cache`, `--workers N`, `--review`, `--verify/--no-verify`, `--synthesize/--no-synthesize`, `--url`, `--url-file`, `--set key=val` (config-only: model, format, compiler, budgets, timeouts) |
| `knox ingest <files>` | Copy files into sources without compiling | `--recursive`, `--url`, `--url-file`, `--parser <backend>`, `--set key=val` |
| `knox organize` | Organize existing wiki content into hierarchy | `--max-depth`, `--min-subcategories`, `--collapse-single-file` |
| `knox status` | Show wiki statistics | none |

### Source Management

| Command | Purpose | Key Flags |
|---------|---------|------------|
| `knox sources list` | List ingested source files with their tags | `--tags <group>` (repeatable; comma-separated AND within a group, multiple values OR across groups) |
| `knox sources tag <file> <tag>` | Add a tag to an already-ingested source | `--config <toml>` |
| `knox sources untag <file> <tag>` | Remove a tag from an ingested source | `--config <toml>` |

Tags are the same `metadata.tags` field written by `ingest`/`compile --tag`;
they persist in the per-source parse cache and propagate to wiki nodes on the
next `knox compile`. No re-ingest required.

### Assets (Attachments)

Supplementary files (photos, binaries, source code) that belong to a source
but are **not** ingested by default. Stored under `assets/<source-slug>/`
and tracked by an append-only manifest (`.knox/assets.jsonl`).

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `knox assets add <source> <file>` | Register an attachment for a source | `--ingest` (parse at compile; text via router, PNG/JPG via VLM), `--embed`/`--no-embed` (override global `search.embed_assets`) |
| `knox assets list` | List registered attachments | `--source <stem>` (filter), `--all` (include deleted) |
| `knox assets remove <path>` | Remove an attachment (tombstone in manifest, deletes file) | — |

Only attachments marked `--ingest` are processed during compile; PNG/JPG
require a configured VLM (`compile.llm.vlm_model`). Set `search.embed_assets =
true` to include attachment content in the vector index. Registered assets
are copied to `wiki/assets/` for relative links in generated content.

### Query & Retrieval

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `knox query <text>` | Natural language query against compiled wiki | `--remember` (stores result in memory), `--raw` (raw output), `--search-mode hybrid\|token\|semantic`, `--top-k N`, `--verbosity brief\|standard\|comprehensive`, `--format text\|json\|toon\|ison` |
| `knox trace [SLUG]...` | List pages with slugs, or trace page(s) to their source chunks | `--with-content`, `--json` |
| `knox capture <text>` | LLM-extract knowledge items from free text | `--tag` (repeatable) |
| `knox learn <text>` | Store a fact verbatim, no LLM | `--tag` (repeatable) |
| `knox write <TITLE>` | Create a one-shot wiki page from a title and optional sources | `--source` (repeatable, bare filename searched in sources/), `--frontmatter` |
| `knox distill` | Synthesize captured notes into full wiki pages via LLM | — |

### Memory

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `knox memory add <query> <answer>` | Add query/answer pair to memory | `--feedback`, `--embedding-model` |
| `knox memory search <query>` | Search past interactions semantically | `--top-k N` |
| `knox memory synthesize` | Generate `sources/_memory/system_context.md` from memory history | `--limit N` |

### Config Commands

| Command | Purpose |
|---------|---------|
| `knox config get <key>` | Show resolved config value |
| `knox config set <key> <value>` | Set a value in `.knox/config.toml` |
| `knox config list` | List all resolved config values with sources |

### Quality & Diagnostics

| Command | Purpose |
|---------|---------|
| `knox lint` | Structural + LLM-powered lint checks on compiled wiki |
| `knox doctor` | System diagnostics (checks env, dependencies, config) |

### Servers

| Command | Purpose |
|---------|---------|
| `knox mcp` | Start stdio MCP server (for agent integration) |
| `knox agent` | Start ACP agent on stdio (for credential-free LLM injection) |
| `knox agents` | Manage TOML-defined conversational agents (list, use, install, remove, validate) |

### ACP Agent Mode

KNOX implements the **Agent Client Protocol (ACP)** to let editors and AI agents
inject their LLM endpoint without sharing credentials. Run:

```bash
knox agent
```

This starts an ACP agent on stdio that:

1. Advertises `providers: {}` capability on `initialize`
2. Accepts `providers/set` to configure the LLM endpoint (base URL, auth headers, API type)
3. Creates sessions and responds to prompts via a conversational agent loop
4. Falls back to `/knox <command>` subprocess dispatch when no LLM is configured

**Conversational agents:** When an LLM is configured, prompts are routed through
a tool-calling loop driven by TOML-defined agents. The LLM reasons over
available tools (compile, query, capture, config) and calls them as needed.
See `knox agents list` for available agents.

**Options:**

- `--agent-file <path>` — use an explicit TOML agent definition instead of the
  configured active agent
- `--mcp-port <port>` — also start an MCP-over-HTTP server on the given port
  so connected AI agents can call KNOX's MCP tools directly

**Why ACP instead of MCP sampling?** KNOX needs full Pydantic structured output
for compilation — MCP sampling is free-text only. With ACP `providers/set`,
the editor configures KNOX's LLM endpoint at init time, and KNOX makes its own
direct HTTP calls with the full litellm pipeline.

**Slash commands:**

| Command | Purpose |
|---------|---------|
| `/knox compile [--start-at-tier <tier>] [--force]` | Compile sources into wiki |
| `/knox query <question> [--tag <name>]` | Query the compiled wiki |
| `/knox capture <note>` | Capture a note |
| `/knox status` | Show vault health and cache stats |
| `/knox help [<command>]` | Show help |

**Config precedence when using ACP:**

1. `providers/set` (ACP) — highest precedence
2. Env vars (`KNOX_LLM_*`)
3. `config.toml` `[compile.llm]`
4. Built-in defaults

This means even without an ACP client, KNOX falls back to env vars or config,
preserving `knox compile` in terminal and `knox mcp` for MCP-based agents.
| `knox serve` | Start web UI server (browser-based wiki viewer) |

## Compilation Pipeline

`knox compile` runs a 5-tier pipeline:

```text
Tier 1: Preprocessing   → parse sources, download URLs, deduplicate
Tier 2: Chunking        → split parsed documents into chunks (hierarchical window)
Tier 3: Resolution      → embed chunks, build vector index, resolve cross-references
Tier 4: Compilation     → LLM-assisted synthesis into wiki nodes with edges
Tier 5: Output          → render wiki nodes to Markdown/JSON with templates
```

Control entry point with `--start-at-tier` (skip earlier tiers) or `--force` (re-run a tier).

Compiler backends:

| Compiler | Behavior |
|----------|----------|
| `graph` (default) | Builds a directed graph of concepts with semantic edges. Best for exploratory/learning wikis. |
| `wicer` | Iterative weak-to-strong extraction. Best for deep extraction from dense documents. |

## MCP Server

Run `knox mcp` to start an MCP server on stdio transport. For credential-free
LLM injection via ACP, use `knox agent` instead (see ACP Agent Mode above).

All MCP tools return `{"status": "ok", ...}` or `{"status": "error", "message": "..."}`.

### Available MCP Tools

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `init_wiki` | Scaffold a new wiki project | `target: str` (directory path), `bare: bool = False`, `git: bool = True` |
| `compile_wiki` | Compile sources into wiki | `sources_dir: str`, `output: str = "./output"`, `output_format: str = "markdown"`, `compiler: str = "graph"`, `urls: list[str] \| None`, `vlm_model: str = ""`, `image_mode: str = "file"` |
| `query_wiki` | Ask a question against the wiki | `query: str`, `wiki_dir: str = "."`, `remember: bool = False`, `search_mode: str = "hybrid"`, `top_k: int = 10` |
| `wiki_status` | Show wiki statistics | `wiki_dir: str = "."` |
| `wiki_capture` | Extract knowledge from conversation | `text: str`, `sources_dir: str`, `tags: list[str] \| None` |
| `wiki_learn` | Store a fact verbatim | `text: str`, `sources_dir: str` |
| `wiki_distill` | Synthesize captured notes into pages | `wiki_dir: str` |
| `memory_add` | Add query/answer to memory | `query: str`, `answer: str`, `feedback: str \| None`, `wiki_dir: str = "."` |
| `memory_search` | Search memory | `query: str`, `top_k: int = 5`, `wiki_dir: str = "."` |
| `memory_synthesize` | Generate system context from memory | `wiki_dir: str = "."`, `limit: int \| None` |
| `config_get` | Get a resolved config value | `key: str`, `wiki_dir: str = "."` |
| `config_set` | Set a config value | `key: str`, `value: str`, `wiki_dir: str = "."` |
| `parse_document` | Parse a single document | `file_path: str` |
| `wiki_list_assets` | List registered attachments (optionally by source) | `source: str \| None`, `wiki_dir: str \| None` |
| `wiki_read_asset` | Read text content (up to 100KB) or metadata for binary files | `source: str`, `asset_name: str`, `wiki_dir: str \| None` |
| `wiki_asset_info` | Return manifest entry + file metadata | `source: str`, `asset_name: str`, `wiki_dir: str \| None` |

## Agent Workflows

### Workflow 1: Onboard a user to KNOX

```text
1. Call init_wiki(target=".", bare=False, git=True) to scaffold
2. Guide user to place source documents in ./sources/
3. Optionally configure compile.llm.default_model in knox.toml (or use env var KNOX_LLM_DEFAULT_MODEL)
4. Call compile_wiki(sources_dir="./sources", output="./output", ...)
5. Verify with query_wiki(query="What's in this wiki?", wiki_dir=".")
```

### Workflow 2: Answer questions and build memory

```text
1. Call query_wiki(query=user_question, wiki_dir=".", remember=True)
2. If user provides follow-up, call memory_add(query=..., answer=..., feedback=...)
3. Periodically, call memory_synthesize(wiki_dir=".") to distill themes into system context
```

### Workflow 3: Capture knowledge from conversation

```text
1. During a chat, periodically call wiki_capture(text=last_turns, sources_dir="./sources")
   to extract decisions, corrections, and discoveries into _captured/ notes
2. After enough captures, call wiki_distill(wiki_dir=".") to synthesize
   captured notes into full wiki pages via LLM
3. Re-compile with compile_wiki to integrate the new pages
```

### Workflow 4: Organize an existing wiki

```text
1. Run organize if wiki content needs hierarchy cleanup:
   knox organize --max-depth 3 --min-subcategories 2
2. Or via MCP: call compile_wiki with compiler="graph" after organization
```

### Workflow 5: Deep extraction from dense documents

```text
Use the wicer compiler for documents where you need exhaustive extraction:

compile_wiki(sources_dir="./sources", output="./output", compiler="wicer")
```

## Output Format

KNOX produces **OKF v0.1** (Open Knowledge Format) bundles by default — an open,
human- and agent-friendly format for representing knowledge as a directory of
markdown files with YAML frontmatter. [Spec →](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)

Legacy formats (Obsidian vault mode, JSON, Typst) are available via
`compile.output.format` config key.

### OKF frontmatter fields

```yaml
---
type: concept
title: "System Architecture"
description: One-line summary of the page
tags: [source-file-1, source-file-2]
timestamp: "2026-07-10T12:00:00"
kind: concept  # KNOX extension
compiled_at: "2026-07-10"  # KNOX extension
sources: [source-file-1.pdf]  # KNOX extension
---
Content in standard Markdown...
```

The wiki also includes:

- `index.md` — table of contents with grouped sections and descriptions
- `log.md` — ISO date headings with bold action prefix
- `concepts/` — subfolder for concept/entity/comparison pages
- `# Citations` section with numbered `[N]` refs from provenance

## Configuration

5-layer precedence (highest wins):

```text
1. CLI flags              → per-invocation overrides
2. KNOX_* env vars        → per-session overrides (e.g., KNOX_LLM_DEFAULT_MODEL)
3. .knox/config.toml      → project config (overrides knox.toml)
4. knox.toml              → project config
5. ~/.config/knox/config.toml → global per-user defaults
6. Built-in defaults      → compiled into the binary
```

Key config keys:

| Key | Default | Purpose |
|-----|---------|---------|
| `compile.llm.default_model` | `""` | LLM for compilation/querying |
| `compile.llm.vlm_model` | `""` | VLM for image captioning |
| `compile.embedding.model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `compile.output.format` | `okf` | Output format (okf, markdown, json, typst) |
| `compile.compiler` | `graph` | Compiler backend (graph, wicer) |
| `compile.workers` | `1` | Parallel worker count |
| `ingest.parser` | `liteparse` | Ingestion backend, or comma-separated priority list (liteparse, docling, pandoc, …; first match wins). Backends are optional installs via `uv sync --group <name>` |
| `compile.memory_enabled` | `true` | Enable memory store |
| `dedup.enabled` | `false` | Enable deduplication |
| `trust.privacy_filter` | `true` | Filter PII from queries |

Set via:

```bash
knox config set compile.llm.default_model "ollama/llama3.2"
```

Or env var:

```bash
export KNOX_LLM_DEFAULT_MODEL="ollama/llama3.2"
```

## Common Patterns

### Start fresh (force full re-compile)

```bash
knox compile ./sources/*.pdf --force --no-cache
```

### Skip to compilation (reuse cached preprocessing)

```bash
knox compile ./sources/* --start-at-tier 4
```

### Use a specific LLM

```bash
knox config set compile.llm.default_model "openai/gpt-4o"
# or per-invocation:
knox compile --set compile.llm.default_model="openai/gpt-4o"
```

### Compile with URLs

```bash
knox compile --url "https://example.com/docs"
```

### Enable provenance tracking

```bash
knox config set compile.provenance true
```

## Error Handling

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "No parser for X" | Unsupported file format | Check file extension. Supported: pdf, docx, pptx, html, md, epub, tex, doc, odt, rst, rtf, txt, png, jpg, csv, xlsx, adoc |
| Compilation fails with LLM error | Model not available / wrong name | Set `compile.llm.default_model` to a model litellm supports. Check connectivity. |
| query returns empty results | Wiki not compiled, or no matching content | Run `compile` first. Try `--search-mode hybrid`. Lower `search.cache.threshold`. |
| "Sources directory not found" | Wrong working directory | Run from project root, or pass --wiki-dir |
| Memory not working | memory disabled or no embedding model | Set `compile.memory_enabled=true` and `compile.embedding.model` |
