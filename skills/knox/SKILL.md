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
│   ├── _crystallized/  ← LLM-synthesized wiki pages
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
| `knox init` | Scaffold new wiki project | `--bare`, `--force`, `--git` |
| `knox compile <files>` | Compile sources into wiki | `--compiler graph\|wicer`, `--start-at-tier`, `--llm-default-model`, `--vlm-model`, `--embed-model`, `--format markdown\|json\|typst`, `--force`, `--no-cache`, `--workers N`, `--review`, `--verify`, `--provenance`, `--private`, `--synthesize`, `--flavor none\|pages\|wiki`, `--online-urls`, `--url-file` |
| `knox ingest <files>` | Copy files into sources without compiling | `--recursive` |
| `knox organize` | Organize existing wiki content into hierarchy | `--max-depth`, `--min-subcategories`, `--collapse-single-file` |
| `knox status` | Show wiki statistics | none |

### Query & Retrieval

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `knox query <text>` | Natural language query against compiled wiki | `--remember` (stores result in memory), `--raw` (raw output), `--search-mode hybrid\|token\|semantic`, `--top-k N`, `--llm-default-model` |
| `knox capture <text>` | Extract knowledge items from conversation text | — |
| `knox learn <text>` | Store a fact verbatim | — |
| `knox crystallize` | Synthesize captured notes into full wiki pages via LLM | — |

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

Run `knox mcp` to start an MCP server on stdio transport. All MCP tools return `{"status": "ok", ...}` or `{"status": "error", "message": "..."}`.

### Available MCP Tools

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `init_wiki` | Scaffold a new wiki project | `target: str` (directory path), `bare: bool = False`, `git: bool = True` |
| `compile_wiki` | Compile sources into wiki | `sources_dir: str`, `output: str = "./output"`, `output_format: str = "markdown"`, `compiler: str = "graph"`, `urls: list[str] \| None`, `vlm_model: str = ""`, `image_mode: str = "file"` |
| `query_wiki` | Ask a question against the wiki | `query: str`, `wiki_dir: str = "."`, `remember: bool = False`, `search_mode: str = "hybrid"`, `top_k: int = 10` |
| `wiki_status` | Show wiki statistics | `wiki_dir: str = "."` |
| `wiki_capture` | Extract knowledge from conversation | `text: str`, `sources_dir: str` |
| `wiki_learn` | Store a fact verbatim | `text: str`, `sources_dir: str` |
| `wiki_crystallize` | Synthesize captured notes into pages | `wiki_dir: str` |
| `memory_add` | Add query/answer to memory | `query: str`, `answer: str`, `feedback: str \| None`, `wiki_dir: str = "."` |
| `memory_search` | Search memory | `query: str`, `top_k: int = 5`, `wiki_dir: str = "."` |
| `memory_synthesize` | Generate system context from memory | `wiki_dir: str = "."`, `limit: int \| None` |
| `config_get` | Get a resolved config value | `key: str`, `wiki_dir: str = "."` |
| `config_set` | Set a config value | `key: str`, `value: str`, `wiki_dir: str = "."` |
| `parse_document` | Parse a single document | `file_path: str` |

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
2. After enough captures, call wiki_crystallize(wiki_dir=".") to synthesize
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

Wiki pages use the **ISON** format (semantically annotated Markdown with YAML frontmatter):

```yaml
---
title: "System Architecture"
type: concept
aliases: [arch, system-design]
tags: [architecture, infrastructure]
provenance:
  - source: "docs/architecture.pdf"
    page: 12
    confidence: 0.87
related:
  - "Database Schema": "references the database module for persistence"
  - "API Gateway": "routes requests through the gateway"
---
Content in standard Markdown...
```

The YAML frontmatter fields (`type`, `aliases`, `tags`, `provenance`, `related`)
carry machine-readable metadata for cross-linking, classification, source
attribution, and semantic relationships.

The wiki also includes:

- `wiki.md` — main compiled document (linearized)
- `index.md` — table of contents (re-generated on re-compile)
- `pages/` — individual page files (when `--flavor pages`)

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
| `compile.format` | `markdown` | Output format (markdown, json, typst) |
| `compile.compiler` | `graph` | Compiler backend (graph, wicer) |
| `compile.workers` | `1` | Parallel worker count |
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

## Event Hooks

Define post-compile or post-query scripts in `knox.toml`:

```toml
[hooks.on_compile]
command = "uv run scripts/notify.py"

[hooks.on_query]
command = "uv run scripts/log_query.py"
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
knox compile ./sources/*.pdf --llm-default-model "openai/gpt-4o"
```

### Compile with URLs

```bash
knox compile ./sources/*.pdf --online-urls "https://example.com/docs"
```

### Enable provenance tracking

```bash
knox compile ./sources/*.pdf --provenance
```

## Error Handling

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "No parser for X" | Unsupported file format | Check file extension. Supported: pdf, docx, pptx, html, md, epub, tex, doc, odt, rst, rtf, txt, png, jpg, csv, xlsx, adoc |
| Compilation fails with LLM error | Model not available / wrong name | Set `compile.llm.default_model` to a model litellm supports. Check connectivity. |
| query returns empty results | Wiki not compiled, or no matching content | Run `compile` first. Try `--search-mode hybrid`. Lower `search.cache.threshold`. |
| "Sources directory not found" | Wrong working directory | Run from project root, or pass --wiki-dir |
| Memory not working | memory disabled or no embedding model | Set `compile.memory_enabled=true` and `compile.embedding.model` |
