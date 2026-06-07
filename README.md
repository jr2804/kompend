# KNOX

**Knowledge Organizer for eXtended documents**

A compiled LLM-wiki system for scientific and technical documents. Parse, chunk, resolve, and compile PDFs, DOCX files, spreadsheets, and web sources into an interlinked, queryable knowledge base — with structure-preserving parsing, hybrid chunking, multi-turn resolution, and WiCER iterative compilation.

Inspired by [Karpathy's compiled LLM-wiki concept](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

## Quick Start

**As a CLI tool:**

```bash
uv tool install knox
```

**As a Python library:**

```bash
uv add knox
```

### 1. Check that a document parses correctly

Before compiling a full corpus, verify a single file extracts cleanly:

```bash
knox parse paper.pdf
```

This prints element counts, types (heading, table, image, formula, …), and the document title. Add `--elements` to see a preview of each extracted element, or pass multiple files and globs:

```bash
knox parse *.pdf ./more_docs/ --elements
knox parse ./research/ --recursive
```

If the output looks wrong, the full pipeline will produce poor results too — fix parsing issues first.

### 2. Compile a document corpus into a wiki

Drop your source files (PDF, DOCX, PPTX, EPUB, Markdown, HTML) into a directory and run:

```bash
knox compile ./sources
```

This runs the full pipeline — parse every document, chunk while preserving tables/figures/equations as atomic units, compile into an interlinked wiki, and write Markdown files to `./wiki` (the default output directory).

You can also pass specific files or globs directly — they'll be copied into the sources directory automatically:

```bash
knox compile paper.pdf notes.docx data/*.xlsx
knox compile ./papers/ --recursive
```

### 3. Extend with online sources and different output formats

```bash
# Pull in remote PDFs before compiling
knox compile ./sources --url https://arxiv.org/pdf/2605.07068v1

# Custom source and output directories
knox compile ./my_docs --sources-dir ./my_docs --output-dir ./output

# Output as JSON graph instead of Markdown
knox compile ./sources --format json

# Build a vector index for semantic search (requires zvec)
knox compile ./sources --format json --index --embed-model all-MiniLM-L6-v2
```

Full option reference: `knox compile --help`, `knox parse --help`

## CLI Reference

### `knox compile`

```
knox compile [OPTIONS] FILES...

Compile documents into a knowledge wiki.

Source files are gathered from the given paths (files, globs, directories)
and consolidated into the sources directory. Files not already there are
copied in. URLs are downloaded as well.

Arguments:
  FILES...              Files, glob patterns, or directories  [required]

Options:
  --sources-dir, -s    Input sources directory  [default: sources]
  --output-dir, -o     Output directory for compiled wiki  [default: wiki]
  --url TEXT            URLs to download before processing (repeatable)
  --recursive / --no-recursive   Search source directories recursively
                                  [env var: KNOX_RECURSIVE]  [default: no-recursive]
  --format, -f TEXT     Output format: markdown, json, typst  [default: markdown]
  --chunk-size INT      Chunk size in tokens  [default: 512]
  --image-mode TEXT     Image handling: file, base64  [default: file]
  --index               Index compiled wiki in zvec for semantic search
  --embed-model TEXT    Embedding model for zvec indexing
  --help                Show this message and exit
```

### `knox parse`

```
knox parse [OPTIONS] FILES...

Parse documents and show their structure.

Accepts one or more file paths, glob patterns (e.g. *.pdf), or directories.
Use --recursive to descend into subdirectories.

Arguments:
  FILES...              Files, glob patterns, or directories to parse  [required]

Options:
  --recursive / --no-recursive   Search source directories recursively
                                  [env var: KNOX_RECURSIVE]  [default: no-recursive]
  --elements, -e                 Print extracted elements
  --help                         Show this message and exit
```

### Global Options

```
knox --version, -v   Show version and exit
knox --help          Show help and exit
```

## What Makes KNOX Different

| Problem | Existing Tools | KNOX |
|---|---|---|
| Tables/figures flattened to text | All (pdfplumber → markdown → wiki) | Structure-preserving parser (Docling JSON sidecar) |
| No structure-aware chunking | Most use char-count splitting | Hybrid: custom atomic splitter + Chonkie for text |
| One-shot compilation | "Compile once and query" | WiCER loop: compile → evaluate → refine |
| Multi-turn resolution missing | No tool supports it | Tier 3: LLM + MCP tools per document |
| Model/context-agnostic | Hard-coded OpenAI/Anthropic | litellm + auto-detect context window + budget |
| GUI required for basic ops | nashsu/llm_wiki GUI-only | Headless CLI + MCP + Python SDK |
| Images orphaned | Flattened to filenames | VLM-captioned, indexed as first-class entities |

## Architecture

5-tier pipeline — each tier independently replaceable:

```
Raw Sources → TIER 1: Pre-Processing → TIER 2: Chunking → TIER 3: Resolution → TIER 4: Compilation → TIER 5: Output
```

Read the full architecture in [`docs/architecture.md`](./docs/architecture.md).

## Documentation

| Document | What's Covered |
|---|---|
| [`docs/overview.md`](./docs/overview.md) | Project summary, design philosophy, core principles |
| [`docs/architecture.md`](./docs/architecture.md) | 5-tier pipeline, data models, integration points |
| [`docs/ecosurvey.md`](./docs/ecosurvey.md) | Survey of existing tools, parsers, chunkers, MCP servers |
| [`docs/roadmap.md`](./docs/roadmap.md) | Milestones, priorities, open questions, tentative dates |
| [`docs/decisions.md`](./docs/decisions.md) | Decision log — what was chosen and why |
| [`docs/openwebui.md`](./docs/openwebui.md) | Connecting KNOX with Open-WebUI for chat/RAG |

## Development

```bash
git clone <repo-url> && cd knox
uv sync                    # install deps
uv run knox --help         # verify CLI
uv run pytest tests/       # run tests
mise test                  # run all quality checks
```

## Project Status

| Milestone | Status |
|---|---|
| M0: Scaffolding | ✅ Complete |
| M1: Docling Parser (Tier 1) | ✅ Complete |
| M2: Hybrid Chunking (Tier 2) | 🎯 Current |
| M3: LLM Resolution (Tier 3) | ⏳ Planned |
| M4: WiCER Compilation (Tier 4) | ⏳ Planned |
| M5: Output Writers (Tier 5) | ⏳ Planned |
| M6: MCP Server | ⏳ Planned |

See [`docs/roadmap.md`](./docs/roadmap.md) for full details.

## License

AGPL-3.0. See [LICENSE](./LICENSE).

## Acknowledgments

- [Karpathy's compiled LLM-wiki Gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — the core concept
- [WiCER paper](https://arxiv.org/pdf/2605.07068v1) (Huerta, Zinnia Tech) — iterative compilation loop
- [Docling](https://github.com/DS4SD/docling) (IBM) — structure-preserving parsing
- [Chonkie](https://github.com/chonkie-inc/chonkie) — multi-strategy chunking
- [litellm](https://github.com/BerriAI/litellm) — provider-agnostic LLM calls
