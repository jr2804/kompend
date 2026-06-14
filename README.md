# KNOX

## Knowledge Organizer for eXtended documents

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

### 1. Initialize a project

```bash
knox init ./my-wiki
cd ./my-wiki
```

This creates a standard project structure with `sources/`, `wiki/`, `.knox/config.toml`, `NOTES.md`, and `README.md`. Add `--bare` for minimal scaffolding, or `--git` to initialize a git repo.

### 2. Check that a document parses correctly

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

### 3. Compile a document corpus into a wiki

Drop your source files (PDF, DOCX, PPTX, EPUB, Markdown, HTML) into `sources/` and run:

```bash
knox compile
```

This runs the full pipeline — parse every document, chunk while preserving tables/figures/equations as atomic units, compile into an interlinked wiki, and write Markdown files to `wiki/`.

You can also pass specific files or globs directly — they'll be copied into the sources directory automatically:

```bash
knox compile paper.pdf notes.docx data/*.xlsx
knox compile ./papers/ --recursive
```

### 4. Extend with online sources and different output formats

```bash
# Pull in remote PDFs before compiling
knox compile --url https://arxiv.org/pdf/2605.07068v1

# Override the project root (sources/ and wiki/ resolved from this)
knox --wiki-dir ./custom-root compile

# Output as JSON graph instead of Markdown
knox compile --format json

# Build a vector index for semantic search (requires zvec)
knox compile --format json --index --embed-model all-MiniLM-L6-v2
```

Full option reference: `knox compile --help`, `knox init --help`, `knox config --help`

## CLI Reference

### `knox init`

```text
knox init [OPTIONS] [DIR]

Scaffold a new wiki project.

Arguments:
  DIR                   Target directory (default: current directory)

Options:
  --bare                Skip authored templates (NOTES.md, README.md)
  --force               Overwrite existing project config
  --git                 Initialize a git repo with an initial commit
  --help                Show this message and exit
```

### `knox config`

```text
knox config [OPTIONS] COMMAND

Inspect and manage configuration files.

Commands:
  get <key>             Print resolved value for a config key
  set <key> <value>     Write a key to project or global config
  list                  Print all resolved config with provenance
  path                  Print the config file path
```

### `knox compile`

```text
knox compile [OPTIONS] FILES...

Compile documents into a knowledge wiki.

Arguments:
  FILES...              Files, glob patterns, or directories

Options:
  --format, -f TEXT     Output format: markdown, json, typst  [default: markdown]
  --chunk-size INT      Chunk size in tokens  [default: 512]
  --image-mode TEXT     Image handling: file, base64  [default: file]
  --index               Index compiled wiki in zvec for semantic search
  --embed-model TEXT    Embedding model for zvec indexing
  --help                Show this message and exit
```

### `knox parse`

```text
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

```text
knox --version, -v        Show version and exit
knox --wiki-dir, -w DIR   Set project root (sources/, wiki/, .knox/ resolved from here)
knox --help               Show help and exit
```

## Architecture

KNOX runs a 5-tier pipeline — each tier independently replaceable:

```text
Raw Sources → T1: Pre-Processing → T2: Chunking → T3: Resolution → T4: Compilation → T5: Output
```

**T1 Pre-Processing** — Parse PDF, DOCX, PPTX, EPUB, Markdown, HTML into a uniform `ParsedDocument` with structure (headings, tables, figures, equations, code). Download URLs via HTTP or crawl4ai.

**T2 Chunking** — Split documents into `Chunk` objects. Atomic elements (tables, figures, formulas, code blocks) are never split; text runs are chunked via Chonkie's RecursiveChunker with markdown awareness.

**T3 Resolution** — Extract entities and resolve cross-references using an LLM (via litellm). Falls back to passthrough when no LLM is available.

**T4 Compilation** — Build an interlinked `CompiledWiki` graph from resolved chunks. Optionally runs the WiCER loop: compile → generate probes → evaluate → diagnose → refine.

**T5 Output** — Write the compiled wiki as Markdown, JSON, or Typst. Optionally index for semantic search via zvec.

For the full architecture, see [`docs/architecture/index.md`](./docs/architecture/index.md).

## Documentation

> **Read the full docs at [knox.anomalyco.dev](https://knox.anomalyco.dev)** — architecture deep-dives, pipeline tier references, WiCER loop explainer, CLI guides, and integration docs.

For a quick reference:

| Topic | Location |
|---|---|
| Architecture & data models | [`docs/architecture/index.md`](./docs/architecture/index.md) |
| Implementation status | [`docs/about/status.md`](./docs/about/status.md) |
| Roadmap & milestones | [`docs/about/roadmap.md`](./docs/about/roadmap.md) |
| Decision log | [`docs/architecture/decisions.md`](./docs/architecture/decisions.md) |
| Ecosystem survey | [`docs/reference/ecosystem.md`](./docs/reference/ecosystem.md) |
| Open-WebUI integration | [`docs/integrations/open-webui.md`](./docs/integrations/open-webui.md) |

## Development

```bash
git clone <repo-url> && cd knox
uv sync                    # install deps
uv run knox --help         # verify CLI
uv run pytest tests/       # run tests
mise test                  # run all quality checks
```

## License

AGPL-3.0. See [LICENSE](./LICENSE).

## Acknowledgments

- [Karpathy's compiled LLM-wiki Gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — the core concept
- [Rohit G00's LLM Wiki v2 Gist](https://gist.github.com/rohitg00/2067ab416f7bbe447c1977edaaa681e2) — memory lifecycle, knowledge graph, hybrid search, and event-driven extensions
- [WiCER paper](https://arxiv.org/pdf/2605.07068v1) (Huerta, Zinnia Tech) — iterative compilation loop
- [Docling](https://github.com/DS4SD/docling) (IBM) — structure-preserving parsing
- [Chonkie](https://github.com/chonkie-inc/chonkie) — multi-strategy chunking
- [litellm](https://github.com/BerriAI/litellm) — provider-agnostic LLM calls
