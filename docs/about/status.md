# Implementation Status

Current state of the KNOX codebase as of the latest `main` branch.

## Tier Status

| Tier | Component | Status | Notes |
|------|-----------|--------|-------|
| T1 | Preprocessing | ✅ Done | Docling, pdfplumber, pandoc, VLM parsers; HTTP downloader + web crawler |
| T2 | Chunking | ✅ Done | HybridChunker (atomic + Chonkie RecursiveChunker), DevChunker, standalone ChonkieChunker adapter |
| T3 | Resolution | ✅ Done | `LLMResolver` (typed relations), `VLMEnricher` (image captioning), pluggable resolver chain |
| T4 | Compilation | ✅ Done | `GraphCompiler` + `WiCERCompiler` (iterative compile-evaluate-refine loop) |
| T5 | Output | ✅ Done | Markdown, JSON, Typst writers |

## Cross-Cutting Components

| Component | Status | Notes |
|-----------|--------|-------|
| CLI (`knox compile`, `knox parse`) | ✅ Done | Full option set: recursive, format, chunk-size, image-mode, URL download, zvec indexing |
| MCP Server | ⚠️ Partial | Functional stdio server with `parse_document`, `compile_wiki`, `wiki_capture`, `wiki_learn` tools. Uses newline-delimited JSON, not JSON-RPC 2.0 |
| RAG / Vector Search | ✅ Done | `ZvecWikiStore` with sentence-transformers / LiteLLM embedding providers; `HybridRetriever` fuses FTS + vector + token-overlap + raw sources via RRF |
| Memory Lifecycle | ✅ Done | Confidence decay, tier promotion (working→episodic→semantic→procedural), supersession detection, stale-page lint integration |
| Event Hooks | ✅ Done | `PRE_COMPILE`, `POST_COMPILE`, `ON_CAPTURE`, `ON_QUERY`, `ON_LINT` hooks; configurable via `[hooks]` in knox.toml |
| Data Models | ✅ Done | Core models (`ElementType`, `Chunk`, `ResolvedChunk`, `CompiledWiki`) are stable |

## Known Gaps

- **MCP protocol**: Server reads newline-delimited JSON instead of JSON-RPC 2.0. Needs proper protocol implementation.
- **HybridChunker budget**: Measures budget in characters, not tokens (M3).
- **DoclingParser**: Requires update for Docling 2.x API — `doc.pages` yields ints, not page objects (C2).
- **Eval harness**: Health score, citation coverage, CI thresholds not yet implemented (N2).

## Lint

| Component | Status | Notes |
|-----------|--------|-------|
| Structural lint rules (8) | ✅ Done | broken-wikilinks, missing-frontmatter, orphan-page, duplicate-title, empty-page, heading-skip, stale-page, source-not-found |
| LLM-powered lint rules (4) | ✅ Done | factual-consistency, clarity-score, coverage-gap, contradiction-detection (opt-in via `--llm`) |
| CLI: `knox lint` | ✅ Done | --fix, --llm, --rule, --threshold, --json |
| CLI: `knox compile --lint` | ✅ Done | Runs lint post-compile |
| Eval harness (N2) | ⏳ Pending | Health score, citation coverage, CI thresholds |

## Parser Coverage

| Format | Parser | Status |
|--------|--------|--------|
| PDF | `DoclingParser` (primary), `PdfPlumberParser` (fallback) | ✅ |
| DOCX / DOC | `DoclingParser`, `PandocParser` | ✅ |
| PPTX | `DoclingParser` | ✅ |
| EPUB | `PandocParser` | ✅ |
| Markdown | `PandocParser` | ✅ |
| HTML | `PandocParser` | ✅ |
| TXT | Direct read | ✅ |
| URLs (PDF) | `HTTPDownloader` | ✅ |
| URLs (HTML) | `WebCrawler` (crawl4ai) | ✅ |

## Configuration

`PipelineConfig` fields currently exposed:

| Field | Default | Used by |
|-------|---------|---------|
| `compile.sources_dir` | `"sources"` | Pipeline |
| `compile.output_dir` | `"wiki"` | Pipeline |
| `compile.image_mode` | `"file"` | DoclingParser |
| `compile.chunk_size` | `512` | HybridChunker |
| `compile.format` | `"markdown"` | Output writer selection |
| `compile.llm.default_model` | set via env / knox.toml | LLMResolver, LLMClient |
| `compile.llm.vlm_model` | `None` (disabled) | VLMEnricher (`--vlm-model`) |
| `compile.resolve_relations` | `False` | LLMResolver typed relation extraction |
| `compile.relation_confidence_threshold` | `0.0` | Relation filtering |
| `compile.memory_enabled` | `True` | Memory lifecycle (`--memory`/`--no-memory`) |
| `compile.memory_half_life_days` | `90.0` | Confidence decay rate |
| `compile.memory_decay_threshold` | `0.3` | Stale-page detection |
| `compile.embedding.dim` | `384` | zvec vector store |
| `hooks` | `{}` | Event-driven hooks |

See [roadmap](./roadmap.md) for milestone-level planning and [architecture](../architecture/index.md) for the full pipeline design.
