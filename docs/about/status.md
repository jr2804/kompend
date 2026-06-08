# Implementation Status

Current state of the KNOX codebase as of the latest `main` branch.

## Tier Status

| Tier | Component | Status | Notes |
|------|-----------|--------|-------|
| T1 | Preprocessing | ✅ Done | Docling, pdfplumber, pandoc, VLM parsers; HTTP downloader + web crawler |
| T2 | Chunking | ✅ Done | HybridChunker (atomic + Chonkie RecursiveChunker), DevChunker, standalone ChonkieChunker adapter |
| T3 | Resolution | ⚠️ Partial | `LLMResolver` works; `VLMEnricher` implemented but not wired into pipeline |
| T4 | Compilation | ✅ Done | `GraphCompiler` + `WiCERCompiler` (iterative compile-evaluate-refine loop) |
| T5 | Output | ✅ Done | Markdown, JSON, Typst writers |

## Cross-Cutting Components

| Component | Status | Notes |
|-----------|--------|-------|
| CLI (`knox compile`, `knox parse`) | ✅ Done | Full option set: recursive, format, chunk-size, image-mode, URL download, zvec indexing |
| MCP Server | ⚠️ Partial | Functional stdio server with `parse_document` and `compile_wiki` tools. Uses newline-delimited JSON, not JSON-RPC 2.0 |
| RAG / Vector Search | ⚠️ Partial | `ZvecWikiStore` works but uses hash-based pseudo-embeddings; needs real embedding model integration |
| Data Models | ⚠️ Partial | Core models (`ElementType`, `Chunk`, `CompiledWiki`) are stable. Some fields in docs differ from implementation |

## Known Gaps

- **VLMEnricher not in pipeline**: `VLMEnricher` exists in `resolution/vlm_enricher.py` but the CLI and MCP server skip resolution entirely. Needs a resolver chain (M7).
- **ZvecWikiStore embeddings**: Uses deterministic hash-based vectors for search. Real embeddings (e.g. sentence-transformers) need integration (M6).
- **MCP protocol**: Server reads newline-delimited JSON instead of JSON-RPC 2.0. Needs proper protocol implementation.
- **HybridChunker budget**: Measures budget in characters, not tokens (M3).
- **DoclingParser**: Requires update for Docling 2.x API — `doc.pages` yields ints, not page objects (C2).

## Lint

| Component | Status | Notes |
|-----------|--------|-------|
| Structural lint rules (8) | ✅ Done | broken-wikilinks, missing-frontmatter, orphan-page, duplicate-title, empty-page, heading-skip, stale-page, source-not-found |
| LLM-powered lint rules (3) | ✅ Done | factual-consistency, clarity-score, coverage-gap (opt-in via `--llm`) |
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
| `source_dir` | required | Pipeline |
| `output_dir` | required | Pipeline |
| `image_mode` | `"file"` | DoclingParser |
| `chunk_size` | `512` | HybridChunker |
| `compile_target` | `"markdown"` | Output writer selection |
| `llm_model` | `"llama3.2"` | LLMResolver (when wired) |
| `llm_provider` | `"ollama"` | LLMResolver (when wired) |
| `vlm_model` | `"phi-3-vision"` | VLMEnricher (when wired) |
| `online_sources` | `[]` | HTTPDownloader |
| `input_files` | `None` | Pipeline (auto-discover when None) |

See [roadmap](./roadmap.md) for milestone-level planning and [architecture](../architecture/index.md) for the full pipeline design.
