# Ecosystem Survey

A living survey of compiled LLM-wiki tools, PDF pre-processors, chunking libraries, and hybrid RAG components. Updated continuously based on weekly research.

## LLM-Wiki Compilers

| Tool | Stars | Lang | PDF Sci Quality | Long Docs (100+) | Formats | Integration | Hermes Ready |
|---|---|---|---|---|---|---|---|
| **xoai/sage-wiki** | 530 | Go | ⚠️ Pure-Go weak (v0.1.9 ext parsers) | ✅ Tiered splitting | **Excellent** (PDF/DOCX/XLSX/PPTX/EPUB/CSV/img) | CLI + MCP + TUI + Web + Packs | ✅ HIGH |
| **atomicstrata/llm-wiki-compiler** | 1.4k | TS | ⚠️ Text-only pdf-parse | ✅ Budget control + Eval | Medium (MD/PDF) | CLI + MCP + Web + Graph | ✅ High |
| **lucasastorian/llmwiki** | 1k | Python+TS | ✅ **Mistral OCR** (best, but paid) | ⚠️ Page-range reads | Good (PDF/MD/HTML/XLSX/DOCX) | CLI + MCP + Web | ✅ High |
| **nashsu/llm_wiki** | 10.2k | Rust+TS | ⚠️ Basic (Tauri pdf-extract) | ⚠️ SHA256 cache | Good (PDF/DOCX/MD/URL) | **GUI only** + Skill | ❌ Low |
| **nvk/llm-wiki** | ~504 | MD/AGS | ⚠️ Agent-dependent | ✅ Collector catalogs | Medium | Skill-only + Collector | ⚠️ Adaptable |
| **ussumant/llm-wiki-compiler** | 277 | MD | ❌ No PDF | N/A | Poor (MD only) | Plugin-only | ❌ Low |
| **Astro-Han/karpathy-llm-wiki** | 978 | MD | ⚠️ Agent-dependent | ❌ No strategy | Medium | Skill-only | ⚠️ Adaptable |
| **kytmanov/obsidian-llm-wiki-local** | ~200 | Py | ❌ No PDF | N/A | Poor (MD only) | CLI-only | ❌ Low |
| **kiwifs/kiwifs** | 768 | Go | ❌ Storage-only | N/A | MD storage | MCP + REST + CLI + NFS | ✅ As backend |
| **Pratiyush/llm-wiki** | 280 | MD | ❌ No PDF | N/A | Session logs | CLI + MCP | ❌ Low |
| **tuirk/Kompl** | 64 | Py+JS | ❌ Gemini truncation | ❌ No chunking | Good | MCP + Docker | ⚠️ Moderate |
| **SwarmVault** (swarmvault.ai) | NEW | TS/JS | ⚠️ Agent-dependent | ✅ Schema-guided | **Excellent** | CLI + MCP + Web | ✅ **HIGH** |
| **Tencent/WeKnora** | 15.9k | Go+TS | ✅ Docling+MinerU parser | ✅ Multi-KB, RBAC | **Excellent** | CLI + MCP + Web + RAG | ✅ **HIGH** |
| **llmwikify** (PyPI) | NEW | Python | ⚠️ Agent-dependent | ✅ SQLite FTS5 | **Excellent** | CLI + MCP (20 tools) | ✅ **HIGH** |

### Most Promising for KNOX's Use Case

1. **lucasastorian/llmwiki** — Mistral OCR gives best PDF quality. MCP native. Worth testing.
2. **atomicstrata/llm-wiki-compiler** — Most polished MCP integration. Prompt budget for long docs.
3. **sage-wiki** — Already works. Needs better PDF pre-processing.
4. **Tencent/WeKnora** — Enterprise-grade wiki+RAG. Docling+MinerU parsing, RBAC.

## PDF→Markdown Pre-Processors

| Tool | Stars | License | Benchmark | Speed | Hybrid AI | LaTeX/Tables | MCP |
|---|---|---|---|---|---|---|---|
| **opendataloader-pdf** | 22.9k | Apache 2.0 | **0.907 (#1)** | 60+ p/s | Yes | Yes | Partial |
| **Docling** (IBM) | 60.7k | MIT | 0.882 | 0.76 s/p | No | Yes | Yes |
| **Marker** (VikParuchuri) | 35.7k | GPL-3.0 | 0.861 | 53 s/p | Yes | Yes | No |
| **MinerU** (opendatalab) | 65.7k | AGPL-3.0 | 0.831 | 5.96 s/p | Yes | Yes | Yes (15 tools) |
| **PyMuPDF4LLM** | ~5k | AGPL | 0.732 | 0.09 s/p | No | Partial | No |
| **markitdown** (Microsoft) | ~10k | MIT | 0.589 | 0.11 s/p | No | Basic | No |
| **GROBID** | ~3k | Apache 2.0 | N/A (XML/TEI) | N/A | No | Yes (Scientific) | No |
| **Nougat** (Meta) | ~8k | MIT | LaTeX-aware OCR | Slow | No | Yes | No |
| **GLM-OCR** (local) | NEW | — | Beats Gemini 3 Pro | Fast (0.9B) | No | Yes | No |
| **llm-ingest** (Amit-1of1) | NEW | Python | figure extraction | N/A | No | Yes | No |
| **paper2llm** (lacerbi) | NEW | Python | Academic PDFs | N/A | No | Yes | No |
| **file2md** (Pathwit) | 103 | MIT | 123 formats | Fast | No | Basic | No |

### KNOX Strategy

- **Primary:** Docling (MIT, 60.7k stars, integrated with KNOX as Tier 1 parser)
- **Alternative:** opendataloader-pdf (best benchmark, hybrid AI) — can be swapped via `BaseParser` interface
- **Legacy .doc:** Oxide Office (<https://office.oxide.fyi/>)
- **Images:** GLM-OCR for local VLM, Mistral OCR for best quality (paid)

## Chunking Libraries

| Library | Chunkers | Focus | Stars | License | KNOX Fit |
|---|---|---|---|---|---|
| **chonkie** | 11+ | Production RAG, multi-strategy | 4.1k+ | MIT | ✅ **Primary** |
| **semchunk** | 1 core | Semantic hierarchical | Low | — | ⚠️ Single strategy |

### chonkie Detail

| Chunker | Use Case | KNOX Application |
|---|---|---|
| `RecursiveChunker` | Hierarchical splitting (para → sentence → word) | Long text sections > token budget |
| `SemanticChunker` | Embedding-based topic boundaries | Multi-topic text sections |
| `TableChunker` | Row-wise with header preservation | Oversized tables (keep headers) |
| `CodeChunker` | AST-aware code splitting | Code blocks in docs |
| `SlumberChunker` | LLM-based agentic chunking | Complex mixed content (expensive) |

**Version:** v1.6.8 (Jun 1, 2026), SIMD-accelerated at 100+ GB/s, 45+ integrations.

**KNOX approach:** Custom structure-aware chunker (headings → atomic elements) + Chonkie for text-heavy groups. See [architecture](../architecture/index.md) for full design.

## Hybrid RAG Components (Sidecar)

| Tool | Type | Stars | License | Purpose |
|---|---|---|---|---|
| **alibaba/zvec** | Vector DB | NEW | Apache 2.0 | Lightweight in-process vector store |
| **tobi/qmd** | CLI search | ~200 | MIT | Local full-text search over wiki corpus |
| **VectifyAI/PageIndex** | Doc index | 32.3k | — | Vectorless, reasoning-based RAG. 98.7% FinanceBench |

### Open-WebUI (GUI)

No custom GUI. Use **open-webui/open-webui** as the query/work interface. KNOX exposes API/MCP; Open-WebUI connects.

## MCP Servers in This Space

| Server | Features | KNOX Relevance |
|---|---|---|
| **kObsidian** (bezata) | 66 tools, filesystem-first vault | Inspiration for Hermes integration |
| **llmwiki serve** (atomicstrata) | `wiki_compile`, `get_context_pack`, `wiki_query` | Competitor + reference |
| **Tencent/WeKnora** | Enterprise RAG + reasoning + wiki | Competitor + reference |
| **KnowledgeBaseMCP** | 9 tools, PDF/DOCX/PPTX/XLSX | Alternative parser integration point |
| **Docling MCP** (IBM) | Document → markdown via MCP | Primary parser's native MCP |
| **MinerU-Document-Explorer** | 15 tools, 3 groups | Alternative parser + tools |
| **PageIndex MCP** (VectifyAI) | Vectorless RAG via MCP | Sidecar integration point |

## Gaps Confirmed (Not Solved by Any Tool)

1. **Smart chunking with semantic boundaries** — no framework uses libraries like chonkie or semchunk natively
2. **Structured JSON ingestion** — Docling outputs JSON with bounding boxes, table schemas. No wiki compiler consumes this
3. **Multi-turn resolution workflows** — abbreviation expansion, cross-reference validation during ingestion. Not supported
4. **Multimodal figure analysis** — extracted images are not sent to vision LLMs during compilation
5. **Context-window-aware ingestion** — no framework auto-detects model limits and adjusts chunking
6. **Headless operation** — nashsu/llm_wiki is GUI-only. Several require web UI for full function
7. **WiCER-style compilation loop** — no tool implements compile-evaluate-refine. All one-shot
8. **Model context pre-check** — no tool queries model metadata (context length, reasoning mode) before operations
9. **Hybrid text+structured chunking** — chonkie works on text; no framework combines with typed document elements

## Document Conversion Catalog

### PDF → Markdown

| Tool | Benchmark | Speed | Notes |
|---|---|---|---|
| **opendataloader-pdf** | **0.907 (#1)** | 60+ p/s | Hybrid AI, PDF/UA auto-tagging |
| **Docling** | 0.882 | 0.76 s/p | Layout detection, reading order, MCP server |
| **Marker** | 0.861 | 53 s/p | LaTeX equations, hybrid LLM mode |
| **MinerU** | 0.831 | 5.96 s/p | Formula→LaTeX, Table→HTML, 109-lang OCR |
| **PyMuPDF4LLM** | 0.732 | 0.09 s/p | Fast but minimal structure |
| **markitdown** | 0.589 | 0.11 s/p | Quick fallback, loses structure |

### DOCX/DOC → Markdown

| Tool | Notes |
|---|---|
| **pandoc** | Swiss army knife. Tables, footnotes, citations. Fallback for all formats. |
| **mammoth** | Style preservation via python-docx |
| **markitdown** | Lightweight. Quick conversion. |
| **LibreOffice CLI** | `soffice --headless --convert-to md` |
| **Oxide Office** | Legacy .doc, .xls, .ppt pre-Office 2007 |

### XLSX/Spreadsheet → Markdown

| Tool | Notes |
|---|---|
| **pandas + tabulate** | `df.to_markdown()`. Best for programmatic. |
| **markitdown** | Sheet-by-sheet table conversion |
| **xlsx2md** | Specialized tool |

### PPTX → Markdown

| Tool | Notes |
|---|---|
| **pandoc** | `pandoc input.pptx -o output.md` |
| **python-pptx** | Programmatic slide text extraction |
| **markitdown** | Quick slide-by-slide |

### Images (with text/figures) → Markdown

| Tool | Notes |
|---|---|
| **Mistral OCR** | Best quality, paid API (€30/1k pages) |
| **Tesseract + layout parsing** | Open source, needs pipeline |
| **Docling** | Vision LLM fallback for figures |
| **opendataloader-pdf** | Hybrid mode with OCR + picture description |
| **GLM-OCR** | 0.9B local model, zero GPU, competitive quality |

### EPUB → Markdown

| Tool | Notes |
|---|---|
| **pandoc** | `pandoc input.epub -o output.md` |
| **calibre (ebook-convert)** | `ebook-convert input.epub output.md` |

## References

- sage-wiki: <https://github.com/xoai/sage-wiki>
- llm-wiki-compiler: <https://github.com/atomicstrata/llm-wiki-compiler>
- llmwiki: <https://github.com/lucasastorian/llmwiki>
- nashsu/llm_wiki: <https://github.com/nashsu/llm_wiki>
- Tencent/WeKnora: <https://github.com/Tencent/WeKnora>
- Docling: <https://github.com/docling-project/docling>
- opendataloader-pdf: <https://github.com/opendataloader-project/opendataloader-pdf>
- Marker: <https://github.com/datalab-to/marker>
- MinerU: <https://github.com/opendatalab/MinerU>
- Chonkie: <https://github.com/feyninc/chonkie>
- zvec: <https://github.com/alibaba/zvec>
- qmd: <https://github.com/tobi/qmd>
- open-webui: <https://github.com/open-webui/open-webui>
- Oxide Office: <https://office.oxide.fyi/>
- VectifyAI/PageIndex: <https://github.com/VectifyAI/PageIndex>
- WiCER paper: <https://arxiv.org/pdf/2605.07068v1>
- Karpathy's Gist: <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
