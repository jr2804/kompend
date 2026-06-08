# KNOX Roadmap

## Milestones

### Milestone 0: Project Scaffolding ✅

| Item | Status |
|---|---|
| Copier template applied | ✅ (`copier-uv-plus` v1.3.0) |
| Project structure (5 tiers) | ✅ |
| CLI (`knox --help`) | ✅ |
| Tests passing | ✅ (8/8) |
| Git initialized | ✅ |
| Docs (overview, architecture, ecosurvey, roadmap, decisions) | 🔄 This sprint |
| beads issue tracker setup | ⏳ Pending |

### Milestone 1: Docling Parser (Tier 1) ✅

| Item | Status |
|---|---|
| `DoclingParser` with structure-preserving extraction | ✅ |
| Table → `list[list[str]]` with metadata | ✅ |
| Figure → extracted PNG + caption + bbox | ✅ |
| Equation → LaTeX source | ✅ |
| Configurable image mode (base64 / file) | ✅ |
| Metadata extraction (title, author, pages, language) | ✅ |
| Tests (mock-based) | ✅ (3/3) |

**Next:** Extend to DOCX/PPTX/XLSX native parsing (via Docling multi-format support).

### Milestone 2: Hybrid Chunking (Tier 2) 🎯 Current

| Item | Priority | Status |
|---|---|---|
| `BaseChunker` interface | P0 | ✅ Stub |
| Structure-aware chunker (headings → atomic elements) | P0 | 🔄 Design |
| Model token limit detection via litellm | P1 | 🔄 |
| Chonkie integration (Recursive, Semantic, Table, Code) | P0 | 🔄 |
| Budget management (20% reserve for prompts) | P1 | 🔄 |
| Tests | P0 | 🔄 |

**Target:** Produce `list[Chunk]` from `ParsedDocument` with structure preserved.

### Milestone 3: LLM Resolution (Tier 3)

| Item | Priority | Status |
|---|---|---|
| `BaseResolver` interface | P1 | ✅ Stub |
| Abbreviation expansion (one-shot default) | P2 | ⏳ |
| Cross-reference validation | P2 | ⏳ |
| Term disambiguation via MCP | P2 | ⏳ |
| VLM enrichment for extracted figures | P2 | ⏳ |
| Multi-turn mode (opt-in per document) | P2 | ⏳ |

**Target:** Enrich `Chunk` → `ResolvedChunk` with entities, relationships, confidence.

### Milestone 4: WiCER Compilation Loop (Tier 4)

| Item | Priority | Status |
|---|---|---|
| `BaseCompiler` interface | P0 | ✅ Stub |
| Blind compilation (one-shot) | P0 | ⏳ |
| Probe Q&A generation | P0 | ⏳ |
| LLM-as-judge evaluation | P0 | ⏳ |
| Targeted diagnosis (extract critical facts) | P0 | ⏳ |
| Refine with pinned facts | P0 | ⏳ |
| Convergence detection / max iterations | P1 | ⏳ |
| Tests | P0 | ⏳ |

**Target:** Compiled wiki quality within 80% of full-context baseline after 1–2 iterations.

### Milestone 5: Output Writers (Tier 5)

| Item | Priority | Status |
|---|---|---|
| Markdown writer (Obsidian vault) | P0 | ✅ Stub |
| JSON graph writer (SQLite) | P1 | ⏳ |
| Typst/PDF writer (academic output) | P2 | ⏳ |
| Output format configurable | P0 | 🔄 |

### Milestone 6: MCP Server

| Item | Priority | Status |
|---|---|---|
| MCP server scaffold | P1 | ✅ Stub |
| `parse_document` tool | P1 | ⏳ |
| `compile_wiki` tool | P1 | ⏳ |
| `query_wiki` tool | P2 | ⏳ |
| Thin CLI wrapper (no business logic duplication) | P1 | ⏳ |

### Milestone 7: Hybrid Sidecar RAG (Optional)

| Item | Priority | Status |
|---|---|---|
| zvec integration (embedded vector DB) | P3 | ⏳ |
| qmd integration (local CLI search) | P3 | ⏳ |
| Open-WebUI integration guide | P3 | ⏳ |

### Milestone 8: Online Sources

| Item | Priority | Status |
|---|---|---|
| HTTP file downloader (direct URL → sources/) | P0 | ✅ |
| Web crawler (crawl4ai for documentation sites) | P1 | 🔄 |
| URL batch processing | P1 | ⏳ |
| Duplicate detection (URL → SHA256) | P1 | ⏳ |

### Milestone 9: Parser Extensions

| Item | Priority | Status |
|---|---|---|
| PDF via opendataloader-pdf (alternative to Docling) | P1 | ⏳ |
| Legacy .doc/.xls/.ppt via Oxide Office | P2 | ⏳ |
| Image → text via GLM-OCR / VLM | P2 | ⏳ |
| Convert-anything-to-PDF-first strategy evaluation | P3 | ⏳ |

### Milestone 10: Testing & Benchmarking

| Item | Priority | Status |
|---|---|---|
| Test corpus (listening-effort papers) | P0 | ⏳ |
| Parse quality benchmark vs Docling baseline | P1 | ⏳ |
| Chunk quality benchmark (token efficiency) | P1 | ⏳ |
| Compilation quality benchmark (WiCER probes) | P1 | ⏳ |
| Per-query latency benchmark vs RAG | P2 | ⏳ |

## Open Questions

1. **Multi-format parsing:** Should KNOX route PDFs through Docling and DOCX through pandoc, or should everything go through a unified PDF-first pipeline?
2. **Context window minimum:** 32K tokens works for local models. Should the default chunk budget be model-dependent?
3. **VLM for figure analysis:** Run locally (GLM-OCR, 0.9B) or via API (Gemini, Claude)? Cost vs. quality tradeoff.
4. **WiCER convergence:** Max 3 iterations or quality threshold based? How to balance compilation cost vs. quality?
5. **Output format default:** Markdown for Obsidian, JSON for programmatic queries, or both always?
6. **Credential management:** URLs requiring auth (IEEE, ACM) — how to handle cookies/API keys securely?

## Decision Log

See [decision log](../architecture/decisions.md) for all decisions made and why.

## Weekly Research

- Worky's weekly survey continues in Obsidian vault: `Worky/research/llm-wiki/weekly/`
- New tools discovered → update `ecosurvey.md`
- Benchmark results → update this roadmap

## Target Dates (Tentative)

| Milestone | Estimated |
|---|---|
| M2: Hybrid chunking | Week of Jun 8 |
| M3: LLM resolution | Week of Jun 15 |
| M4: WiCER compilation | Week of Jun 22 |
| M5: Output writers | Week of Jun 29 |
| M6: MCP server | Week of Jul 6 |
| M10: Testing & benchmark | Week of Jul 13 |
| First public release | Mid-Jul 2026 |

## References

- Worky's research directory: `Worky/research/llm-wiki/` (Obsidian vault)
- Weekly reports: `Worky/research/llm-wiki/weekly/`
- Testing corpus: `Worky/research/listening-effort/` (local only, never committed)
