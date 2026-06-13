# KNOX Roadmap

## Completed Milestones

### Milestone 0: Project Scaffolding ✅

| Item | Status |
|---|---|
| Copier template applied | ✅ (`copier-uv-plus` v1.3.0) |
| Project structure (5 tiers) | ✅ |
| CLI (`knox --help`) | ✅ |
| Tests passing | ✅ |
| Git initialized | ✅ |
| Docs (overview, architecture, ecosystem, roadmap, decisions) | ✅ |
| beads issue tracker setup | ✅ |

### Milestone 1: Docling Parser (Tier 1) ✅

| Item | Status |
|---|---|
| `DoclingParser` with structure-preserving extraction | ✅ |
| Table → `list[list[str]]` with metadata | ✅ |
| Figure → extracted PNG + caption + bbox | ✅ |
| Equation → LaTeX source | ✅ |
| Configurable image mode (base64 / file) | ✅ |
| Metadata extraction (title, author, pages, language) | ✅ |
| Additional parsers (pandoc, pandas, HTTP, crawl4ai) | ✅ |

### Milestone 2: Hybrid Chunking (Tier 2) ✅

| Item | Status |
|---|---|
| `BaseChunker` interface | ✅ |
| Structure-aware chunker (headings → atomic elements) | ✅ |
| Chonkie integration (HybridChunker) | ✅ |
| Budget management | ✅ |
| Tests | ✅ |

### Milestone 3: LLM Resolution (Tier 3) ✅

| Item | Status |
|---|---|
| `BaseResolver` interface | ✅ |
| `LLMResolver` — typed relation extraction with confidence | ✅ |
| `VLMEnricher` — vision enrichment for figures | ✅ |
| Wired into CLI (`--vlm-model`, `--resolve-relations`) and MCP | ✅ |
| Resolver chain accumulation | ✅ |

### Milestone 4: Compilation (Tier 4) ✅

| Item | Status |
|---|---|
| `BaseCompiler` interface | ✅ |
| `GraphCompiler` — heading-based grouping, page-kind inference, typed edges | ✅ |
| `TwoPhaseCompiler` — extract-then-generate concept pages | ✅ |
| `WiCERCompiler` — iterative compile-evaluate-refine (CEGAR) loop | ✅ |
| Typed knowledge-graph relations with confidence | ✅ |
| Built-in page kinds (overview/comparison/entity/concept) | ✅ |
| Tests | ✅ |

### Milestone 5: Output Writers (Tier 5) ✅

| Item | Status |
|---|---|
| Markdown writer (Obsidian vault, multiple flavors) | ✅ |
| JSON graph writer | ✅ |
| Typst/PDF writer | ✅ |
| Output format configurable | ✅ |
| Frontmatter + page-kind annotations | ✅ |

### Milestone 6: MCP Server ✅

| Item | Status |
|---|---|
| MCP server scaffold (FastMCP) | ✅ |
| `parse_document` tool | ✅ |
| `compile_wiki` tool | ✅ |
| `wiki_capture` / `wiki_learn` tools | ✅ |
| VLM model parameter | ✅ |

### Milestone 7: Hybrid Sidecar RAG ✅

| Item | Status |
|---|---|
| zvec integration (embedded vector DB) | ✅ |
| SentenceTransformer / LiteLLM embedding providers | ✅ |
| `HybridRetriever` — FTS + vector + token + raw fusion with RRF | ✅ |
| Query wired to vector store | ✅ |

### Milestone 8: Online Sources ✅

| Item | Status |
|---|---|
| HTTP file downloader | ✅ |
| Web crawler (crawl4ai) | ✅ |
| URL batch processing | ✅ |
| Duplicate detection (URL → SHA256) | ✅ |

## Recent Feature Additions

### Memory Lifecycle ✅

| Item | Status |
|---|---|
| Four-tier consolidation (WORKING → EPISODIC → SEMANTIC → PROCEDURAL) | ✅ |
| Confidence scoring with asymptotic evolution | ✅ |
| Retention decay (exponential, configurable half-life) | ✅ |
| Supersession detection (same title, different ID) | ✅ |
| Lint integration (stale/decayed/superseded pages) | ✅ |

### Event-Driven Automation Hooks ✅

| Item | Status |
|---|---|
| `HookRegistry` with exception isolation | ✅ |
| Events: PRE_COMPILE, POST_COMPILE, ON_CAPTURE, ON_QUERY, ON_LINT | ✅ |
| Config-driven loading (`[hooks]` in `knox.toml`) | ✅ |

### Capture and Crystallize ✅

| Item | Status |
|---|---|
| `capture` — LLM-assisted extraction from notes/conversations | ✅ |
| `learn` — store verbatim facts | ✅ |
| `crystallize` — synthesize captured notes into wiki source pages | ✅ |

### Lint Rules ✅

| Item | Status |
|---|---|
| 8 structural rules (broken links, orphans, duplicates, etc.) | ✅ |
| 4 LLM-powered rules (factual-consistency, clarity, coverage, contradiction) | ✅ |
| Self-healing auto-fix for broken links and frontmatter | ✅ |

## Upcoming Milestones

### Milestone 9: Parser Extensions

| Item | Priority | Status |
|---|---|---|
| Legacy .doc/.xls/.ppt via Oxide Office | P2 | ⏳ |
| Image → text via VLM/OCR | P2 | ⏳ |
| Convert-anything-to-PDF-first strategy evaluation | P3 | ⏳ |

### Milestone 10: Testing & Benchmarking

| Item | Priority | Status |
|---|---|---|
| Test corpus (listening-effort papers) | P0 | ✅ |
| Parse quality benchmark vs Docling baseline | P1 | ⏳ |
| Chunk quality benchmark (token efficiency) | P1 | ⏳ |
| Compilation quality benchmark (WiCER probes) | P1 | ⏳ |
| Per-query latency benchmark vs RAG | P2 | ⏳ |
| Eval harness for automated regression testing | P1 | ⏳ |

### Milestone 11: Quality & Polish

| Item | Priority | Status |
|---|---|---|
| Move all LLM prompts into Jinja2 templates (T1–T10) | P1 | ⏳ |
| Real token-based budget for HybridChunker | P2 | ⏳ |
| MCP protocol upgrade (JSON-RPC 2.0) | P2 | ⏳ |
| DoclingParser update to latest API | P2 | ⏳ |
| Remove all PLC0415 inline-import violations | P1 | ⏳ |

## Open Questions

1. **Multi-format parsing:** Should KNOX route PDFs through Docling and DOCX through pandoc, or unify on a PDF-first pipeline?
2. **Context window minimum:** 32K tokens works for local models. Should the default chunk budget be model-dependent?
3. **VLM for figure analysis:** Run locally (GLM-OCR, 0.9B) or via API (Gemini, Claude)? Cost vs. quality tradeoff.
4. **WiCER convergence:** Max 3 iterations or quality threshold based? How to balance compilation cost vs. quality?
5. **Eval harness:** What benchmark suite should KNOX use for regression testing?

## Decision Log

See [decision log](../architecture/decisions.md) for all decisions made and why.

## References

- Testing corpus: `Worky/research/listening-effort/` (local only, never committed)
- beads issue tracker: `.beads/` in project root
