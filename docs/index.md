---
icon: lucide/book-open
---

# KNOX

## Knowledge Organizer for eXtended documents

A compiled LLM-wiki system for scientific and technical documents.
Parse, chunk, resolve, and compile PDFs, DOCX, spreadsheets, and
web sources into an interlinked, queryable knowledge base.

---

## 5-Tier Pipeline

KNOX follows a parser-agnostic, structure-preserving pipeline:

``` mermaid
graph LR
  T1[Preprocessing] --> T2[Chunking]
  T2 --> T3[Resolution]
  T3 --> T4[Compilation]
  T4 --> T5[Output]
```

### Quick Start

```bash
# Initialize a project
knox init ./my-wiki && cd ./my-wiki

# Compile documents into a wiki
knox compile *.pdf

# Query the compiled wiki
knox query "what is the main result?"
```

### Key Features

- **Parser-agnostic** — Docling (PDF), Pandoc (DOCX/EPUB/HTML), crawl4ai (web)
- **Hybrid chunking** — Atomic elements + recursive token-aware splitting
- **VLM enrichment** — Image/figure captioning via vision-language models (`--vlm-model`)
- **Typed knowledge graph** — Entity relations with confidence (`uses`, `depends_on`, `contradicts`, …)
- **Memory lifecycle** — Confidence decay, tier promotion (working→episodic→semantic→procedural), supersession
- **Hybrid search** — FTS + vector + token-overlap fused with reciprocal rank fusion (RRF)
- **Crystallize** — Synthesize captured notes into coherent wiki pages
- **Event hooks** — Pluggable callbacks at compile/capture/query/lint lifecycle points
- **WiCER compilation** — Iterative compile-evaluate-refine loop for quality
- **Multi-format output** — Markdown (incl. Obsidian vault), JSON, Typst
- **LLM-powered linting** — 8 structural rules + 4 LLM rules (incl. contradiction detection)

---

## Documentation

- [Getting Started](guides/cli.md) — CLI setup, compile/query examples
- [Initialization](guides/initialization.md) — `knox init` project scaffolding
- [Configuration](guides/configuration.md) — 5-layer config system, `knox config` subcommands
- [Basic Principles](concepts/basic-principles.md) — What KNOX does and why
- [Architecture](architecture/index.md) — 5-tier pipeline design
- [Pipeline Tiers](concepts/pipeline-tiers.md) — Per-tier component reference
- [Memory Lifecycle](concepts/memory-lifecycle.md) — Confidence, tiers, decay, supersession
- [Query Memory](guides/query-memory.md) — Episodic memory, context synthesis, and MCP integration
- [Crystallize](concepts/crystallize.md) — Captured notes → synthesized wiki pages
- [Compilation Guide](guides/compilation.md) — Compilers, VLM, relations, memory, hooks
- [Linting Guide](guides/lint.md) — Structural + LLM-powered rules
- [Trust & Memory](guides/trust.md) — Provenance, memory lifecycle, decay
- [Wiki Structure](guides/wiki-structure.md) — index.md, log.md, overview.md
- [Caching](caching.md) — Multi-level content-addressed caching
- [Cache Layout](architecture/cache.md) — `.knox/` directory layout
- [Status](about/status.md) — Feature completion matrix
- [Roadmap](about/roadmap.md) — Milestones and upcoming work
- [Ecosystem](reference/ecosystem.md) — LLM-wiki tool survey
- [Concepts](concepts/basic-principles.md) — Pipeline tiers, WiCER loop, design rationale
- [Reference](reference/ecosystem.md) — Ecosystem survey of LLM-wiki tools
