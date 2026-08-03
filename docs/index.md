---
---

# KNOX

## Knowledge Organizer for eXtended documents

A compiled LLM-wiki system for scientific and technical documents.
Parse, chunk, resolve, and compile PDFs, DOCX, spreadsheets, and
web sources into an interlinked, queryable knowledge base.

Unlike most tools in this space, KNOX is provider-agnostic, adapts
to whatever LLM you have — from a local 8B model to a frontier API —
and uses structure-preserving parsing so technical content (tables,
equations, figures) survives extraction intact. [Read why →](about/motivation.md)

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

- **Parser-agnostic** — 14 formats via Docling + 5 legacy/text formats via Pandoc + binaryornot text fallback. [See full breakdown](guides/document-extraction.md)
- **Hybrid chunking** — Atomic elements + recursive token-aware splitting
- **VLM enrichment** — Image/figure captioning via vision-language models (`--vlm-model`)
- **Typed knowledge graph** — Entity relations with confidence (`uses`, `depends_on`, `contradicts`, …)
- **Memory lifecycle** — Confidence decay, tier promotion (working→episodic→semantic→procedural), supersession
- **Hybrid search** — FTS + vector + token-overlap fused with reciprocal rank fusion (RRF)
- **Distill** — Synthesize captured notes into coherent wiki pages
- **Event hooks** — Pluggable callbacks at compile/capture/query/lint lifecycle points
- **WiCER compilation** — Iterative compile-evaluate-refine loop for quality
- **Multi-format output** — [OKF v0.2](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) (default), Markdown (incl. Obsidian vault), JSON, Typst
- **LLM-powered linting** — 8 structural rules + 4 LLM rules (incl. contradiction detection)

---

## Documentation

### Getting Started

- [Motivation](about/motivation.md) — Why KNOX exists and what makes it different
- [Getting Started](guides/cli.md) — CLI setup, compile/query examples
- [Initialization](guides/initialization.md) — `knox init` project scaffolding
- [Configuration](guides/configuration.md) — 5-layer config system, `knox config` subcommands
- [Basic Principles](concepts/basic-principles.md) — What KNOX does and why
- [Document Extraction](guides/document-extraction.md) — Supported formats, parser selection, framework decisions

### Pipeline & Architecture

- [Architecture](architecture/index.md) — 5-tier pipeline design
- [Pipeline Tiers](concepts/pipeline-tiers.md) — Per-tier component reference
- [Decision Log](architecture/decisions.md) — Architectural decisions and rationale
- [Caching](reference/caching.md) — Multi-level content-addressed caching
- [Cache Layout](architecture/cache.md) — `.knox/` directory layout
- [Output Architecture](architecture/output.md) — Output writers, renderer, flavor system

### Compilation & Output

- [Compilation Guide](guides/compilation.md) — Compilers, VLM, relations, memory, hooks
- [Output Configuration](guides/output.md) — Markdown flavors, frontmatter, formatting
- [Assets (Attachments)](guides/assets.md) — Supplementary files per source, opt-in ingestion
- [WiCER Loop](concepts/wicer-loop.md) — Iterative compile-evaluate-refine quality loop
- [Review Queue](guides/review.md) — Human-in-the-loop approval for wiki pages
- [Wiki Structure](guides/wiki-structure.md) — index.md, log.md, overview.md

### LLM Configuration

- [LLM Configuration](guides/llm-config.md) — Provider setup, role routing, failover chains
- [PydanticAI Agents & Skills](reference/pydantic-ai-agents.md) — Agent architecture, per-role models, skill configuration
- [Model Context Adaptation](guides/model-context.md) — Context window discovery, adaptive sizing

### Linting & Quality

- [Linting Guide](guides/lint.md) — Structural + LLM-powered rules
- [Tracing](guides/trace.md) — Trace wiki pages back to their source files
- [Trust & Memory](guides/trust.md) — Provenance, memory lifecycle, decay
- [Dedup](guides/dedup.md) — Content-level duplicate detection and suppression

### Memory & Knowledge

- [Memory Lifecycle](concepts/memory-lifecycle.md) — Confidence, tiers, decay, supersession
- [Query Memory](guides/query-memory.md) — Episodic memory, context synthesis, and MCP integration
- [Distill](concepts/distill.md) — Captured notes → synthesized wiki pages

### Integrations

- [MCP Server](architecture/index.md#4-agent-native) — Agent integration via MCP
- [ACP Agent](integrations/acp-agent.md) — Editor integration via Agent Client Protocol
- [Open-WebUI](integrations/open-webui.md) — Connecting KNOX with Open-WebUI
- [Ecosystem](reference/ecosystem.md) — LLM-wiki tool survey

### Project & Community

- [Architecture Decisions](architecture/decisions.md) — Design choices and rationale
- [Acknowledgements](about/acknowledgements.md) — Credits and investigated projects
- **Active Work** — See [beads issue tracker](https://github.com/Hermes/knox) in `.beads/` for milestones, active tasks, and project status
