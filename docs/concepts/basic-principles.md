---
icon: lucide/lightbulb
---

# Basic Principles

KNOX is a compiled LLM-wiki system for scientific and technical documents.
This page explains what problem it solves, what makes it different, and the
core principles that guide its design.

## What Problem Does KNOX Solve?

Raw documents (PDFs, DOCX, web pages) are not directly queryable by LLMs.
Existing RAG pipelines flatten them to text, losing tables, figures, equations,
and reading order. KNOX preserves structure through a 5-tier pipeline and
compiles sources into an interlinked, queryable knowledge wiki.

``` mermaid
flowchart LR
    A[Raw Sources] --> B[Tier 1<br>Preprocessing]
    B --> C[Tier 2<br>Chunking]
    C --> D[Tier 3<br>Resolution]
    D --> E[Tier 4<br>Compilation]
    E --> F[Tier 5<br>Output]
    F --> G[Compiled Wiki]
```

## How Documents Flow Through the Pipeline

``` mermaid
sequenceDiagram
    participant S as Source
    participant P as Parser
    participant H as HybridChunker
    participant R as Resolver
    participant C as Compiler
    participant W as Writer

    S->>P: PDF / DOCX / URL
    P->>P: Extract tables, figures, equations
    P->>H: ParsedDocument (structured JSON)
    H->>H: Group by heading, never split atomic elements
    H->>R: list[Chunk]
    R->>R: Expand abbreviations, resolve cross-refs
    R->>C: list[ResolvedChunk]
    C->>C: Compile → Evaluate → Diagnose → Refine
    C->>W: CompiledWiki
    W->>W: Markdown / JSON / Typst
    W->>G: wiki/ (Obsidian vault)
```

## KNOX vs. Flatten-and-RAG

``` mermaid
flowchart TD
    subgraph Traditional[RAG Pipeline]
        T1[PDF] --> T2[Flatten to text]
        T2 --> T3[Chunk by char count]
        T3 --> T4[Embed + Store]
        T4 --> T5[Vector Search]
        T5 --> T6[LLM Answer]
    end

    subgraph KNOX[KNOX Pipeline]
        K1[PDF] --> K2[Structure-preserving parse]
        K2 --> K3[Hybrid chunking]
        K3 --> K4[Resolution + VLM]
        K4 --> K5[WiCER Compilation]
        K5 --> K6[Compiled Wiki]
        K6 --> K7[Query + Synthesis]
    end

    Traditional -->|Loses tables, figures, equations| F1[Flat text answer]
    KNOX -->|Preserves structure, cross-refs| F2[Cited wiki answer]
```

## The 8 Core Principles

### 1. Parser-Agnostic

No parser is baked into KNOX's core. The `BaseParser` interface accepts raw
documents and produces `ParsedDocument`. The `ParserRouter` selects by file
extension and user preference. If a new parser emerges, add a module
implementing `BaseParser` — no other tier changes.

See [`architecture.md`](../architecture/index.md#1-parser-agnostic) for details.

### 2. Structure-Preserving

Flattening a 50-row comparison table to markdown destroys queryability. KNOX
stores tables as structured JSON (`list[list[str]]` with cell type annotations).
Figures retain captions, bounding boxes, and extracted PNGs. Equations keep
LaTeX source and image fallback.

See [`architecture.md`](../architecture/index.md#2-structure-preserving) for details.

### 3. Model-Aware

litellm provides provider abstraction, but KNOX goes further:

- Auto-detect context window from model metadata
- Reserve budget for system prompts (20%)
- Cheap local model for pre-processing
- Expensive/remote model only for WiCER compilation
- Fallback chain: local → remote → error

See [`architecture.md`](../architecture/index.md#3-model-aware) for details.

### 4. Agent-Native

Three interfaces, same core:

- **CLI:** `knox compile sources/ --output wiki/`
- **Python SDK:** `pipeline.compile(sources=[Path("paper.pdf")])`
- **MCP Server:** tools for Hermes, Claude Code, Codex

No GUI. No web server required. Headless cron-friendly.

See [`architecture.md`](../architecture/index.md#4-agent-native) for details.

### 5. Resolution-Ready

Default: one-shot ingestion (fast). Opt-in per document: multi-turn resolution
(accurate). Resolvers are pluggable via MCP or CLI tools:

- Abbreviation expansion
- Cross-reference validation
- Term disambiguation

See [`architecture.md`](../architecture/index.md#5-resolution-ready) for details.

### 6. Multimodal

Vision-capable LLMs analyze extracted figures during compilation. The
`VLMEnricher` sends figure images + surrounding text + caption to a vision
model. Output: searchable caption + classification. Figures become first-class
wiki nodes.

See [`architecture.md`](../architecture/index.md#6-multimodal) for details.

### 7. Iterative Compilation (WiCER)

Inspired by [arXiv:2605.07068v1](https://arxiv.org/pdf/2605.07068v1). Blind
compilation over-compresses 2–3× beyond targets. WiCER recovers ~80% of lost
quality in 1–2 iterations via probe Q&A and targeted diagnosis.

See [`concepts/wicer-loop.md`](wicer-loop.md) for a deep dive.

### 8. Hybrid Chunking

Structure-aware first: tables/figures/equations never split. Heading hierarchy
groups elements. Only text-heavy sections get Chonkie treatment. This preserves
queryability of structured data while leveraging Chonkie's mature algorithms
(Recursive, Semantic, Table, Code) for free-form text.

See [`architecture.md`](../architecture/index.md#8-hybrid-chunking) for details.

## Further Reading

| Document | Purpose |
|----------|---------|
| [Architecture](../architecture/index.md) | Full 5-tier design, data models, integration points |
| [Pipeline Tiers](pipeline-tiers.md) | Deep-dive per tier with class references |
| [WiCER Loop](wicer-loop.md) | Iterative compilation explainer with state diagram |
| [Decision Log](../architecture/decisions.md) | All architectural decisions and rationale |
