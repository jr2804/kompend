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

## CLI Commands vs. Pipeline Tiers

KNOX exposes the pipeline through discrete CLI commands. Each command runs a
specific subset of tiers:

``` mermaid
flowchart TB
    subgraph Init["knox init"]
        I1[Scaffold project<br>sources/ wiki/ .knox/]
    end

    subgraph Ingest["knox ingest (no LLM)"]
        IG1[Tier 0: Download URLs]
        IG2[Tier 1: Parse all sources]
        IG1 --> IG2
    end

    subgraph Compile["knox compile (needs LLM)"]
        C1[Tier 2: Chunk<br>adaptive sizing from<br>LLM context window]
        C2[Tier 3: Resolve<br>relations + VLM]
        C3[Tier 4: Compile<br>extract + generate]
        C4[Tier 5: Output<br>markdown / json / typst]
        C1 --> C2 --> C3 --> C4
    end

    subgraph Post["Post-compile"]
        L1["knox lint<br>structural + LLM rules"]
        Q1["knox query<br>semantic search"]
    end

    Init --> Ingest
    Ingest --> Compile
    Compile --> Post
```

| Command | Tiers | Needs LLM? | What it produces |
|---------|-------|------------|------------------|
| `knox init` | — | No | Empty project scaffold |
| `knox parse` | 1 only | No | Structured JSON per file (debugging) |
| `knox ingest` | 0–1 | **No** | Parsed documents cached in `.knox/cache/`; deterministic, no API costs |
| `knox compile` | 2–5 | **Yes** | Compiled wiki pages on disk; chunking uses LLM context window for adaptive sizing |
| `knox lint` | — | Optional | Quality report + auto-fixes |
| `knox query` | — | Optional | Semantic search over indexed wiki |

**The key boundary is between Tier 1 (parse) and Tier 2 (chunk).**
Tiers 0–1 (download + parse) are deterministic — no LLM calls, no API
costs. Tier 2 and above (chunk, resolve, compile, output) require a
configured LLM. Chunking needs the model's context window for adaptive
sizing; resolution and compilation need the model for inference.

`knox ingest` stops after Tier 1. It downloads URLs and parses every
source into structured JSON cached in `.knox/cache/`. This is fast,
free, and idempotent — you can re-run it anytime without LLM costs or
API keys.

`knox compile` picks up where ingest left off (or starts from scratch
if you skip ingest). It runs Tiers 2–5: chunking (with adaptive sizing
from the LLM's discovered context window), resolving relations,
compiling pages, and writing output files. This is where the LLM work
happens.

Splitting ingest from compile lets you:

- **Iterate on parsing/chunking** without burning LLM tokens.
- **Swap LLM models** between runs without re-parsing everything.
- **Debug pipeline issues** at the exact tier where they occur.
- **Run ingest in CI** (no API keys needed) and compile only when sources change.

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

Most document pipelines treat PDFs and DOCX as text blobs: flatten to
plain text, split at an arbitrary character count, embed, and vector-search.
This works for prose but destroys structured content — tables lose their
rows and columns, figures lose captions, equations become unreadable glyphs,
and cross-references vanish.

KNOX takes a different approach: structure-preserving parsing keeps tables
as `list[list[str]]`, figures as captioned images, equations as LaTeX source.
Hybrid chunking never splits atomic elements. Multi-turn resolution expands
abbreviations and resolves cross-references before compilation turns everything
into an interlinked wiki. Querying this wiki retrieves *relevant context*, not
isolated text snippets.

``` mermaid
flowchart TB
    subgraph Traditional[RAG Pipeline]
        direction TB
        T1["PDF"] --> T2["Flatten to text<br/><i>loses tables, figures, equations, structure</i>"]
        T2 --> T3["Chunk by char count<br/><i>splits tables mid-row, breaks code blocks</i>"]
        T3 --> T4["Embed + Store<br/><i>no provenance, no cross-refs</i>"]
        T4 --> T5["Vector Search<br/><i>retrieves isolated text snippets</i>"]
        T5 --> T6["LLM Answer<br/><i>flat text, hallucinations, no citations</i>"]
    end

    subgraph KNOX[KNOX Pipeline]
        direction TB
        K1["PDF"] --> K2["Structure-preserving parse<br/><i>tables, figures, equations as first-class objects</i>"]
        K2 --> K3["Hybrid chunking<br/><i>atomic elements never split; headings group content</i>"]
        K3 --> K4["Resolution + VLM<br/><i>expand abbrevs, resolve cross-refs, caption images</i>"]
        K4 --> K5["WiCER Compilation<br/><i>extract -> evaluate -> diagnose -> refine</i>"]
        K5 --> K6["Compiled Wiki<br/><i>interlinked pages with provenance</i>"]
        K6 --> K7["Query + Synthesis<br/><i>semantic search over structured knowledge</i>"]
    end
```

### What changes in practice

| Aspect | Flatten-and-RAG | KNOX |
|--------|-----------------|------|
| **Table query** | Flat text — LLM guesses cell values. | `knox query "compare latency figures"` returns the structured table with cell annotations. |
| **Figure access** | Blurry captionless gap in the text. | Captioned wiki nodes with VLM-enriched descriptions. Searchable. |
| **Cross-refs** | "As shown in section 3.1" — meaningless after chunking. | Resolver expands every cross-ref into the target page slug. |
| **Equations** | LaTeX becomes unreadable escape sequences. | Equations keep LaTeX source + rendered image. Query by concept or formula. |
| **Answer quality** | Hallucinated, no provenance, no citations. | Every answer cites source pages with provenance back to the original document. |
| **Cost profile** | Cheap to build, expensive to query (bad context). | One-time compile cost, cheap queries against structured wiki. |

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
