---
---

# Pipeline Tiers

kompend follows a 5-tier pipeline. Each tier is independently replaceable,
testable, and has a clear `Base` interface.

## Tier 1: Preprocessing

**Input:** Raw source files (PDF, DOCX, XLSX, PPTX, EPUB, MD, TXT, URLs)  
**Output:** `ParsedDocument` with structured JSON sidecar  
**Interface:** `BaseParser` in `src/kompend/preprocessing/parsers/base.py`

``` mermaid
flowchart LR
    classDef source fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef output fill:#fff9c4,stroke:#f57f17,stroke-width:2px

    PDF[PDF / DOCX / PPTX / XLSX / EPUB] --> Parser[ParsecraftParser<br>per-page routing]
    TXT[Text / Markdown / HTML] --> Parser
    IMG[Images] --> Parser
    URL[URL] --> HTTP[HTTPDownloader]
    URL2[Web URL] --> Crawler[WebCrawler]

    Parser --> PD[ParsedDocument]
    HTTP --> PD
    Crawler --> PD

    class PDF,TXT,IMG,URL,URL2 source
    class Parser,HTTP,Crawler process
    class PD output
```

**Key classes:**

| Class | File | Purpose |
|-------|------|---------|
| `BaseParser` | `parsers/base.py` | Abstract interface: `parse(path) → ParsedDocument` |
| `ParsecraftParser` | `parsers/parsecraft_parser.py` | The one backend; per-page routing across parsecraft's sub-backends |
| `HTTPDownloader` | `downloaders/http.py` | Direct file URLs |
| `WebCrawler` | `downloaders/crawler.py` | Web pages via crawl4ai |
| `ParserRouter` | `parsers/_registry.py` | Extension-based dispatch |

## Tier 2: Chunking

**Input:** `ParsedDocument`  
**Output:** `ChunkTree` (hierarchical tree of `ChunkNode` objects)  
**Interface:** `BaseChunker` in `src/kompend/chunking/base.py`

``` mermaid
flowchart LR
    PD[ParsedDocument] --> Walk[Walk elements in reading order]
    Walk --> Heading[HEADING → open/close internal node]
    Walk --> Atomic[ATOMIC → structured leaf node]
    Walk --> Text[TEXT/LIST_ITEM → buffer]
    Text --> Split[Split if > token budget]
    Heading --> Tree[ChunkTree]
    Atomic --> Tree
    Split --> Tree
```

**Key classes:**

| Class | File | Purpose |
|-------|------|---------|
| `BaseChunker` | `chunking/base.py` | Abstract interface: `chunk(doc) → ChunkTree` |
| `HierarchicalChunker` | `chunking/hierarchical.py` | Structure-aware tree builder with adaptive strategy |
| `NodeSummarizer` | `chunking/summarizer.py` | Bottom-up LLM summaries for internal nodes |
| `extract_structured` | `chunking/extractors.py` | Table/equation/figure → typed StructuredData |

## Tier 3: Resolution

**Input:** `list[ChunkTree]`  
**Output:** `list[ResolvedChunk]` + `CrossReferenceResult`  
**Interface:** `BaseResolver` in `src/kompend/resolution/base.py`

``` mermaid
flowchart LR
    classDef input fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef output fill:#fff9c4,stroke:#f57f17,stroke-width:2px

    Trees[list[ChunkTree]] --> Registry[EntityRegistry<br>scan_tree]
    Trees --> VLM[VLMEnricher<br>per-leaf]
    Registry --> Checker[CrossReferenceChecker]
    Checker --> Xref[CrossReferenceResult]
    VLM --> RC[list[ResolvedChunk]]

    class Trees input
    class Registry,VLM,Checker process
    class Xref,RC output
```

**Key classes:**

| Class | File | Purpose |
|-------|------|---------|
| `BaseResolver` | `resolution/base.py` | Abstract interface: `resolve(chunk) → ResolvedChunk` |
| `VLMEnricher` | `resolution/vlm_enricher.py` | Figure captioning via vision LLM |
| `ResolverChain` | `resolution/chain.py` | Sequential resolver application |
| `EntityRegistry` | `core/entity_registry.py` | Programmatic acronym/definition extraction, dedup |
| `CrossReferenceChecker` | `core/xref_checker.py` | Definition consistency, reference validation, table checks |

The old `LLMResolver` (per-leaf typed relation extraction) has been removed.
Entity extraction is now programmatic via `EntityRegistry`; cross-document
relations are inferred during compilation from shared entities.

## Tier 4: Compilation

**Input:** `list[ResolvedChunk]`  
**Output:** `CompiledWiki`  
**Interface:** `BaseCompiler` in `src/kompend/compilation/base.py`

``` mermaid
flowchart LR
    classDef input fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef decision fill:#fce4ec,stroke:#c62828,stroke-width:2px
    classDef output fill:#fff9c4,stroke:#f57f17,stroke-width:2px

    RC[list[ResolvedChunk]] --> Compile[Blind Compilation]
    Compile --> Evaluate[Probe Q&A]
    Evaluate --> Diagnose[Diagnose Failures]
    Diagnose --> Refine[Refine with Pinned Facts]
    Refine --> Compile
    Refine --> Wiki[CompiledWiki]

    class RC input
    class Compile,Evaluate,Diagnose,Refine process
    class Evaluate decision
    class Wiki output
```

**Key classes:**

| Class | File | Purpose |
|-------|------|---------|
| `BaseCompiler` | `compilation/base.py` | Abstract interface: `compile(chunks) → CompiledWiki` |
| `GraphCompiler` | `compilation/graph.py` | Heading-based grouping, page-kind inference, typed edges |
| `TwoPhaseCompiler` | `compilation/two_phase.py` | Two-phase extract-then-generate |
| `WiCERCompiler` | `compilation/wicer.py` | Iterative compile-evaluate-refine loop |

See [WiCER Loop](wicer-loop.md) for a detailed walkthrough.

## Tier 5: Output

**Input:** `CompiledWiki`  
**Output:** Files on disk (OKF, Markdown, JSON, Typst)  
**Interface:** `BaseOutputWriter` in `src/kompend/output/base.py`

``` mermaid
flowchart LR
    Wiki[CompiledWiki] --> OKF[OkfWriter]
    Wiki --> MD[MarkdownWriter]
    Wiki --> JSON[JSONWriter]
    Wiki --> Typst[TypstWriter]
    OKF --> wiki_okf[wiki/ OKF bundles]
    MD --> wiki_md[wiki/*.md]
    JSON --> wiki_json[wiki.json]
    Typst --> wiki_typ[wiki.typ]
```

**Key classes:**

| Class | File | Purpose |
|-------|------|---------|
| `BaseOutputWriter` | `output/base.py` | Abstract interface: `write(wiki, path)` |
| `OkfWriter` | `output/okf_writer.py` | **Default.** [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) v0.1 bundles |
| `MarkdownWriter` | `output/markdown.py` | Obsidian vault (one `.md` per node + `index.md`). Use `--format markdown` |
| `JSONWriter` | `output/json_writer.py` | Machine-readable JSON graph |
| `TypstWriter` | `output/typst.py` | Academic PDF via Typst |

## Cross-Cutting Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `PipelineConfig` | `core/pipeline.py` | Configuration dataclass for all tiers |
| PydanticAI agents | `agents/*.py` | LLM calls route through `KompendAgentBase` subclasses (Extraction, Generation, Query, WiCER, Doctor, Capture) wrapped via `KompendModel`; failover via `FallbackModel`. See [LLM Configuration](../guides/llm-config.md) |
| `TemplateLoader` | `core/template_loader.py` | Jinja2 template loading with override chain |
| `LintRunner` | `lint/runner.py` | 8 structural + 4 LLM-powered lint rules |
| `MCP Server` | `mcp/server.py` | stdio MCP tools for agent integration |
