---
---

# Pipeline Tiers

kompend follows a 5-tier pipeline. Each tier is independently replaceable,
testable, and has a clear `Base` interface.

## Tier 1: Preprocessing

**Input:** Raw source files (PDF, DOCX, XLSX, PPTX, EPUB, MD, TXT, URLs)  
**Output:** `ParsedDocument` with structured JSON sidecar  
**Interface:** `ParserBackend` protocol in `src/kompend/preprocessing/parsers/base.py`

``` mermaid
flowchart LR
    classDef source fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1B1B1B
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#1B1B1B
    classDef output fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#1B1B1B

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
| `ParserBackend` | `parsers/base.py` | Protocol: `can_parse(path)`, `parse(path) → ParsedDocument` |
| `DocumentParser` | `parsers/base.py` | The guards around every parse: mislabel refusal, surrogate scrubbing, backend identity |
| `ParsecraftParser` | `parsers/parsecraft/` (adapter.py) | The one backend; per-page routing across parsecraft's sub-backends |
| `HTTPDownloader` | `downloaders/http.py` | Direct file URLs |
| `WebCrawler` | `downloaders/crawler.py` | Web pages via crawl4ai |
| `build_parser` | `parsers/document_parser.py` | The single construction point; takes `ParserOptions` |

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
    classDef input fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1B1B1B
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#1B1B1B
    classDef output fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#1B1B1B

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

### Layered figure gates

A figure is captioned only after every cheaper gate passes, in order — the
first gate that decides pays nothing further:

1. **Floors** (`VLMEnricher`, always on): byte size and minimum pixel
   dimension (`compile.vlm_min_image_bytes`, `compile.vlm_min_image_dimension`)
   drop tiny logos and spacers. Status: `vlm_filtered`.
2. **Structural/pixel pre-filter** (`resolution/image_heuristics.py`, opt-in
   via `compile.vlm_heuristic_filter_enabled`, default **off**): one bounded
   measurement per figure — header-only when possible, otherwise a downscaled
   sample capped by `vlm_heuristic_max_analysis_edge` /
   `vlm_heuristic_max_analysis_pixels` — classifies line-like shapes (extreme
   aspect ratio with a thin short edge), near-uniform fills and near-blank
   frames. Verdicts are conservative: `UNSURE` and sub-threshold drops fall
   through; a transparency-dominant canvas is a hint, never a drop. The
   measured facts also ride the figure context as a factual `visual_hints`
   clause. Status: `vlm_filtered` (reason names the signature).
3. **Text relevance gate** (opt-in decision model, `compile.decision`): a
   text-only classifier judges the figure's surroundings and the measured
   facts; a failed classification falls back to captioning, never a skip.
   Status: `vlm_relevance_skipped`.
4. **VLM self-skip**: the caption prompt's `SKIP` contract — the model's own
   answer. Status: `vlm_skipped`.

Every verdict lands in the stage ledger as a `resolve` row with a reason code
(`vlm_filtered`, `vlm_relevance_skipped`, `vlm_skipped`, or `ok` for a
caption), so `kompend explain` and the doctor's failure histogram show why a
figure has no caption without log archaeology.

Tuning guidance: enable the pre-filter
(`KOMP_VLM_HEURISTIC_FILTER_ENABLED=true`) on corpora with many decorative
figures, then spot-check `kompend explain` for `vlm_filtered` rows naming real
diagrams. If genuine line art is dropped, loosen
`vlm_heuristic_near_blank_stddev` first, then `vlm_heuristic_uniform_stddev`;
raising `vlm_heuristic_min_confidence` (default `0.9`) makes drops rarer,
never more aggressive. The defaults are deliberately conservative — ambiguous
figures are captioned, not dropped.

## Tier 4: Compilation

**Input:** `list[ResolvedChunk]`  
**Output:** `CompiledWiki`  
**Interface:** `BaseCompiler` in `src/kompend/compilation/base.py`

``` mermaid
flowchart LR
    classDef input fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1B1B1B
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#1B1B1B
    classDef decision fill:#fce4ec,stroke:#c62828,stroke-width:2px,color:#1B1B1B
    classDef output fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#1B1B1B

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
| `LintRunner` | `lint/runner.py` | 14 structural + 4 LLM-powered lint rules |
| `MCP Server` | `mcp/server.py` | stdio MCP tools for agent integration |
