---
icon: lucide/git-branch
---

# Pipeline Tiers

KNOX follows a 5-tier pipeline. Each tier is independently replaceable,
testable, and has a clear `Base` interface.

## Tier 1: Preprocessing

**Input:** Raw source files (PDF, DOCX, XLSX, PPTX, EPUB, MD, TXT, URLs)  
**Output:** `ParsedDocument` with structured JSON sidecar  
**Interface:** `BaseParser` in `src/knox/preprocessing/parsers/_base.py`

``` mermaid
flowchart LR
    PDF[PDF] --> Docling[DoclingParser]
    DOCX[DOCX] --> Docling
    XLSX[XLSX] --> Pandas[PandasParser]
    PPTX[PPTX] --> Pandoc[PandocParser]
    EPUB[EPUB] --> Pandoc
    MD[Markdown] --> Pandoc
    URL[URL] --> HTTP[HTTPDownloader]
    URL2[Web URL] --> Crawler[WebCrawler]

    Docling --> PD[ParsedDocument]
    Pandas --> PD
    Pandoc --> PD
    HTTP --> PD
    Crawler --> PD
```

**Key classes:**

| Class | File | Purpose |
|-------|------|---------|
| `BaseParser` | `parsers/_base.py` | Abstract interface: `parse(path) → ParsedDocument` |
| `DoclingParser` | `parsers/docling_parser.py` | Primary PDF parser (IBM Docling, MIT) |
| `PdfPlumberParser` | `parsers/pdfplumber_parser.py` | Fallback PDF parser |
| `PandocParser` | `parsers/pandoc_parser.py` | DOCX/PPTX/EPUB/MD/HTML |
| `PandasParser` | `parsers/pandas_parser.py` | XLSX spreadsheets |
| `HTTPDownloader` | `downloaders/http.py` | Direct file URLs |
| `WebCrawler` | `downloaders/crawler.py` | Web pages via crawl4ai |
| `ParserRouter` | `parsers/_registry.py` | Extension-based dispatch |

## Tier 2: Chunking

**Input:** `ParsedDocument`  
**Output:** `list[Chunk]`  
**Interface:** `BaseChunker` in `src/knox/chunking/_base.py`

``` mermaid
flowchart LR
    PD[ParsedDocument] --> Group[Group by heading]
    Group --> Atomic[Atomic elements<br>tables, figures, equations]
    Group --> Text[Text sections]
    Text --> Chonkie[Chonkie<br>RecursiveChunker]
    Atomic --> Chunks[list[Chunk]]
    Chonkie --> Chunks
```

**Key classes:**

| Class | File | Purpose |
|-------|------|---------|
| `BaseChunker` | `chunking/_base.py` | Abstract interface: `chunk(doc) → list[Chunk]` |
| `HybridChunker` | `chunking/hybrid.py` | Structure-aware + Chonkie fallback |
| `DevChunker` | `chunking/dev_chunker.py` | Development/test chunker |
| `ChonkieChunker` | `chunking/chonkie_chunker.py` | Standalone Chonkie adapter |

## Tier 3: Resolution

**Input:** `list[Chunk]`  
**Output:** `list[ResolvedChunk]`  
**Interface:** `BaseResolver` in `src/knox/resolution/_base.py`

``` mermaid
flowchart LR
    Chunks[list[Chunk]] --> LLM[LLMResolver]
    Chunks --> VLM[VLMEnricher]
    LLM --> Chain[ResolverChain]
    VLM --> Chain
    Chain --> RC[list[ResolvedChunk]]
```

**Key classes:**

| Class | File | Purpose |
|-------|------|---------|
| `BaseResolver` | `resolution/_base.py` | Abstract interface: `resolve(chunks) → list[ResolvedChunk]` |
| `LLMResolver` | `resolution/llm.py` | Abbreviation expansion, term disambiguation |
| `VLMEnricher` | `resolution/vlm_enricher.py` | Figure captioning via vision LLM |
| `ResolverChain` | `resolution/chain.py` | Sequential resolver application |

## Tier 4: Compilation

**Input:** `list[ResolvedChunk]`  
**Output:** `CompiledWiki`  
**Interface:** `BaseCompiler` in `src/knox/compilation/_base.py`

``` mermaid
flowchart LR
    RC[list[ResolvedChunk]] --> Compile[Blind Compilation]
    Compile --> Evaluate[Probe Q&A]
    Evaluate --> Diagnose[Diagnose Failures]
    Diagnose --> Refine[Refine with Pinned Facts]
    Refine --> Compile
    Refine --> Wiki[CompiledWiki]
```

**Key classes:**

| Class | File | Purpose |
|-------|------|---------|
| `BaseCompiler` | `compilation/_base.py` | Abstract interface: `compile(chunks) → CompiledWiki` |
| `GraphCompiler` | `compilation/graph.py` | Two-phase extract-then-generate |
| `WiCERCompiler` | `compilation/wicer.py` | Iterative compile-evaluate-refine loop |

See [WiCER Loop](wicer-loop.md) for a detailed walkthrough.

## Tier 5: Output

**Input:** `CompiledWiki`  
**Output:** Files on disk (markdown, JSON, Typst)  
**Interface:** `BaseOutputWriter` in `src/knox/output/_base.py`

``` mermaid
flowchart LR
    Wiki[CompiledWiki] --> MD[MarkdownWriter]
    Wiki --> JSON[JSONWriter]
    Wiki --> Typst[TypstWriter]
    MD --> wiki_md[wiki/*.md]
    JSON --> wiki_json[wiki.json]
    Typst --> wiki_typ[wiki.typ]
```

**Key classes:**

| Class | File | Purpose |
|-------|------|---------|
| `BaseOutputWriter` | `output/_base.py` | Abstract interface: `write(wiki, path)` |
| `MarkdownWriter` | `output/markdown.py` | Obsidian vault (one `.md` per node + `index.md`) |
| `JSONWriter` | `output/json_writer.py` | Machine-readable JSON graph |
| `TypstWriter` | `output/typst.py` | Academic PDF via Typst |

## Cross-Cutting Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `PipelineConfig` | `core/pipeline.py` | Configuration dataclass for all tiers |
| `LLMClient` | `llm/client.py` | Shared litellm wrapper with failover chain |
| `TemplateLoader` | `core/template_loader.py` | Jinja2 template loading with override chain |
| `LintRunner` | `lint/runner.py` | 8 structural + 3 LLM-powered lint rules |
| `MCP Server` | `mcp/server.py` | stdio MCP tools for agent integration |
