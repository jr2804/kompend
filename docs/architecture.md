# KNOX Architecture

## Design Philosophy

KNOX decouples **parsing** from **compilation** and adds layers that existing frameworks skip: structure-aware chunking, multi-turn resolution, and a self-correcting compilation loop (WiCER). The architecture is tiered so each layer is independently replaceable.

## 5-Tier Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAW SOURCES                                   │
│  PDF  DOCX  XLSX  PPTX  EPUB  MD  TXT  images  URLs (files)  URLs (web) │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 1: PRE-PROCESSING (Parser-Agnostic Router)                      │
│  ┌─────────────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Docling (PDF)   │  │ Pandoc     │  │ pandas   │  │ HTTP     │     │
│  │ │ IBM 60.7k★   │  │ │ (DOCX)   │  │ │(XLSX)  │  │ │ Down-  │     │
│  │ │ Extracts:     │  │ │ Fallback │  │ │ tabulate│  │ │ loader │     │
│  │ │ tables, figs, │  │ │ for all  │  │ │ for     │  │ │ for    │     │
│  │ │ equations,    │  │ │ formats  │  │ │ spreadsheets│ │ remote │     │
│  │ │ LaTeX,        │  │ │          │  │ │         │  │ │ files  │     │
│  │ │ reading order │  │ │          │  │ │         │  │ │       │     │
│  │ └─────────────────┘  └────────────┘  └──────────┘  └──────────┘     │
│  ┌─────────────────┐  ┌─────────────────┐                              │
│  │ Oxide Office    │  │ Web Crawler     │                              │
│  │ │ (legacy .doc) │  │ │ crawl4ai       │                              │
│  │ │ pre-2007 fmt  │  │ │ for web docs  │                              │
│  │ └─────────────────┘  └─────────────────┘  ┌──────────────┐           │
│  │                                           │  Convert-    │           │
│  │  Strategy per format:                     │  anything-   │           │
│  │  ├─ PDF → Docling (primary)               │  to-PDF-     │           │
│  │  ├─ DOCX → Docling or pandoc              │  first?      │           │
│  │  ├─ XLSX → pandas → structured JSON       │  (Open)      │           │
│  │  ├─ PPTX → pandoc                         │              │           │
│  │  ├─ EPUB → pandoc                         │              │           │
│  │  ├─ images → GLM-OCR / VLM                │              │           │
│  │  └─ URLs → HTTPDownloader OR WebCrawler   │              │           │
│  └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Output: Markdown (.md) + Structured JSON sidecar (.knox.json)          │
│  ─ Tables as schemas (list[list[str]] + cell types)                     │
│  ─ Figures with captions, bounding boxes, extracted PNG                 │
│  ─ Equations as LaTeX source + base64 image fallback                      │
│  ─ Heading hierarchy preserved with level metadata                        │
│  ─ Reading order index per element                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 2: CHUNKING (Context-Window Aware Hybrid)                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  Step 1: Group by heading hierarchy                          │      │
│  │  ├─ Section → Subsection → (elements... )                    │      │
│  │  │  Tables/figures/equations/code: NEVER SPLIT (atomic)       │      │
│  │  │  Text sections: split if > token_budget                  │      │
│  │  └─ Heading preserved as chunk-level metadata                │      │
│  │                                                             │      │
│  │  Step 2: Select chunker based on content type              │      │
│  │  ├─ Dense text → Chonkie SemanticChunker            │      │
│  │  ├─ Mixed topic → Chonkie RecursiveChunker           │      │
│  │  ├─ Large table → Chonkie TableChunker               │      │
│  │  ├─ Code block → Chonkie CodeChunker                 │      │
│  │  └─ Small text → Fixed-size window                    │      │
│  │                                                             │      │
│  │  Step 3: Budget management                                  │      │
│  │  ├─ Detect model token limit via litellm                    │      │
│  │  ├─ Reserve 20% budget for prompts/system/instructions       │      │
│  │  └─ Distribute remainder across chunks                       │      │
│  └──────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 3: RESOLUTION (Optional, Multi-Turn)                            │
│  ┌──────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │ LLM Resolver │  │ VLM Enricher    │  │ MCP Tool Hooks  │           │
│  │ │ abbreviat- │  │ │ analyze       │  │ │ external      │           │
│  │ │ ion expand │  │ │ extracted figs│  │ │ lookup tools  │           │
│  │ │ cross-ref  │  │ │ via context   │  │ │ (DOI, gloss-  │           │
│  │ │ validation │  │ │ from nearby   │  │ │ ary, etc.)    │           │
│  │ │ term dis-  │  │ │ text          │  │ │               │           │
│  │ │ ambig-     │  │ └─────────────────┘  │               │           │
│  │ │ uation     │  └──────────────────────┘               │           │
│  │ └──────────────┘                                        │           │
│  │                                                         │           │
│  │  Per-document resolution workflow:                       │           │
│  │  ├─ Default: one-shot (no resolution, fastest)           │           │
│  │  └─ Opt-in: multi-turn chat for document                │           │
│  │                                                           │           │
│  │  Each resolver is pluggable via MCP or CLI tool.         │           │
│  └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 4: COMPILATION (Model-Agnostic, WiCER Loop)                     │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                                                             │      │
│  │  +──────────────────────────────+                          │      │
│  │  │  Step A: BLIND COMPILATION   │                          │      │
│  │  │  ├─ litellm for provider abstraction                      │      │
│  │  │  ├─ Model selection: cheap local (Ollama) first         │      │
│  │  │  ├─ Fallback to remote (deepseek, openrouter, etc.)     │      │
│  │  │  ├─ Multimodal: feed figures to vision LLM              │      │
│  │  │  └─ Output: compiled wiki W₀                            │      │
│  │  +──────────────────────────────+                          │      │
│  │              │                                             │      │
│  │  +───────────▼───────────+                                  │      │
│  │  │  Step B: EVALUATION     │                               │      │
│  │  │  ├─ Probe Q&A per source document                      │      │
│  │  │  ├─ Score answers from W₀ (LLM-as-judge)               │      │
│  │  │  ├─ Identify score-1 failures (catastrophic loss)     │      │
│  │  │  └─ Report: per-source quality + fact loss              │      │
│  │  +──────────────────────────+                               │      │
│  │              │                                             │      │
│  │  +───────────▼───────────+                                  │      │
│  │  │  Step C: DIAGNOSIS      │                               │      │
│  │  │  ├─ For each failure: trace lost facts to source       │      │
│  │  │  ├─ Extract critical facts as "preserve" items          │      │
│  │  │  └─ Score: generic pinning (+0.16) vs targeted (+0.95) │      │
│  │  +──────────────────────────+                               │      │
│  │              │                                             │      │
│  │  +───────────▼───────────+                                  │      │
│  │  │  Step D: REFINE         │                               │      │
│  │  │  ├─ Re-compile with pinned facts as "must-include"      │      │
│  │  │  ├─ Repeat until convergence or max iterations           │      │
│  │  │  └─ Final output: W_final (quality ~80% recovered)     │      │
│  │  +──────────────────────────+                               │      │
│  │              │                                             │      │
│  │              └──────────────────────────────────────► (repeat)   │
│  └─────────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 5: OUTPUT (Multi-Target)                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐│
│  │ Markdown     │  │ JSON Graph   │  │ Typst/PDF    │  │ MCP Server  ││
│  │ (Obsidian)   │  │ (SQLite)     │  │ (Academic)   │  │ (for agents)││
│  │ └────────────┘  │ └────────────┘  │ └────────────┘  │ └───────────┘│
│  └─────────────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Models

### `ParsedDocument`
The output of Tier 1 (pre-processing).

```python
@dataclass
class ParsedDocument:
    source_path: Path
    elements: list[DocumentElement]
    metadata: DocumentMetadata
    images_dir: Path | None
    reading_order: list[int]  # indices into elements sorted by reading order
    toc: list[TOCEntry]       # heading hierarchy
```

### `DocumentElement`

```python
class ElementType(StrEnum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    CODE = "code"
    LIST_ITEM = "list_item"
    UNKNOWN = "unknown"

@dataclass
class DocumentElement:
    element_type: ElementType
    content: str               # text / markdown / LaTeX
    metadata: dict[str, Any]  # bounding box, caption, language, etc.
    extracted_image: Path | None  # for IMAGE/FORMULA
```

### `Chunk`

```python
@dataclass
class Chunk:
    chunk_id: str
    elements: list[DocumentElement]
    heading_context: str       # e.g., "Section 3.2 > Subsection 3.2.1"
    token_count: int
    source_doc: Path
    reading_order: tuple[int, int]  # (start, end) indices
```

### `ResolvedChunk`

```python
@dataclass
class ResolvedChunk:
    chunk: Chunk
    entities: list[Entity]     # resolved terms, abbreviations, etc.
    relationships: list[Relationship]  # cross-document links
    confidence: float
```

### `CompiledWiki`

```python
@dataclass
class CompiledWiki:
    nodes: list[WikiNode]
    cross_references: list[CrossReference]
    contradictions: list[Contradiction]
    metadata: WikiMetadata
    quality_score: float  # WiCER evaluation metric
    iteration: int       # WiCER loop count
```

## Core Principles (Expanded)

### 1. Parser-Agnostic
No parser is baked into KNOX's core. The `BaseParser` interface accepts raw documents and produces `ParsedDocument`. The `ParserRouter` selects by file extension and user preference. If a new parser emerges (`docu-sonic-9000`), add a module implementing `BaseParser()` — no other tier changes.

### 2. Structure-Preserving
Flattening a 50-row comparison table to markdown destroys queryability. KNOX stores tables as structured JSON (`list[list[str]]` with cell type annotations). Figures retain captions, bounding boxes, and extracted PNGs. Equations keep LaTeX source and image fallback.

### 3. Model-Aware
litellm provides provider abstraction, but KNOX goes further:
- Auto-detect context window from model metadata
- Reserve budget for system prompts (20%)
- Cheap local model for pre-processing (chunking heuristics)
- Expensive/remote model only for WiCER compilation
- Fallback chain: local → remote → error

### 4. Agent-Native
Three interfaces, same core:
- **CLI:** `knox compile sources/ --output wiki/`
- **Python SDK:** `pipeline.compile(sources=[Path("paper.pdf")])`
- **MCP Server:** `mcp_knox_compile`, `mcp_knox_query` tools for Hermes/Claude Code/Codex

No GUI. No web server required. Headless cron-friendly.

### 5. Resolution-Ready
Default: one-shot ingestion (fast). Opt-in per document: multi-turn resolution (accurate). Resolvers are pluggable via MCP or CLI tools:
- Abbreviation expansion → `mcp_glossary_resolve`
- Cross-reference validation → `mcp_doi_lookup`
- Term disambiguation → `mcp_wikipedia_disambiguate`

### 6. Multimodal
Vision-capable LLMs analyze extracted figures during compilation. The `VLMBoundBoxAnalyzer` sends figure images + surrounding text + figure caption to a vision model. Output: searchable caption + classification ("architecture diagram", "performance graph", "workflow"). Figures become first-class wiki nodes.

### 7. Iterative Compilation (WiCER)
Inspired by [arXiv:2605.07068v1](https://arxiv.org/pdf/2605.07068v1). Blind compilation over-compresses 2–3× beyond targets. WiCER recovers 80% of lost quality in 1–2 iterations via probe Q&A and targeted diagnosis. KNOX's `WikiCompiler` implements the CEGAR-style loop: compile → evaluate → diagnose → refine.

### 8. Hybrid Chunking
Structure-aware first: tables/figures/equations never split. Heading hierarchy groups elements. Only text-heavy sections get Chonkie treatment. This preserves queryability of structured data while leveraging Chonkie's mature algorithms (Recursive, Semantic, Table, Code) for free-form text.

## Integration Points

### Hermes Agent
KNOX provides an MCP server (`src/knox/mcp/server.py`) exposing `parse_document` and `compile_wiki` tools. Hermes skills can call these tools directly. The CLI remains the source of truth — the MCP layer is a thin wrapper.

### Open-WebUI
No custom GUI. KNOX exposes a queryable JSON graph / SQLite database. Open-WebUI connects via API. This follows the user's explicit requirement to minimize UI overhead.

### Hybrid RAG (Optional)
For sidecar traditional RAG alongside the compiled wiki:
- **alibaba/zvec** — embedded vector database for similarity search
- **tobi/qmd** — local CLI search engine over the wiki corpus
Both are optional extras. KNOX works without them.

## File Flow

```
source.pdf
    → Tier 1: DoclingParser.parse() → source.md + source.knox.json
    → Tier 2: HybridChunker.chunk() → list[Chunk]
    → Tier 3: LLMResolver.resolve() → list[ResolvedChunk]
    → Tier 4: WikiCompiler.compile() → compiled_wiki.json
    → Tier 5: MarkdownOutputWriter.write() → wiki/ (markdown files)
```

## References

- Karpathy's compiled LLM-wiki: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- WiCER paper (arXiv:2605.07068v1): https://arxiv.org/pdf/2605.07068v1
- Docling: https://github.com/DS4SD/docling
- Chonkie: https://github.com/chonkie-inc/chonkie
- litellm: https://github.com/BerriAI/litellm
