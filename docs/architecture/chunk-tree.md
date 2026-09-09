# Hierarchical Chunk Tree

## Fractal Architecture

The same analytical pattern applies at two scales:

| Scale | Structure | Unit | Relations |
|-------|-----------|------|-----------|
| **Intra-document** | ChunkTree (tree) | ChunkNode | parent/child by heading hierarchy |
| **Inter-document** | CompiledWiki (graph) | WikiNode | typed edges (entity_reference, uses, depends_on) |

Both reuse the same abstractions: entity extraction, cross-reference resolution,
relationship graphs, and structured data models. The document level has a
hierarchical tree structure; the wiki level has a graph structure. This fractal
pattern means the same reasoning tools work at both scales.

## ChunkTree / ChunkNode Data Structures

```python
@dataclass
class ChunkNode:
    id: str
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0
    heading_chain: tuple[str, ...] = field(default_factory=tuple)
    chunk: Chunk | None = None  # leaf prose content
    structured: StructuredData | None = None  # leaf table/equation/figure
    summary: str = ""  # internal node summary (LLM-generated)
    token_count: int = 0


@dataclass
class ChunkTree:
    root_id: str
    nodes: dict[str, ChunkNode]
    source_file: Path | None = None
    strategy: ChunkStrategy = SINGLE_PASS
```

### Node types

- **Root node** — single root per document, depth 0, no heading path
- **Internal nodes** — represent heading sections, carry `summary` (LLM-generated
  from children), have `children_ids`
- **Leaf prose nodes** — carry `chunk` (text content), no `structured`
- **Leaf structured nodes** — carry `structured` (StructuredTable/Equation/Figure),
  also have `chunk` (text representation for backward compat)

## StructuredData Types

DoclingParser already produces structured data (table grids as `list[list[str]]`,
LaTeX for formulas, image captions). The extractors in Phase 2 are
**representation converters** — they transform already-parsed content from raw
text/grids into typed records, preserving structure that was previously
destroyed by stringification during chunking.

```python
@dataclass
class StructuredTable:
    columns: list[str]
    rows: list[dict[str, str]]
    caption: str = ""


@dataclass
class StructuredEquation:
    latex: str
    variables: list[EquationVariable] = field(default_factory=list)
    caption: str = ""


@dataclass
class StructuredFigure:
    caption: str
    alt_text: str = ""
    image_ref: str = ""
    description: str = ""
```

## Scaling Matrix

The adaptive strategy is selected by the (doc_size, context_window) pair. The
true metric is **chunk count** — how many chunks fit in the CW-sized pieces
of the document — not a raw multiplier on CW:

| Chunks needed | Strategy | Cost |
|-------------:|----------|------|
| 1 (doc fits in CW) | `single_pass` | No summaries. Trivial root + leaves. |
| 2–10 | `two_level` | Root summary from section children. |
| >10 | `multi_resolution` | Full-depth tree, all internal nodes summarized. |

Chunk count = `ceil(doc_tokens / effective_max)`. The 10-chunk ceiling is
derived from summary budget: ~150 tokens per chunk summary, ~1500 tokens
root summary cap → 1500/150 = 10. Configurable.

## Multi-Pass Processing (future design)

The chunk tree is designed to support a multi-pass processing strategy across
the tree hierarchy. Today's implementation covers Passes 1–2; 3–5 are future
expansions.

| Pass | Purpose | Status |
|------|---------|--------|
| 1. Structure | Build TOC + per-section summaries | ✅ implemented (NodeSummarizer) |
| 2. Entity registry | Extract terms, definitions, key concepts | ✅ implemented (EntityRegistry) |
| 3. Deep extraction | Per-section structured analysis | 🔲 future |
| 4. Cross-reference | Resolve refs, detect inconsistencies | ✅ implemented (CrossReferenceChecker) |
| 5. Synthesis | Holistic analysis | 🔲 future |

Verification at data level (entity registry dedup, table consistency checks)
runs before LLM synthesis — the LLM only interprets results that were already
exhaustively checked outside its context.

## Adaptive Retrieval

The ChunkTree enables three retrieval modes:

- **Zoom-in** — summary hit → expand to correct leaf nodes
- **Zoom-out** — many leaf hits → collapse to section summaries
- **Targeted** — table/equation query → direct structured lookup, no prose retrieval

The ContextBudgetManager ensures retrieval never exceeds available context
tokens, scaling depth based on the query and available budget.

## Implementation Phases

| Phase | Issue | What was built |
|-------|-------|----------------|
| 0 | knox-evmg | ChunkNode, ChunkTree, StructuredData data structures |
| 1 | knox-ctwe | HierarchicalChunker replacing HybridChunker |
| 2 | knox-hhf5 | Structured extractors + NodeSummarizer |
| 3 | knox-rves | EntityRegistry + CrossReferenceChecker |
| 4 | knox-96qo | ContextBudgetManager + AdaptiveRetriever |
| 5 | knox-ne47 | Multi-resolution compilation from ChunkTree |
