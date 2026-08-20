# Page Provenance

KNOX tracks which source files and line ranges each wiki page was compiled from.
The `provenance` command shows a compiled page's source chunks — so you can
verify where a claim or concept came from.

> **Naming:** this is called **provenance** (not "trace") to avoid confusion
> with the OpenTelemetry tracing feature used for agent observability.
> See [Observability](observability.md) for that one.

## Listing pages

You need a page's slug to inspect it. Run `provenance` with no argument to list
every page in the compiled wiki with its slug and source count:

```bash
$ knox provenance

3 page(s) in compiled wiki:

  self-attention                           (3 sources)  Self-Attention
  transformer                              (2 sources)  Transformer Architecture
  bert                                     (1 source)   BERT
```

Copy any slug from the first column to inspect that page.

## Inspecting a page

```bash
knox provenance <page-slug>... [--with-content] [--json]
```

Pass one slug to inspect a single page, or several to inspect multiple pages
at once (useful for agents and batch inspection):

### Single page

```bash
$ knox provenance self-attention

# self-attention
  Source: attention.pdf (lines 42-58)
  Source: transformer.pdf (lines 123-130)
  Source: bert.pdf (lines 7-15)
```

### Multiple pages

```bash
$ knox provenance self-attention bert

# Self-Attention
  Source: attention.pdf (lines 42-58)
  Source: bert.pdf (lines 7-15)

# BERT
  Source: bert.pdf (lines 1-40)
```

### With content snippets

```bash
$ knox provenance self-attention --with-content

# self-attention
  Source: attention.pdf (lines 42-58)
    "The attention mechanism computes a weighted sum of values..."
  Source: transformer.pdf (lines 123-130)
    "Self-attention relates different positions of a single sequence..."
```

### Machine-readable output

```bash
$ knox provenance self-attention --json
{
  "page": "Self-Attention",
  "slug": "self-attention",
  "sources": [
    {"source": "attention.pdf", "lines": "42-58"},
    {"source": "transformer.pdf", "lines": "123-130"}
  ]
}
```

With multiple slugs, `--json` returns an **array** of objects (one per
slug); with a single slug it returns a single object for easier parsing.

## How provenance works

Provenance data comes from `SourceChunk` references tracked during compilation.
When a chunker splits a document, each chunk carries an ID like:

```text
<source-file>:<start-line>-<end-line>
```

These IDs are stored in `WikiNode.source_chunks` as `list[str]`.
The compilation pipeline preserves these references when assembling pages.

### Pipeline stages

1. **Preprocessing:** Source files are chunked; each chunk gets a provenance ID
2. **Resolution:** LLM may reference specific chunks when generating concepts
3. **Compilation:** Pages collect all contributing chunk IDs
4. **Output:** Chunk IDs are stored in the wiki metadata

### Data model

```python
@dataclass
class WikiNode:
    source_chunks: list[str] | None = None
    # ["attention.pdf:42-58", "transformer.pdf:123-130"]
```

## Integrating provenance in your workflow

### During development

```bash
# Compile with provenance tracking
knox compile docs/ --provenance

# List pages, then inspect any one
knox provenance
knox provenance transformer
```

### In scripts

```python
from knox.core.models import CompiledWiki

# Load the wiki
wiki = CompiledWiki.load(".knox/last_wiki.json")

# Find a node
for node in wiki.nodes:
    if node.title == "Transformer":
        print(f"Sources: {', '.join(node.source_chunks or [])}")
```

## MCP equivalent

The same functionality is exposed via the MCP tool `provenance` (not `trace`)
with the same arguments:

```text
provenance(wiki_dir=None, page_slugs=[...], with_content=False, json_output=False)
```

## Limitations

- Provenance is only as granular as chunking allows. Without line-number
  tracking (Q16, future), chunk IDs may point to whole-file ranges.
- The `--provenance` flag must be passed to `compile` — it is off by default
  to reduce metadata size.
- External parser output (JSON protocol) must include the `source` field for
  provenance to work.
