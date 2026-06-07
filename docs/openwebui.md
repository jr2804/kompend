# Connecting KNOX with Open-WebUI

KNOX has no custom GUI. It exposes a compiled knowledge base as machine-readable
files (JSON graph, markdown, Typst) and an MCP server for agent integration.
Open-WebUI can consume both.

## Quick Start

### 1. Compile a wiki with vector index

```bash
knox compile ./sources --output ./wiki --format json --index
```

This produces:
- `wiki/wiki.json` — full compiled knowledge graph
- `wiki/.knox/vectors/` — zvec vector store for semantic search (if zvec is available)

### 2. Point Open-WebUI at the JSON graph

Open-WebUI can ingest the compiled wiki as a knowledge source:

1. Open Open-WebUI → Workspace → Knowledge
2. Upload or link to `./wiki/wiki.json`
3. The JSON structure contains: nodes (concepts/sections with content),
   edges (cross-references), and TOC (table of contents)
4. Open-WebUI ingests it as structured documents

### 3. Use the MCP server

KNOX ships with an MCP server that exposes `parse_document` and `compile_wiki` tools:

```bash
# Start the MCP server (stdio)
knox mcp
```

In Open-WebUI, configure an MCP connection:

```json
{
  "mcpServers": {
    "knox": {
      "command": "knox",
      "args": ["mcp"]
    }
  }
}
```

### 4. Query via the CLI (no GUI needed)

```bash
# Search compiled wiki with zvec vector store
knox query "what are the performance requirements?" --wiki ./wiki

# Or search directly the JSON graph
cat ./wiki/wiki.json | jq '.nodes[] | select(.title | test("performance"))'
```

## Why Open-WebUI + KNOX?

| Approach | When to use |
|----------|-------------|
| Direct CLI (`knox compile`, `knox query`) | Batch processing, cron jobs, scripts |
| MCP server | Agent integration (Hermes, Claude Code, etc.) |
| Open-WebUI + JSON graph | Chat interface with RAG over the compiled wiki |
| zvec vector store | Semantic search without an external vector DB |

## zvec — Embedded Vector Search

[zvec](https://zvec.org) is Alibaba's in-process vector database (Apache 2.0).
It stores vectors directly on disk — no server, no Docker, no API keys.

```python
from knox.rag.zvec_store import ZvecWikiStore

store = ZvecWikiStore("./.knox/vectors")
results = store.search("performance requirements", top_k=5)
for r in results:
    print(f"[{r['score']:.2f}] {r['title']}: {r.get('content_preview')[:100]}")
```

When no embedding model is configured, a deterministic hash-based vector is used
(suitable for testing). For production, set `--embed-model` to a sentence-transformer
model name.