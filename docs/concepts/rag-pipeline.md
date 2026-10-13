# RAG Pipeline

kompend's query pipeline includes optional RAG enhancement techniques that improve retrieval quality. Each is a **plug-in step** — configurable via `kompend.toml` under `[search]`, and all default to **disabled**.

## Pipeline Flow

```text
User Query
    │
    ├── 1. Multi-Query Expansion (RAG-Fusion) ── generate N variants
    │       ↓
    ├── 2. HyDE ── generate hypothetical answer for embedding
    │       ↓
    ├── 3. Hybrid Retrieval (FTS + vector + token-overlap + raw)
    │       ↓
    ├── 4. Neural Re-ranker ── cross-encoder re-ranking
    │       ↓
    ├── 5. CRAG ── LLM relevance grading, drop noise
    │       ↓
    └── 6. LLM Synthesis ── generate final answer
```

## Step Details

### 1. Multi-Query Expansion (RAG-Fusion)

**Config:** `[search.expansion]` — `enabled`, `variants`

Generates N query variants via LLM (keyword rewrites + semantic rewrites), retrieves independently for each variant, then merges all ranked lists via Reciprocal Rank Fusion (RRF).

- **When to enable:** Ambiguous or under-specified queries where the first retrieval pass might miss relevant content.
- **Cost:** N× retrieval cost + 1 LLM call for expansion.
- **Variants:** 3 (default). More variants = better recall but higher cost.

### 2. HyDE (Hypothetical Document Embeddings)

**Config:** `[search.hyde]` — `enabled`

For sparse or ambiguous queries, generates a hypothetical ideal answer passage via LLM, then uses its embedding for retrieval instead of the raw query.

- **When to enable:** Short queries (1-3 words) or queries where the user doesn't know the right terminology.
- **Cost:** 1 LLM call per query + 1 extra vector search.

### 3. Hybrid Retrieval

Always active. Combines four retrieval methods via RRF:

- **FTS** — Full-text search over compiled wiki pages (SQLite FTS5)
- **Vector** — Semantic search via zvec (pydantic-ai embedders: `<provider>:<model>`, e.g. `ollama:granite-embedding:278m`)
- **Token-overlap** — Simple token intersection for fallback
- **Raw sources** — BM25 search over original source files

### 4. Neural Re-ranker

**Config:** `[search.reranker]` — `enabled`, `model`, `top_k`

Re-ranks the top-K candidates from hybrid retrieval using a cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`). Cross-encoders score query-passage pairs directly, producing more accurate relevance scores than embedding similarity.

- **When to enable:** When ranking quality matters more than latency. The cross-encoder is slower than embedding similarity but significantly more accurate.
- **Model:** Default `cross-encoder/ms-marco-MiniLM-L-6-v2`. Any sentence-transformers CrossEncoder model works.
- **top_k:** 20 (default). Higher = more candidates re-ranked but slower.

### 5. CRAG (Corrective RAG)

**Config:** `[search.crag]` — `enabled`, `threshold`

After retrieval, the LLM grades each candidate chunk for relevance to the query. Chunks scoring below the threshold are dropped before synthesis.

- **When to enable:** When retrieval often returns irrelevant content (noisy corpus, broad queries).
- **Threshold:** `0.3` (default). Lower = more permissive, higher = stricter filtering.
- **Cost:** 1 LLM call per candidate chunk. Use with `top_k` to control cost.

### 6. LLM Synthesis

Always active when an LLM is configured. Synthesizes the final answer from the filtered, ranked candidates.

## Configuration Reference

```toml
[search]

# Multi-query expansion (RAG-Fusion)
expansion.enabled = false
expansion.variants = 3

# CRAG (Corrective RAG)
crag.enabled = false
crag.threshold = 0.3

# Neural re-ranker
reranker.enabled = false
reranker.model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker.top_k = 20

# HyDE
hyde.enabled = false

# Contextual Retrieval (chunk enrichment during T2)
contextual.enabled = false
contextual.model = ""
```

## Environment Variables

| Variable | Config Key | Description |
|----------|------------|-------------|
| `KOMP_SEARCH_EXPANSION_ENABLED` | `search.expansion.enabled` | Enable multi-query expansion |
| `KOMP_SEARCH_EXPANSION_VARIANTS` | `search.expansion.variants` | Number of query variants |
| `KOMP_SEARCH_CRAG_ENABLED` | `search.crag.enabled` | Enable CRAG relevance grading |
| `KOMP_SEARCH_CRAG_THRESHOLD` | `search.crag.threshold` | CRAG relevance threshold |
| `KOMP_SEARCH_RERANKER_ENABLED` | `search.reranker.enabled` | Enable neural re-ranker |
| `KOMP_SEARCH_RERANKER_MODEL` | `search.reranker.model` | Cross-encoder model name |
| `KOMP_SEARCH_RERANKER_TOP_K` | `search.reranker.top_k` | Candidates to re-rank |
| `KOMP_SEARCH_HYDE_ENABLED` | `search.hyde.enabled` | Enable HyDE |
| `KOMP_SEARCH_CONTEXTUAL_ENABLED` | `search.contextual.enabled` | Enable contextual retrieval |
| `KOMP_SEARCH_CONTEXTUAL_MODEL` | `search.contextual.model` | LLM model for context generation |
