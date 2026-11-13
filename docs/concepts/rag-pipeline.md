# RAG Pipeline

kompend's query pipeline includes optional RAG enhancement techniques that improve retrieval quality. Each is a **plug-in step** — configurable via `kompend.toml` under `[search]`. The optional steps default to **disabled**; hybrid retrieval is always active.

## Pipeline Flow

```text
User Query
    │
    ├── 0. Intent routing (optional [decision]) ── knowledge vs small talk
    │       ↓
    ├── 1. Multi-Query Expansion (RAG-Fusion) ── generate N variants
    │       ↓
    ├── 2. HyDE ── generate hypothetical answer for embedding
    │       ↓
    ├── 3. Hybrid Retrieval (FTS + vector + token-overlap + raw)
    │       ↓
    ├── 3b. Adaptive retrieval (--adaptive) ── zoom in/out/targeted
    │       ↓
    ├── 4. Neural Re-ranker ── cross-encoder re-ranking
    │       ↓
    ├── 5. CRAG ── LLM relevance grading, drop noise
    │       ↓
    ├── 6. Synthesis ── one-shot answer, or the opt-in agentic tool loop
```

## Step Details

### 0. Intent routing (optional)

**Config:** `[decision]` — `enabled`, `model`, `timeout`

When the decision sidecar is enabled, `kompend query` first classifies the
message as a knowledge question or small talk. Regex rules answer the obvious
greetings and thanks; only the queries those rules leave as knowledge reach
the decision model, which catches small talk the patterns miss. A non-knowledge
verdict answers with the canned reply and skips retrieval entirely. A disabled
or failing sidecar keeps the deterministic path. `--no-classify` skips the
step and always retrieves. See
[Decision model](../guides/llm-config.md#decision-model-optional-sidecar).

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
- **Vector** — Semantic search via sqlite-vec (pydantic-ai embedders: `<provider>:<model>`, e.g. `ollama:granite-embedding:278m`)
- **Token-overlap** — Simple token intersection for fallback
- **Raw sources** — BM25 search over original source files

### 3b. Adaptive retrieval

**Flag:** `kompend query --adaptive`

Adaptive retrieval walks the chunk tree instead of returning a flat top-K, and
selects the granularity per candidate:

- **ZOOM_IN** — retrieve section summaries first, then expand the hottest
  sections to their full leaf chunks.
- **ZOOM_OUT** — when too many leaves hit, collapse them to section summaries
  so the answer sees structure rather than fragments.
- **TARGETED** — when the query names a table or equation, query the typed
  structured node directly.

The choice is context-budget-aware: the retriever keeps the assembled context
inside the model's input window instead of truncating the tail. It works best
with a compiled chunk tree; without one, hybrid retrieval remains the default.

### 4. Neural Re-ranker

**Config:** `[search.reranker]` — `enabled`, `backend`, `model`, `top_k`

Re-ranks the top-K candidates from hybrid retrieval. The `backend` decides how:

- **`bi-encoder`** (default) — embeds the query and each candidate with the vector store's embedding provider and sorts by cosine similarity. The candidates were ranked by that model already, so this mostly reorders them.
- **`flashrank`** — a cross-encoder that reads the query and the page body together, and judges the pair. Needs the optional extra (`uv add 'kompend[rerank]'`); the model is downloaded on first use and cached under `.kompend/cache`.
- **`none`** — skip the stage.
- **When to enable:** When ranking quality matters more than latency. Bi-encoder re-ranking adds one embedding call per candidate; the cross-encoder adds one inference per candidate.
- **Model:** the cross-encoder's model name (`reranker.model`, empty = its default). The bi-encoder uses the vector store's model (`compile.embedding.model`).
- **top_k:** 20 (default). Higher = more candidates re-ranked but slower.

### 5. CRAG (Corrective RAG)

**Config:** `[search.crag]` — `enabled`, `threshold`

After retrieval, the LLM grades each candidate chunk for relevance to the query. Chunks scoring below the threshold are dropped before synthesis.

- **When to enable:** When retrieval often returns irrelevant content (noisy corpus, broad queries).
- **Threshold:** `0.3` (default). Lower = more permissive, higher = stricter filtering.
- **Cost:** 1 LLM call per candidate chunk. Use with `top_k` to control cost.

### 6. LLM Synthesis

Always active when an LLM is configured. Synthesizes the final answer from the filtered, ranked candidates.

Synthesis runs as one unified query session: the retrieved candidates are
the first-turn context, and the wiki tools (`search_wiki`, `read_page`,
figure/table lookups) stay available to the model. For a single-fact query
the session answers in one request from the assembled context; for a broad
question the model may investigate - split the question, read pages,
search again - before answering. `--search-mode raw` skips synthesis
entirely, and tag/attribute filters are honoured both in the upfront search
and in every tool call.

## Configuration Reference

```toml
[search]

# Multi-query expansion (RAG-Fusion)
expansion.enabled = false
expansion.variants = 3

# CRAG (Corrective RAG)
crag.enabled = false
crag.threshold = 0.3

# Neural re-ranker (bi-encoder; uses the vector store's embedding model)
reranker.enabled = false
reranker.top_k = 20

# HyDE
hyde.enabled = false

# Contextual Retrieval (chunk enrichment during T2)
contextual.enabled = false
contextual.model = ""

# Agentic synthesis (multi-turn tool loop; default: one-shot)
agentic.enabled = false
agentic.model = ""
agentic.max_sub_questions = 5
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
