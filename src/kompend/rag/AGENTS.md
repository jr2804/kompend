# DOX — kompend.rag

## Purpose

Vector-store sidecar for semantic search over compiled wikis.
`sqlite_vec_store.py` wraps [sqlite-vec](https://github.com/asg017/sqlite-vec)
(embedded vector extension for SQLite, Apache 2.0) and exposes
`SqliteVecWikiStore`; `vector_store.py` holds the backend-independent layer —
`VectorField`, the `EmbeddingProvider` ABC / `VectorStoreLike` protocol, and
the composition rules that decide *what* is indexed; `rerank.py` holds the
optional cross-encoder.

The backend is one implementation of `VectorStoreLike`, not the contract. A
second implementation (an ANN engine, say) slots in without touching the
retrievers.

## Ownership

- `src/kompend/rag/` — the vector store, its embedding providers, the rerankers.

## Modules

| Module | Contents |
|--------|----------|
| `vector_store.py` | `VectorField`, `VectorStoreLike`, `EmbeddingProvider` / `PydanticAIEmbeddingProvider`, `make_provider` / `build_embedder`, and the composition rules (`indexable_nodes`, `content_hash`, `embed_text`, `embed_node`, figure helpers) |
| `sqlite_vec_store.py` | `SqliteVecWikiStore` — the vec0 table, the side tables, the FTS5 index, the filters, the fusion, the drift guard |
| `multimodal.py` | `OllamaMediaEmbedder`, `MediaEmbeddingUnsupported` |

## Local Contracts

- **The index is built from the markdown bundle, and staleness is judged by the
  same rule.** `indexable_nodes()` (in `vector_store.py`) defines which nodes
  the index covers; every builder and `is_stale()` must use it. Two
  regressions came from violating this: the compile path indexed the
  in-memory graph while `kompend query` compared against
  `load_wiki_from_files()` (which also yields summary pages), and
  `embed_attachments=False` filtered the build but not the comparison. Both
  made the index look stale after every compile, so the next query silently
  re-embedded the whole corpus. Pass the same `embed_attachments` to
  `build_index()` and `is_stale()` — it is part of what "up to date" means.
  `build_index()` applies the rule itself so a caller cannot skip it.
- **Staleness also covers content, not just ids.** `is_stale()` compares the
  recorded `content_hash()` per node — the embedded text plus the filterable
  metadata (node type, tags, attributes) — so an edited page, a `lint --fix`,
  or a reworded recompile rebuilds instead of serving vectors for text that is
  gone. A manifest row written before hashes existed (empty mapping) is stale
  once, then records them.
- **The manifest row names the backend as well as the model.**
  `vector_manifest` in `kompend.db` records the indexed ids, the content
  hashes, the embed-text version, the embedding model and the `backend`.
  Vectors of one model are not vectors of another, and the vectors of one
  backend are not readable by another — without either column a stale index
  keeps being served. The `backend` column is additive (`reconcile_columns`),
  never a `SCHEMA_VERSION` bump: a bump would drop the whole vault store.
- `VectorField` is the single source of truth for the field names. The schema,
  the writes in `_index_nodes`, and the keys read from search hits must all
  use its members — never string literals.
- **The vector table is `vec0` with `distance_metric=cosine`.** The score is
  `1 - distance`, which for L2-normalised embeddings equals the cosine
  similarity — the same convention the previous backend used, so descending
  score order is unchanged.
- **The filters are parameterised SQL, never a filter string.** `tags_all`,
  `attributes_all` and `attribute_keys` mean "the node has *every* one of
  these" — one `EXISTS` subquery per value, ANDed. `tags_dnf` is an OR of ANDs
  (DNF): each group is a set of tags the node must all have, the groups are
  OR-joined inside one parenthesised clause. Every value travels as a bound
  parameter; a value never enters the SQL text. The canonical semantics live
  in `search/filters.py` — the SQL must pick exactly what those predicates
  pick, and a test says so.
- **A filtered vector query over-samples the candidate pool** (the filters run
  over the candidates, so `k` is raised before they apply). Otherwise a
  restrictive filter returns fewer than `top_k` rows.
- **The lexical index is FTS5 over `title` + `content_preview`, and it is
  preview-bounded** (`content_preview` is capped at `_PREVIEW_LEN`). Page
  bodies live in the wiki markdown the LLM reads for synthesis, so lexical
  search only has to *find* the right pages — do not grow the stored payload
  to full text.
- **A user query is not FTS5 syntax.** FTS5 reads `-` as NOT and `:` as a
  column filter, so `context-window` fails to parse ("no such column:
  window"). The query is split into word characters and each term is quoted
  before it reaches FTS5. A query that still fails answers with no lexical
  results and one logged warning — never an exception into the query path.
- **Fusion is Python-side, weighted RRF with k=60.** `fts_search()` fuses the
  two lexical fields; `hybrid_search()` fuses `[vector, title-fts,
  preview-fts]` with the weights `[vector_weight, fts_weight, fts_weight]`.
  The store has no server-side reranker, so the fusion runs here. Scores stay
  higher-is-better: the lexical path returns `-bm25()` (SQLite's bm25 is
  lower-is-better), the vector path `1 - distance`.
- **The store's own fingerprint decides a rebuild.** `vec_meta` records the
  backend, the dimension and the schema fingerprint. A mismatch (another
  schema, another dimension, another backend) drops the file and rebuilds it —
  the vector index is derived data, so dropping it is the correct recovery.
- **The extension must be loaded on every connection.** A connection without
  `sqlite_vec.load()` does not see the `vec0` tables at all. Every connection
  comes from `sqlite_vec_client.SQLiteVecClient.create_connection` (which
  also applies the pragmas); never open the file with a bare
  `sqlite3.connect`.
- **The vector search is brute-force exact, not approximate.** `vec0` scans the
  candidates — there is no ANN index. That buys exact recall and no index
  build step, at a latency that grows linearly with the corpus (measured on
  the slowest supported CPU: ~9 ms at 10k nodes, ~111 ms at 100k). If the
  corpus ever needs an approximate index, add it as a second
  `VectorStoreLike` implementation — never inside this one.
- **The embedding provider is untouched by the store.** `EmbeddingProvider`,
  `make_provider` and `build_embedder` are backend-independent and unchanged;
  no store embeds images.
- **A node with a figure is embedded with that figure.** `multimodal.py`
  (`OllamaMediaEmbedder`) calls Ollama's `POST /api/embed` with
  `input: {text, image}` — the image as raw base64, one per request, the first
  figure of the node. A model without an image encoder answers
  `501 model does not support media embeddings`; that is
  `MediaEmbeddingUnsupported`, not a transport error, so it is never retried.
  The node is then indexed from its text alone and the degradation is logged
  (`1 node(s) with figures were indexed without their image: …`) — never
  silent. Only Ollama has this endpoint; every other provider reports the same
  way.
- The staleness fingerprint covers the figures' content digest, because the
  figure is part of what the embedder saw. `EMBED_TEXT_VERSION` covers the
  composition; a change to it invalidates every index once.
- **The embedding model is part of the index identity.** The manifest row
  records the provider's `model_id`; a different model makes `is_stale()`
  true and the index is rebuilt. Without it, a model switch kept serving the
  old vectors.

## Work Guidance

- Import from the package root: `from kompend.rag import SqliteVecWikiStore`.
- The store file lives beside the vault store, not inside it: the vault
  store's schema management recreates its file on a version mismatch, which
  would destroy the index and force a full re-embed.

## Verification

- `uv run pytest tests/unit/test_sqlite_vec_filters.py tests/unit/test_sqlite_vec_hybrid.py tests/unit/test_sqlite_vec_store_rebuild.py tests/unit/test_vector_embed_text.py tests/unit/test_multimodal_embedding.py -q`
- `uv run ruff check src/kompend/rag && uv run ty check src/kompend/rag`
