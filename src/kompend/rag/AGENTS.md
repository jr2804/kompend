# DOX — kompend.rag

## Purpose

Vector-store sidecar for semantic search over compiled wikis.
`sqlite_vec/` wraps [sqlite-vec](https://github.com/asg017/sqlite-vec)
(embedded vector extension for SQLite, Apache 2.0) and exposes
`SqliteVecWikiStore`; `vector_store.py` holds the backend-independent layer —
`VectorField`, the `VectorStoreLike` protocol and the composition rules that
decide *what* is indexed; `embedding_base.py` holds the `EmbeddingProvider`
ABC; `embedding_pydantic.py` holds the pydantic-ai-backed provider;
`rerank.py` holds the optional cross-encoder.

The backend is one implementation of `VectorStoreLike`, not the contract. A
second implementation (an ANN engine, say) slots in without touching the
retrievers.

## Ownership

- `src/kompend/rag/` — the vector store, its embedding providers, the rerankers.

## Modules

| Module | Contents |
|--------|----------|
| `embedding_base.py` | `EmbeddingProvider` ABC, `MediaEmbeddingUnsupported` — the import-light interface; heavy implementations depend on it, never the reverse |
| `vector_store.py` | `VectorField`, `VectorStoreLike`, `to_pydantic_ai_embedding_string`, and the composition rules (`indexable_nodes`, `content_hash`, `embed_text`, `embed_node`, figure helpers) — **import-light** |
| `embedding_pydantic.py` | `PydanticAIEmbeddingProvider`, `make_provider`, `build_embedder` — the pydantic-ai stack lives here and nowhere else |
| `sqlite_vec/` | `SqliteVecWikiStore` (public ops + lifecycle), `schema.py` (DDL, table vocabulary, fingerprint), `index.py` (embed + insert), `query.py` (vector/FTS SQL, hit shape, RRF), `filters.py` (filter SQL), `manifest.py` (manifest row + staleness) |
| `sqlite_vec_store.py` | compatibility module — re-exports the two names the published docs import; new code uses the package root |
| `build.py` | `build_vector_index` — the index builder; it lives here, not in the package `__init__`, because it needs the sqlite-vec backend |
| `rerank.py` | `Reranker` protocol, `FlashRankReranker` (optional extra) |
| `multimodal.py` | `OllamaMediaEmbedder` |

## Local Contracts

- **The backend is a package with one seam per job.** `sqlite_vec/schema.py`
  owns the DDL, the table vocabulary and the fingerprint; `index.py` the
  embed-and-insert path; `query.py` the vec0/FTS5 SQL, the hit shape and the
  RRF; `filters.py` the filter SQL; `manifest.py` the manifest row and the
  staleness rules; `store.py` the public operations and the lifecycle that
  wires them. The SQL is hand-written by contract — `vec0` and FTS5 are
  virtual-table grammars SQLAlchemy Core cannot express (bead knox-97pg), so
  splitting this store into ORM calls is out of scope. `schema.py`, `index.py`,
  `query.py` and `filters.py` take an open connection as an argument and never
  open one — `store.py` owns the store's connection and its lifecycle. Every
  statement is a `Final[str]` template filled by `schema.sql(template, **values)`:
  the fill helper takes a plain `str`, which is the shape bandit accepts, so the
  family carries no `S608` waiver (knox-88xy). `manifest.py` is the exception and
  opens `kompend.db` itself: the manifest row
  is read by `recorded_embedding_model()` before any store is opened (the
  doctor calls it on another generation's store), so it cannot depend on a
  `SqliteVecWikiStore` instance. `sqlite_vec_store.py` is a compatibility
  surface only — it keeps the documented import resolving; new code imports
  from the package root.
- **The contract layer stays import-light.** `vector_store.py` and
  `embedding_base.py` must not import pydantic-ai, the provider SDKs or
  `kompend.llm.models`: a module that only needs the protocol, the field
  names or the composition rules must not pay the provider stack (measured:
  ~20 s of interpreter startup). The pydantic-ai provider lives in
  `embedding_pydantic.py`, and a module that builds a provider imports it
  from there. The store resolves its provider through the package's lazy
  export (`kompend.rag.make_provider`) so importing the store stays light
  too. A string constant like `OLLAMA_DEFAULT_BASE_URL` comes from
  `kompend.llm.constants`, never from `kompend.llm.models`. Model-string
  parsing comes from `kompend.llm.model_id` (stdlib-only, via the package's
  lazy export) — never a local split.
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
  preview-bounded** (`content_preview` is capped at `_PREVIEW_LEN`, a class
  attribute of `SqliteVecWikiStore`, passed into `index.py`). Page
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
- **The import chain stays light.** `kompend/rag/__init__.py` attaches every
  name via `lazy_loader` — importing `kompend.rag` or any of its light
  submodules (`embedding_base`, `multimodal`) must not pull the pydantic-ai
  provider stack into interpreter startup, and it must not pull the sqlite-vec
  backend either (measured: 2.3 s for the backend alone, so an eager import
  there made `import kompend.rag` 100× heavier). Code that only handles
  providers abstractly (dedup, memory, test fixtures) imports
  `EmbeddingProvider` from `embedding_base.py`, not `vector_store.py`. New
  heavy imports go behind `lazy.attach`, never into the package `__init__`
  top level (function-level imports are banned repo-wide — see
  `.agents/CODE.md`). A *function* that needs a lazy name lives in the
  submodule that owns it (`build.py` next to the store) — it may not sit in
  the package `__init__`, where the name would have to be imported to be
  visible. The package exports its names through `lazy.attach`'s `__all__`,
  not a literal one, so no name is declared twice.
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
  old vectors. The doctor reads the recorded string with
  `recorded_embedding_model()` (read-only `connect()`, never `open_store()`:
  the schema ensure would clean-cut a foreign-generation `kompend.db`).
- **The rerank stage is optional and degrades.** `rerank.py` imports flashrank
  lazily, so a vault that never enables the stage never pays for onnxruntime.
  A missing model is a stage failure: the candidates come back unchanged.

## Work Guidance

- Import from the package root: `from kompend.rag import SqliteVecWikiStore`.
- The store file lives beside the vault store, not inside it: the vault
  store's schema management recreates its file on a version mismatch, which
  would destroy the index and force a full re-embed.

## Verification

- `uv run pytest tests/unit/test_sqlite_vec_filters.py tests/unit/test_sqlite_vec_hybrid.py tests/unit/test_sqlite_vec_store_rebuild.py tests/unit/test_vector_embed_text.py tests/unit/test_multimodal_embedding.py tests/unit/test_rag.py -q`
- `uv run ruff check src/kompend/rag && uv run ty check src/kompend/rag`
- Every module in `sqlite_vec/` must import on its own:
  `uv run python -c "import kompend.rag.sqlite_vec.<module>"`.
