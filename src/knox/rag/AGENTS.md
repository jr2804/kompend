# DOX — knox.rag

<!-- on entry: loaded when agent enters src/knox/rag -->
<!-- parent: ../AGENTS.md | ../../AGENTS.md -->

## Purpose

Vector-store sidecar for semantic search over compiled wikis. `zvec_store.py`
wraps zvec (embedded vector DB, Apache 2.0) and exposes `ZvecWikiStore`,
`ZvecField`, the `EmbeddingProvider` ABC / `VectorStoreLike` protocol, and the
`make_provider` / `build_embedder` embedding factories.

## Ownership

- `src/knox/rag/` — the vector store and its embedding providers.

## Local Contracts

- `ZvecField` is the single source of truth for zvec field and vector names.
  The schema (`_expected_field_schemas`), the writes in `_index_nodes`, and the
  keys read from search hits must all use its members — never string literals.
- The embedding field is indexed with `HnswIndexParam(metric_type=MetricType.COSINE)`.
  HNSW is built lazily: `build_index()` calls `optimize()` **exactly once** after
  the bulk insert. Never call `optimize()` per document (~40 ms each).
- `TAGS` and `ATTRIBUTES` are `ARRAY_STRING` fields with `InvertIndexParam`.
  Filter them with `CONTAIN_ANY` / `CONTAIN_ALL` only — `LIKE` on an
  `ARRAY_STRING` field segfaults zvec 0.7.0.
- The reopen drift guard compares a `(name, "data_type/index_kind")` fingerprint,
  not names or types alone: a field whose type changed while keeping its name
  (e.g. `tags` STRING → ARRAY_STRING) **and** a field that gained an index
  (e.g. `title` gaining `FtsIndexParam`, which leaves name/type unchanged) must
  destroy and rebuild, never reopen.
- `TITLE` and `CONTENT_PREVIEW` carry `FtsIndexParam`; lexical recall is
  **preview-bounded** (`content_preview` is capped at `_PREVIEW_LEN`). Page
  bodies live in the wiki markdown the LLM reads for synthesis, so lexical search
  only has to *find* the right pages — do not grow the stored payload to full text.
- `ATTR_KEYS` is a bare-key `ARRAY_STRING` (alongside `ATTRIBUTES`' `key=value`
  tokens) so attribute **key-presence** is expressible server-side. `attribute_keys`
  filters `ATTR_KEYS CONTAIN_ALL`.
- Server-side filters are `CONTAIN_ANY`/`CONTAIN_ALL` only (never `LIKE` on an
  array field). `tags_dnf` is OR-of-ANDs: each group is `tags CONTAIN_ALL (...)`
  and groups are `or`-joined inside one parenthesised clause; clause *types*
  combine with `and`.
- `fts_search()` fuses the two FTS fields with `RrfReRanker`; `hybrid_search()`
  fuses `[vector, title-fts, content_preview-fts]` in **one** `collection.query`
  with `WeightedReRanker([vector_weight, fts_weight, fts_weight])`. FTS and
  reranker scores are already higher-is-better — unlike the vector path, do NOT
  apply `_score_from_distance` to them. Both degrade to `[]`/vector-only when the
  collection has no FTS index (logged once, never raised into the query path).
- `_prune_orphans` derives its candidate set from the collection's actual
  contents (`iter_docs()`), not the manifest diff, so docs written before the
  manifest existed are pruned too.
- `search()` returns `score = 1 - cosine_distance` (higher is better) so
  descending-score callers keep their ordering, and keeps the same result dict
  shape (`id`, `score`, plus the stored fields).
- Embeddings route through `make_provider` / `build_embedder` (pydantic-ai);
  zvec's built-in embedders do not cover Ollama.

## Work Guidance

## Verification

- `uv run pytest tests/unit/test_rag.py tests/unit/test_zvec_store_rebuild.py tests/unit/test_zvec_tags_schema.py -q`
- `uv run ruff check src/knox/rag && uv run ty check src/knox/rag`

## Child DOX Index

None.
