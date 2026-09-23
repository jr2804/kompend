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
- The reopen drift guard compares `(name, data_type)` pairs, not names alone: a
  field whose type changed while keeping its name (e.g. `tags` STRING →
  ARRAY_STRING) must destroy and rebuild, never reopen.
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
