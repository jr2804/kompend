# DOX — kompend.search

<!-- on entry: loaded when agent enters src/kompend/search -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Semantic search over compiled wikis. Multi-retriever pipeline with adaptive
selection, query classification, budget management, and tag filtering.

## Ownership

- ``src/kompend/search/`` — retriever implementations, search pipeline, budget control

## Modules

| Module | Contents |
|--------|----------|
| ``retriever.py`` | Search entry point: ``HybridRetriever`` (vector-store-backed, server-side filters), ``KeywordRetriever`` (store-less); re-exports the public names of ``retrieval/`` |
| ``retrieval/`` | Retrieval stages: ``results.py`` (``SearchResult``/``SynthesisResult``/``Citation`` shapes), ``vector.py`` (the one store call, hit decoders), ``assemble.py`` (hit → result, snippets, figures), ``rank.py`` (keyword scoring, graph expansion, store-less filter predicates), ``synthesis.py`` (context formatting, LLM call) |
| ``filters.py`` | ``matches_tag_filter`` / ``matches_attr_filter`` — canonical tag/attribute predicates |
| ``adaptive_retriever.py`` | ``AdaptiveRetriever`` — multi-strategy retriever with auto-selection |
| ``classifier.py`` | ``QueryClassifier`` — classify queries by type/domain for retriever selection |
| ``budget.py`` | ``SearchBudget`` — per-query token and time budget tracking |
| ``pipeline.py`` | ``EnhancedRetriever`` — HybridRetriever + pluggable RAG steps |

## Rules

- Import: ``from kompend.search.retriever import HybridRetriever``,
  ``from kompend.search.pipeline import EnhancedRetriever``.
- Retrieval stages live in ``search/retrieval/`` (``__init__.py`` is the stage
  API); ``retriever.py`` composes them and stays the caller-facing import
  site. Stages run vector → assemble → rank → synthesis; adding a stage means
  adding it to that order, not calling back into ``retriever.py``.
- The store call distinguishes **could not ask** (``vector_search`` returns
  ``None``: no store, unavailable, or the call raised — the result is empty)
  from **answered nothing** (``[]``: only then does the keyword path run).
  Collapsing the two turns a broken index into silent keyword results.
- Tag/attribute predicates live in ``filters.py`` (canonical; the CLI query
  paths import them from there). Tags are an **OR of ANDs** (DNF); attributes are an
  **AND** of ``(key, value)`` and key-presence (``value is None``) checks.
- ``HybridRetriever.search`` pushes filters down into the vector store, which
  evaluates them as SQL (``EXISTS`` subqueries over the tag/attribute side
  tables); ``KeywordRetriever`` applies the same predicates in Python
  (store-less path). The two must agree — ``filters.py`` holds the canonical
  semantics and a test pins the store to it. The agentic query tools thread
  the caller's filters through ``QueryAgentDeps`` into ``retriever.search``.
- ``pipeline.py``'s optional steps (query expansion, HyDE, neural rerank,
  CRAG grading) DO call LLM agents — classification is programmatic, those
  steps are not. Each one is best-effort enrichment: on failure it records
  `record_unit_failure(...)` and keeps the unmodified input (query, candidate
  list, or a 0.5 grade). Anything that is NOT enrichment (the final
  synthesis) must burn.
- **A result carries the figures of the page it points at** (``figures`` on
  ``SearchResult`` and on a citation). They are read from the page body
  (``core.text.page_media``), so the path is the one the reader can open —
  never the cache path the parser wrote. Every stage that rebuilds a result
  forwards ``content`` and ``figures``; a stage that drops ``content`` hands
  the synthesis a snippet where the page body was meant.
