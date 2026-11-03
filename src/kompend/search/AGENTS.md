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
| ``retriever.py`` | ``HybridRetriever`` (zvec-backed, server-side filters), ``KeywordRetriever`` (store-less), ``SearchResult``, ``synthesize_results``, ``run_agentic_query`` |
| ``filters.py`` | ``matches_tag_filter`` / ``matches_attr_filter`` — canonical tag/attribute predicates |
| ``adaptive_retriever.py`` | ``AdaptiveRetriever`` — multi-strategy retriever with auto-selection |
| ``classifier.py`` | ``QueryClassifier`` — classify queries by type/domain for retriever selection |
| ``budget.py`` | ``SearchBudget`` — per-query token and time budget tracking |
| ``pipeline.py`` | ``EnhancedRetriever`` — HybridRetriever + pluggable RAG steps |

## Rules

- Import: ``from kompend.search.retriever import HybridRetriever``,
  ``from kompend.search.pipeline import EnhancedRetriever``.
- Tag/attribute predicates live in ``filters.py`` (canonical; re-exported by
  ``cli._query_helpers``). Tags are an **OR of ANDs** (DNF); attributes are an
  **AND** of ``(key, value)`` and key-presence (``value is None``) checks.
- ``HybridRetriever.search`` pushes filters down into zvec
  (``CONTAIN_ANY``/``CONTAIN_ALL``); ``KeywordRetriever`` applies the same
  predicates in Python (store-less path). The agentic query tools thread the
  caller's filters through ``QueryAgentDeps`` into ``retriever.search``.
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
