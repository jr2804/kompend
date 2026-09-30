# DOX — knox.search

<!-- on entry: loaded when agent enters src/knox/search -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Semantic search over compiled wikis. Multi-retriever pipeline with adaptive
selection, query classification, budget management, and tag filtering.

## Ownership

- ``src/knox/search/`` — retriever implementations, search pipeline, budget control

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

- Import: ``from knox.search.retriever import HybridRetriever``,
  ``from knox.search.pipeline import EnhancedRetriever``.
- Tag/attribute predicates live in ``filters.py`` (canonical; re-exported by
  ``cli._query_helpers``). Tags are an **OR of ANDs** (DNF); attributes are an
  **AND** of ``(key, value)`` and key-presence (``value is None``) checks.
- ``HybridRetriever.search`` pushes filters down into zvec
  (``CONTAIN_ANY``/``CONTAIN_ALL``); ``KeywordRetriever`` applies the same
  predicates in Python (store-less path). The agentic query tools thread the
  caller's filters through ``QueryAgentDeps`` into ``retriever.search``.
- No LLM calls in the search path — classification is programmatic.
