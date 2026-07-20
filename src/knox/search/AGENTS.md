# DOX — knox.search

<!-- on entry: loaded when agent enters src/knox/search -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Semantic search over compiled wikis. Multi-retriever pipeline with adaptive
selection, query classification, budget management, and tag filtering.

## Ownership

- ``src/knox/search/`` — retriever implementations, search pipeline, budget control

## Modules

| Module | Contents |
|--------|----------|
| ``retriever.py`` | ``BaseRetriever``, ``ZvecRetriever`` — zvec-backed vector search |
| ``adaptive_retriever.py`` | ``AdaptiveRetriever`` — multi-strategy retriever with auto-selection |
| ``classifier.py`` | ``QueryClassifier`` — classify queries by type/domain for retriever selection |
| ``budget.py`` | ``SearchBudget`` — per-query token and time budget tracking |
| ``pipeline.py`` | ``SearchPipeline`` — orchestrator combining classification, retrieval, filtering |

## Rules

- Import: ``from knox.search.retriever import ZvecRetriever``,
  ``from knox.search.pipeline import SearchPipeline``.
- All retrievers implement ``BaseRetriever`` ABC (``retrieve(query, k, filter)``).
- ``SearchPipeline`` applies tag filtering (AND/OR) after retrieval via
  ``_matches_tag_filter``.
- No LLM calls in the search path — classification is programmatic.
