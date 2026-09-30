# DOX — knox.chunking

<!-- on entry: loaded when agent enters src/knox/chunking -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Tier 2 chunking — split parsed documents into sized ``ChunkTree`` based on
each model's auto-detected context window. Atomic elements are never split.

## Ownership

- ``src/knox/chunking/`` — chunk tree construction, extractors, contextual enrichment

## Modules

| Module | Contents |
|--------|----------|
| ``hierarchical.py`` | ``ChunkTreeBuilder`` — hierarchical chunking from ``ParsedDocument`` |
| ``extractors.py`` | ``ContentExtractors`` — per-element-type extraction helpers |
| ``summarizer.py`` | ``ChunkSummarizer`` — LLM-based chunk summaries via ``SummarizerAgent`` |
| ``contextual.py`` | ``ContextualEnricher`` — prepend situating context to chunks before embedding |

## Rules

- Import: ``from knox.chunking.hierarchical import ChunkTreeBuilder``.
- Chunk boundaries respect element types: heading-delimited sections and
  ``ATOMIC_TYPES`` (tables, code blocks, equations) are never split.
- Context-window size comes from ``discover_context_window()`` — not hardcoded.
- Tags from ``ParsedDocument.metadata["tags"]`` propagate through ``ChunkTree``.
