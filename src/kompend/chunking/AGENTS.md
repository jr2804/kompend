# DOX — kompend.chunking

<!-- on entry: loaded when agent enters src/kompend/chunking -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Tier 2 chunking — split parsed documents into sized ``ChunkTree`` based on
each model's auto-detected context window. Atomic elements are never split.

## Ownership

- ``src/kompend/chunking/`` — chunk tree construction, extractors, contextual enrichment

## Modules

| Module | Contents |
|--------|----------|
| ``hierarchical.py`` | ``ChunkTreeBuilder`` — hierarchical chunking from ``ParsedDocument`` |
| ``extractors.py`` | ``ContentExtractors`` — per-element-type extraction helpers |
| ``summarizer.py`` | ``NodeSummarizer`` — bottom-up node summaries via ``SummarizerAgent`` or the agentic summarizer session |
| ``contextual.py`` | ``ContextualEnricher`` — prepend situating context to chunks before embedding |

## Rules

- Import: ``from kompend.chunking.hierarchical import ChunkTreeBuilder``.
- Chunk boundaries respect element types: heading-delimited sections and
  ``ATOMIC_TYPES`` (tables, code blocks, equations) are never split.
- Context-window size comes from ``discover_context_window()`` — not hardcoded.
- Tags from ``ParsedDocument.metadata["tags"]`` propagate through ``ChunkTree``.
- ``NodeSummarizer`` re-runs an agentic session for the nodes it left
  unsummarized when the session exhausts its request budget (``UsageLimitExceeded``),
  stopping only when a round summarizes nothing new.
