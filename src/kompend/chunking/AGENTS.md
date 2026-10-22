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
| ``summarizer.py`` | ``NodeSummarizer`` — bottom-up node summaries via ``SummarizerAgent`` or the agentic summarizer session; ``SummarizerSessionPolicy`` (batch size, stall retries), ``SummarizerContentPolicy`` (content chars, window chars, target chars, skip-single-child) |
| ``contextual.py`` | ``ContextualEnricher`` — prepend situating context to chunks before embedding |

## Rules

- Import: ``from kompend.chunking.hierarchical import ChunkTreeBuilder``.
- Chunk boundaries respect element types: heading-delimited sections and
  ``ATOMIC_TYPES`` (tables, code blocks, equations) are never split.
- Context-window size comes from ``discover_context_window()`` — not hardcoded.
- Tags from ``ParsedDocument.metadata["tags"]`` propagate through ``ChunkTree``.
- ``NodeSummarizer`` runs agentic summaries in bounded batches
  (``SummarizerSessionPolicy`` from ``budget.summary_session_batch_size`` and
  ``budget.summary_session_stall_retries``). A session that exhausts its request
  budget or records nothing for its own payload defers only its batch: the rest
  of the tree continues, stalled batches are retried per the policy, and
  summaries written for ids outside the payload never count as progress.
- Node content is **windowed, never truncated**: content longer than
  ``budget.summary_max_content_chars`` (or the agentic payload window
  ``budget.summary_window_chars``) is summarised in several calls, one per window.
  The per-node summary is the join of its window summaries; window ids carry
  a ``#wN`` suffix, a single-window node keeps its plain id.
- The summarizer's numbers live in ``compile.budget`` only:
  ``SummarizerContentPolicy.from_config(budget)`` and
  ``SummarizerSessionPolicy`` carry them into the summarizer. Never repeat a
  budget number as a default in this module — a default there silently
  overrides the config.
