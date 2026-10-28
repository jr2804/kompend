"""Agentic extraction flow - structured memory carried between sections.

``extract_agentic()`` implements the ``AGENTIC`` extraction strategy.
Unlike ``INCREMENTAL`` (raw ``message_history`` that grows unbounded),
this flow maintains a structured ``ExtractionMemory`` digest and injects
a compacted summary as a ``SystemPromptPart`` between sections.

Key differences vs other strategies:

- **Memory-not-history**: the digest is structured (concepts/entities/tags)
  and compaction-managed, not raw message history.
- **Sequential per source**: the agent walks sections in order; each
  section adds to memory. Memory itself is never pruned - but the digest
  handed to the model is consolidated by a model call
  (:class:`kompend.agents.memory_consolidation.MemoryConsolidationAgent`)
  once it exceeds the token budget the context window leaves beside the
  section text, so the flow works with any configured window. The digest
  is never cropped (see ``.agents/CODE.md`` "LLM calls").
- **Per-source final output**: the final ``Extraction`` is derived from
  memory, not merged from per-batch outputs.
- **Source-level caching**: since memory state depends on prior sections,
  per-batch caching is disabled; the final result is cached keyed on
  ``(source, full_text_hash, strategy)``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

from kompend.agents.deps import MemoryConsolidationDeps, PipelineAgentDeps
from kompend.agents.extraction import ExtractionAgent
from kompend.agents.memory_consolidation import MemoryConsolidationAgent
from kompend.compilation._two_phase_core import (
    Extraction,
    ProgressFn,
    TwoPhaseRuntime,
    context_safety_tokens,
)
from kompend.compilation.asset_tools import build_asset_tools
from kompend.compilation.extraction_memory import ExtractionMemory
from kompend.compilation.schema import ExtractionOutput
from kompend.core.models.enums import CacheDeps
from kompend.core.utils.tokens import estimate_tokens
from kompend.llm.context_window import ContextWindow

#: System part that announces the asset-lookup tools. Module-level so the
#: budget math and the injected message can never drift apart.
_TOOLS_PREAMBLE: Final[str] = (
    "You have access to on-demand asset lookup tools: "
    "`get_table(asset_id)` returns pipe-delimited table text; "
    "`get_figure_description(asset_id)` returns figure caption/alt-text. "
    "Use them when you need full table data or figure details "
    "beyond the compact `\u00b6` reference in the section text."
)

#: Digest budgets below this leave no useful prior-section context; the
#: section runs without one and says so, rather than shipping a digest
#: so small it cannot carry any structure.
_MIN_DIGEST_BUDGET_TOKENS: Final[int] = 200

#: Consolidation is retried once with a tighter target before burning.
_CONSOLIDATION_ATTEMPTS: Final[int] = 2

#: Error-text markers of a context-window overflow across providers
#: (Ollama, OpenAI-compatible gateways). Matched case-insensitively.
_OVERFLOW_MARKERS: Final[tuple[str, ...]] = (
    "exceeds the available context size",
    "context_length_exceeded",
    "maximum context length",
    "reduce the length of the messages",
    "reduce the length of your input",
)

logger = logging.getLogger(__name__)


@dataclass
class _RollingConsolidation:
    """Snapshot of what has already been consolidated into the running digest.

    The full :class:`ExtractionMemory` stays untouched as the source of
    truth; this tracks only how much of it the model has already merged,
    so a consolidation call re-sends just the delta.
    """

    consolidated: str = ""
    section_count: int = 0
    concept_keys: frozenset[str] = frozenset()
    entity_keys: frozenset[str] = frozenset()
    tags: frozenset[str] = frozenset()

    def snapshot(self, memory: ExtractionMemory) -> None:
        """Record *memory*'s current extent as consolidated."""
        self.section_count = len(memory.section_summaries)
        self.concept_keys = frozenset(memory.concepts)
        self.entity_keys = frozenset(memory.entities)
        self.tags = frozenset(memory.tags)


def extract_agentic(
    compiler: TwoPhaseRuntime,
    source: str,
    texts: list[str],
    extraction_system: str,
    *,
    progress: ProgressFn | None = None,
) -> Extraction | None:
    """Run agentic extraction with structured memory across sections.

    Parameters
    ----------
    compiler:
        The TwoPhaseCompiler instance (provides config, cache, budget).
    source:
        Source identifier (file path stem, URL, etc.).
    texts:
        List of section text strings, one per agent call, in document
        order. Each section is submitted independently (not batched).
    extraction_system:
        The system prompt body (for caching / progress logging).
    progress:
        Optional callback ``(sections done, sections total, activity)``.
        It is called after every finished section and while the running
        digest is consolidated, so a long document shows movement
        instead of a frozen bar.

    Returns:
    -------
    Extraction or None if all sections failed.

    Notes:
    -----
    - Caching is at the source level (keyed on text hash), NOT per
      section, because memory depends on prior section results.
    - The digest injected per section is sized against the context window
      (window minus answer reserve, safety margin, system prompt and the
      section text). While it fits, the full digest is injected; once it
      crosses the budget it is consolidated by a model call and the
      consolidated context rolls forward - never cropped.
    - A provider context-overflow 400 fails the section without retry:
      resending an unchanged oversized payload cannot succeed.
    """
    full_text_hash = _full_text_hash(texts)

    # Source-level cache
    cache_deps: dict[str, str] = {
        CacheDeps.SOURCE_TEXT: full_text_hash,
        CacheDeps.STAGE: "agentic",
    }
    if not compiler._config.no_cache:
        cached = compiler._cache_store.get_dict("extract", cache_deps)
        if cached is not None:
            logger.info("TwoPhase: agentic extraction cache hit for '%s'", source)
            return Extraction.from_dict(cached)
        logger.info(
            "TwoPhase: agentic extraction cache miss for '%s' (text_hash=%.16s)",
            source,
            full_text_hash,
        )

    # Build agent — with asset lookup tools when the store is populated
    tools = build_asset_tools(compiler._asset_store) if compiler._asset_store and compiler._asset_store.total_count() > 0 else None
    agent = ExtractionAgent(
        model_string=compiler._model or "",
        timeout=compiler._timeout,
        skills=compiler._config.skills,
        delta_retry_enabled=compiler._config.llm.delta_retry_enabled,
        reasoning_effort=compiler._config.llm.reasoning_effort,
        context_manager=compiler._config.llm.context_manager,
        context_window=compiler._context_window.max_input,
        tools=tools or [],
        usage_tracker=compiler._usage_tracker,
        llm=compiler._config.llm,
    )

    memory = ExtractionMemory()
    consolidation = _RollingConsolidation()
    consolidator = MemoryConsolidationAgent(
        model_string=compiler._model or "",
        timeout=compiler._timeout,
        reasoning_effort=compiler._config.llm.reasoning_effort,
        context_window=compiler._context_window.max_input,
        llm=compiler._config.llm,
    )
    static_tokens = compiler._extract_system_tokens + (estimate_tokens(_TOOLS_PREAMBLE) if tools else 0)
    batch_retries = compiler._config.llm.batch_retry_attempts
    overall_succeeded = False
    sections_done = 0

    def _activity(label: str) -> None:
        """Name what the running call is doing, for the progress bar."""
        if progress is not None:
            progress(sections_done, len(texts), label)

    for section_idx, section_text in enumerate(texts):
        logger.debug(
            "TwoPhase: agentic section %d/%d for '%s'",
            section_idx + 1,
            len(texts),
            source,
        )

        # Build message history with memory digest injected as a system message.
        # The digest is sized against the window for THIS section; once the
        # full digest outgrows the budget it is model-consolidated, never cropped.
        budget_tokens = _digest_budget_tokens(compiler._context_window, static_tokens, section_text)
        memory_digest = _running_digest(memory, consolidation, budget_tokens, consolidator, source, activity=_activity)
        message_history: list[ModelMessage] = []
        if tools:
            message_history.append(ModelRequest(parts=[SystemPromptPart(content=_TOOLS_PREAMBLE)]))
        if memory_digest.strip():
            msg = ModelRequest(parts=[SystemPromptPart(content=f"=== Context from prior sections ===\n{memory_digest}")])
            message_history.append(msg)

        deps = PipelineAgentDeps(text=section_text, source_id=f"{source}:{section_idx}")

        validated: ExtractionOutput | None = None
        for attempt in range(1 + batch_retries):
            try:
                validated, _new_messages = agent.run_with_messages(
                    deps,
                    message_history=message_history or None,
                )
            except Exception as exc:
                if _is_context_overflow(exc):
                    # The unchanged payload can never fit: retrying it is
                    # guaranteed waste. Fail the section now; the digest
                    # budget above is what keeps this from recurring.
                    logger.error(
                        "TwoPhase: agentic section %d for '%s' exceeds the context window "
                        "(%s) - skipping without retry; consider a larger "
                        "compile.llm.context_window or smaller chunk sizes",
                        section_idx + 1,
                        source,
                        exc,
                    )
                    validated = None
                    break
                logger.debug(
                    "TwoPhase: agentic section %d attempt %d failed: %s",
                    section_idx + 1,
                    attempt + 1,
                    exc,
                )
                validated = None

            if validated is not None:
                break
            if attempt < batch_retries:
                logger.info(
                    "TwoPhase: agentic section retry %d/%d for '%s' batch %d",
                    attempt + 1,
                    batch_retries,
                    source,
                    section_idx,
                )

        compiler._batch_tracker.record(failed=validated is None)
        sections_done = section_idx + 1
        if progress is not None:
            progress(sections_done, len(texts), "")

        if validated is None:
            logger.warning(
                "TwoPhase: agentic section %d failed for '%s' — skipping",
                section_idx + 1,
                source,
            )
            continue

        # Convert validated output to Extraction and ingest into memory
        section_extraction = Extraction(
            source=source,
            summary=validated.summary,
            entities=list(validated.entities),
            concepts=list(validated.concepts),
            tags=validated.tags,
        )
        memory.ingest(section_extraction)
        overall_succeeded = True

    if not overall_succeeded:
        logger.warning("TwoPhase: agentic extraction failed for all sections of '%s'", source)
        return None

    final = memory.to_extraction(source)

    if not compiler._config.no_cache:
        compiler._cache_store.set("extract", cache_deps, final.to_dict())
        logger.info("TwoPhase: cached agentic result for '%s'", source)

    return final


def _full_text_hash(texts: list[str]) -> str:
    """SHA-256 of concatenated section texts for cache key."""
    return hashlib.sha256("".join(texts).encode()).hexdigest()


def _digest_budget_tokens(context_window: ContextWindow, static_tokens: int, section_text: str) -> int:
    """Tokens the prior-section digest may occupy in one section request.

    Mirrors the batch path's :func:`~kompend.compilation.flows.extraction._extract_char_budget`
    terms - window minus answer reserve, shared safety margin, measured
    system prompt and the section text itself - so the agentic path sizes
    against the same physics instead of discovering the window at the
    provider's 400.
    """
    return (
        context_window.max_input
        - context_window.answer_reserve
        - context_safety_tokens(context_window.max_input)
        - static_tokens
        - estimate_tokens(section_text)
    )


def _is_context_overflow(exc: BaseException) -> bool:
    """Whether *exc* is a context-window overflow reported by the provider.

    Walks the exception chain (gateway errors arrive wrapped) and matches
    the markers providers use for "the request does not fit the window".
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        blob = f"{type(current).__name__}: {current}".lower()
        if any(marker in blob for marker in _OVERFLOW_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _running_digest(
    memory: ExtractionMemory,
    state: _RollingConsolidation,
    budget_tokens: int,
    consolidator: MemoryConsolidationAgent,
    source: str,
    *,
    activity: Callable[[str], None] | None = None,
) -> str:
    """The prior-section context for the next section request.

    While the full digest fits the budget it is injected unchanged (short
    documents never pay a consolidation call). Once it crosses the budget
    the running context is model-consolidated from here on.

    *activity* is called with a label just before the consolidation call,
    so the progress bar can name the model call it is waiting for.
    """
    if budget_tokens < _MIN_DIGEST_BUDGET_TOKENS:
        logger.warning(
            "TwoPhase: no room for prior-section context in '%s' (section leaves %d of the window) - extracting without it",
            source,
            budget_tokens,
        )
        return ""
    full = memory.summarize()
    if estimate_tokens(full) <= budget_tokens:
        return full
    if not full:
        return ""
    logger.debug(
        "TwoPhase: digest for '%s' (~%d tokens) exceeds the %d-token budget - consolidating",
        source,
        estimate_tokens(full),
        budget_tokens,
    )
    if activity is not None:
        activity("consolidating context")
    return _consolidate_digest(consolidator, memory, state, budget_tokens, source)


def _consolidate_digest(
    consolidator: MemoryConsolidationAgent,
    memory: ExtractionMemory,
    state: _RollingConsolidation,
    budget_tokens: int,
    source: str,
) -> str:
    """Consolidate the running digest down to *budget_tokens* via the model.

    Sends only the delta since the last consolidation; the merged result
    replaces the running context. A result that still exceeds the budget
    is retried once with a tighter target, then burns: a digest that
    cannot fit the window beside its section is a configuration problem
    (window too small for the chunk size), not something to crop silently.
    """
    delta = memory.delta_since(
        state.section_count,
        state.concept_keys,
        state.entity_keys,
        state.tags,
    )
    if not delta.strip() and state.consolidated:
        # Nothing new since the last consolidation and the full digest is
        # still over budget: the rolling context is already the answer.
        return state.consolidated
    for attempt in range(1, _CONSOLIDATION_ATTEMPTS + 1):
        target = max(_MIN_DIGEST_BUDGET_TOKENS, budget_tokens // attempt)
        result = consolidator.run(
            MemoryConsolidationDeps(
                previous=state.consolidated,
                delta=delta,
                target_tokens=target,
                source_id=source,
            )
        ).strip()
        if estimate_tokens(result) <= budget_tokens:
            state.consolidated = result
            state.snapshot(memory)
            logger.debug(
                "TwoPhase: consolidated agentic digest for '%s' to ~%d tokens",
                source,
                estimate_tokens(result),
            )
            return result
    raise RuntimeError(
        f"agentic extraction digest for '{source}' cannot be consolidated below "
        f"{budget_tokens} tokens (window too small for the section size) - "
        "increase compile.llm.context_window or lower compile.budget.chunk_size_max"
    )
