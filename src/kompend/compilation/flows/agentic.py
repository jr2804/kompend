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
from typing import Final, Protocol

from pydantic_ai import Tool
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

from kompend.agents.deps import MemoryConsolidationDeps, PipelineAgentDeps
from kompend.agents.extraction import ExtractionAgent
from kompend.agents.memory_consolidation import MemoryConsolidationAgent
from kompend.compilation._two_phase_core import (
    Extraction,
    ProgressFn,
    TwoPhaseRuntime,
)
from kompend.compilation.asset_tools import build_asset_tools
from kompend.compilation.extraction_memory import ExtractionMemory
from kompend.compilation.schema import ExtractionOutput
from kompend.core.models.enums import CacheDeps
from kompend.core.text.windows import split_text_windows
from kompend.llm.context_window import ContextWindow
from kompend.llm.overflow import is_context_overflow, stated_context_size
from kompend.llm.request_budget import RequestBudget, chars_to_tokens

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


class Consolidator(Protocol):
    """Anything that merges a digest delta in one typed call.

    The flows only need ``run``; :class:`MemoryConsolidationAgent`
    satisfies this structurally, and so do scripted test doubles.
    """

    def run(self, deps: MemoryConsolidationDeps) -> str: ...


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
    - A provider context-overflow 400 is never retried unchanged: the window
      is lowered to the limit the provider named and the section is split
      into windows that fit, so its content is still extracted. Only a
      failure that cannot be split (no stated limit, or the window is already
      at that limit) skips the section.
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
    agent, consolidator = _section_agents(compiler, tools)

    memory = ExtractionMemory()
    consolidation = _RollingConsolidation()
    cpt = compiler._config.llm.default_chars_per_token
    static_tokens = compiler._extract_system_tokens + (chars_to_tokens(len(_TOOLS_PREAMBLE), chars_per_token=cpt) if tools else 0)
    batch_retries = compiler._config.llm.batch_retry_attempts
    overall_succeeded = False
    sections_done = 0

    # Sections still to extract. A section the provider rejects as too large
    # is split and put back at the front, so this list - and the bar's total -
    # can grow mid-run.
    pending: list[str] = list(texts)

    def _section_total() -> int:
        """Units the bar counts: finished sections plus the pending ones."""
        return sections_done + len(pending)

    def _activity(label: str) -> None:
        """Name what the running call is doing, for the progress bar."""
        if progress is not None:
            progress(sections_done, _section_total(), label)

    while pending:
        section_text = pending.pop(0)
        logger.debug(
            "TwoPhase: agentic section %d/%d for '%s'",
            sections_done + 1,
            _section_total(),
            source,
        )

        message_history = _section_message_history(
            compiler,
            memory,
            consolidation,
            consolidator,
            source,
            section_text,
            static_tokens,
            cpt,
            include_tool_preamble=bool(tools),
            activity=_activity,
        )

        deps = PipelineAgentDeps(text=section_text, source_id=f"{source}:{sections_done}")

        validated: ExtractionOutput | None = None
        replanned = False
        for attempt in range(1 + batch_retries):
            try:
                validated, _new_messages = agent.run_with_messages(
                    deps,
                    message_history=message_history or None,
                )
            except Exception as exc:
                if _is_context_overflow(exc):
                    # The unchanged payload can never fit, but its content can:
                    # lower the window to the provider's limit and split the
                    # section into windows that do. Retrying it unchanged is
                    # the only guaranteed waste.
                    windows = _replan_section(compiler, section_text, static_tokens, cpt, exc)
                    if windows is not None:
                        logger.warning(
                            "TwoPhase: agentic section %d for '%s' exceeds the %d-token window (%s) - split into %d window(s) and re-planned",
                            sections_done + 1,
                            source,
                            compiler._context_window.max_input,
                            exc,
                            len(windows),
                        )
                        # The split windows go back to the front of the queue:
                        # they replace the section that just overflowed.
                        pending = [*windows, *pending]
                        replanned = True
                        validated = None
                        # Both agents baked the old window in at construction
                        # (compaction thresholds, output ceiling); rebuild
                        # them against the window the provider just named.
                        agent, consolidator = _section_agents(compiler, tools)
                        break
                    logger.error(
                        "TwoPhase: agentic section %d for '%s' exceeds the context window "
                        "(%s) - cannot split it smaller; skipping; consider a larger "
                        "compile.llm.context_window or smaller chunk sizes",
                        sections_done + 1,
                        source,
                        exc,
                    )
                    validated = None
                    break
                logger.debug(
                    "TwoPhase: agentic section %d attempt %d failed: %s",
                    sections_done + 1,
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
                    sections_done,
                )

        if replanned:
            # The section is queued as smaller windows; it did not fail.
            continue

        compiler._batch_tracker.record(failed=validated is None)
        sections_done += 1
        if progress is not None:
            progress(sections_done, _section_total(), "")

        if validated is None:
            logger.warning(
                "TwoPhase: agentic section %d failed for '%s' — skipping",
                sections_done,
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


def _section_message_history(
    compiler: TwoPhaseRuntime,
    memory: ExtractionMemory,
    consolidation: _RollingConsolidation,
    consolidator: Consolidator,
    source: str,
    section_text: str,
    static_tokens: int,
    chars_per_token: float,
    *,
    include_tool_preamble: bool,
    activity: Callable[[str], None],
) -> list[ModelMessage]:
    """The message history for one section: tool preamble + running digest.

    The digest is sized against the window for THIS section; once the full
    digest outgrows the budget it is model-consolidated, never cropped.
    """
    budget_tokens = _digest_budget_tokens(compiler._context_window, static_tokens, section_text, chars_per_token=chars_per_token)
    logger.info(
        "TwoPhase: agentic digest budget (%s) -> %d tokens for the digest",
        _digest_budget_report(compiler._context_window, static_tokens, section_text, chars_per_token=chars_per_token),
        budget_tokens,
    )
    memory_digest = _running_digest(
        memory,
        consolidation,
        budget_tokens,
        consolidator,
        source,
        chars_per_token=chars_per_token,
        activity=activity,
    )
    history: list[ModelMessage] = []
    if include_tool_preamble:
        history.append(ModelRequest(parts=[SystemPromptPart(content=_TOOLS_PREAMBLE)]))
    if memory_digest.strip():
        history.append(ModelRequest(parts=[SystemPromptPart(content=f"=== Context from prior sections ===\n{memory_digest}")]))
    return history


def _full_text_hash(texts: list[str]) -> str:
    """SHA-256 of concatenated section texts for cache key."""
    return hashlib.sha256("".join(texts).encode()).hexdigest()


def _digest_budget_tokens(
    context_window: ContextWindow,
    static_tokens: int,
    section_text: str,
    *,
    chars_per_token: float,
) -> int:
    """Tokens the prior-section digest may occupy in one section request.

    Mirrors the batch path's :func:`~kompend.compilation.flows.extraction._extract_char_budget`
    terms - window minus answer reserve, shared safety margin, measured
    system prompt and the section text itself - so the agentic path sizes
    against the same physics instead of discovering the window at the
    provider's 400.

    The arithmetic lives in :class:`~kompend.llm.request_budget.RequestBudget`,
    so this stage cannot drift from the batch path on what the window pays for.
    """
    budget = RequestBudget(
        window=context_window,
        reserved={
            "static": static_tokens,
            "section": chars_to_tokens(len(section_text), chars_per_token=chars_per_token),
        },
    )
    # Signed: a negative budget means the section itself does not fit the
    # window, which the caller reports as "no room for prior-section context".
    return budget.available_tokens


def _digest_budget_report(
    context_window: ContextWindow,
    static_tokens: int,
    section_text: str,
    *,
    chars_per_token: float,
) -> str:
    """One-line description of the digest budget, in the shared report shape."""
    return RequestBudget(
        window=context_window,
        reserved={
            "static": static_tokens,
            "section": chars_to_tokens(len(section_text), chars_per_token=chars_per_token),
        },
    ).report()


def _section_agents(
    compiler: TwoPhaseRuntime,
    tools: list[Tool[PipelineAgentDeps]] | None,
) -> tuple[ExtractionAgent, MemoryConsolidationAgent]:
    """The section and consolidation agents, sized against the current window.

    Built per window rather than once per source: a provider overflow lowers
    the window mid-source, and both agents turn it into harness compaction
    thresholds and the request's output ceiling when they are constructed. A
    stale value would keep compacting against a window the provider has
    already rejected.
    """
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
    consolidator = MemoryConsolidationAgent(
        model_string=compiler._model or "",
        timeout=compiler._timeout,
        reasoning_effort=compiler._config.llm.reasoning_effort,
        context_window=compiler._context_window.max_input,
        llm=compiler._config.llm,
    )
    return agent, consolidator


def _replan_section(
    compiler: TwoPhaseRuntime,
    section_text: str,
    static_tokens: int,
    chars_per_token: float,
    exc: BaseException,
) -> list[str] | None:
    """Split an overflowing section against the provider's stated limit.

    Returns the windows to process instead of the section, or ``None`` when
    the failure cannot be re-planned: the provider named no limit, the window
    is already at that limit, or the section already fits the re-planned
    budget. A ``None`` means the unchanged payload cannot be made to fit, so
    retrying it would only repeat the rejection.
    """
    limit = stated_context_size(exc)
    if limit is None:
        return None
    if not compiler._clamp_window(limit, reason=f"provider enforces {limit} tokens"):
        return None
    # What the section itself may carry: the window minus the answer reserve,
    # the safety margin and the static system prompt (the digest is re-sized
    # against the same window on the next pass).
    budget = RequestBudget(window=compiler._context_window, reserved={"static": static_tokens})
    windows = split_text_windows(section_text, budget.content_chars(chars_per_token))
    return windows if len(windows) > 1 else None


def _is_context_overflow(exc: BaseException) -> bool:
    """Whether *exc* is a context-window overflow reported by the provider.

    Delegates to :func:`kompend.llm.overflow.is_context_overflow` — one
    definition of the provider's markers, shared with the agent base and the
    window planner.
    """
    return is_context_overflow(exc)


def _running_digest(
    memory: ExtractionMemory,
    state: _RollingConsolidation,
    budget_tokens: int,
    consolidator: Consolidator,
    source: str,
    *,
    chars_per_token: float,
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
    if chars_to_tokens(len(full), chars_per_token=chars_per_token) <= budget_tokens:
        return full
    if not full:
        return ""
    logger.debug(
        "TwoPhase: digest for '%s' (~%d tokens) exceeds the %d-token budget - consolidating",
        source,
        chars_to_tokens(len(full), chars_per_token=chars_per_token),
        budget_tokens,
    )
    if activity is not None:
        activity("consolidating context")
    return _consolidate_digest(consolidator, memory, state, budget_tokens, source, chars_per_token=chars_per_token)


def _consolidate_digest(
    consolidator: Consolidator,
    memory: ExtractionMemory,
    state: _RollingConsolidation,
    budget_tokens: int,
    source: str,
    *,
    chars_per_token: float,
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
        if chars_to_tokens(len(result), chars_per_token=chars_per_token) <= budget_tokens:
            state.consolidated = result
            state.snapshot(memory)
            logger.debug(
                "TwoPhase: consolidated agentic digest for '%s' to ~%d tokens",
                source,
                chars_to_tokens(len(result), chars_per_token=chars_per_token),
            )
            return result
    raise RuntimeError(
        f"agentic extraction digest for '{source}' cannot be consolidated below "
        f"{budget_tokens} tokens (window too small for the section size) - "
        "increase compile.llm.context_window or lower compile.budget.chunk_size_max"
    )
