"""Context compaction for kompend agents, on ``pydantic_ai_harness.compaction``.

Rank-based, tiered stack. The capability ``build_fallback_compaction``
returns is a ``FallbackCompaction`` whose first tier is a growth-gated
``TieredCompaction`` — cheap zero-cost passes first (clear old tool
results, blank superseded file reads), an optional judge-ranked
``RelevanceCompaction`` pass, the ``SummarizingCompaction`` last — and
whose fallback tier is the zero-cost ``SlidingWindowCompaction``.

Why the fallback stays: the summarizer runs in a nested, toolless agent
that never gets an explicit retry budget (harness 0.64,
``_summarizing_compaction.py``), so it inherits ``ToolManager``'s default
of one retry; since it is fed a transcript of the run's own tool calls, a
small model imitates one, the single unknown-tool refusal is consumed, and
``UnexpectedModelBehavior`` escapes ``before_model_request`` and kills the
parent run (pydantic/pydantic-ai#12307). ``FallbackCompaction`` keeps that
from being fatal: when the summarizer cannot answer, the oldest messages
are dropped instead and the run continues. Only
``UnexpectedModelBehavior`` (retry exhaustion, unusable model behaviour)
falls through; provider and programming errors propagate unchanged.

Why the growth gate: a compaction that re-fires on a static large context
re-folds the same material every request and destroys the prompt-cache
prefix. ``GrowthGatedCompaction`` therefore runs the tiered stack only
when the context grew since the last compaction (tracked in
``_baseline_tokens``); a skipped gate is not an error, so the sliding
fallback stays reserved for genuine summarizer failures.

NEVER-CROP reconciliation (``.agents/CODE.md``): compaction folds only
re-derivable conversation history. Source text never rides in prompts
alone — agents re-read it through their tools — so no source, chunk or
result set is dropped by this change. Load-bearing state survives
verbatim: the first user turn (which carries the run's task assignment)
is marked with the harness's pin contract before any tier runs, and every
shipped strategy re-injects pinned content; state kompend carries in deps
(``pinned_facts``, ``known_titles``, the extraction memory digest, the
plan) is re-rendered into each run's prompt and never depends on history.
The summarizer's instructions come from the doctrine template
(``compaction/doctrine.md.j2``), which spells out what must survive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import INVALID_JSON_KEY, ModelMessage, ModelRequest, ToolCallPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.compaction import (
    DEFAULT_CONTEXT_WINDOW,
    ClearToolResults,
    CompactionStrategy,
    DeduplicateFileReads,
    FallbackCompaction,
    RelevanceCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    SupportsFocus,
    TieredCompaction,
    estimate_context_tokens,
    is_pinned,
    pin,
)
from pydantic_core import ValidationError

from kompend.logging import get_logger

#: Sliding-window defaults (zero-cost, no LLM call). Owned here so the
#: ``sliding`` preset and the summarizer fallback cannot drift apart: the
#: fallback is defined as "do what the sliding preset does".
_SLIDING_MAX_MESSAGES: Final[int] = 100
_SLIDING_KEEP_MESSAGES: Final[int] = 50

#: Share of the compaction trigger the fallback retains as message-content
#: tokens. Below 1.0 so the fallback actually reclaims room — a result still
#: above the trigger would compact again on every request.
_FALLBACK_KEEP_SHARE: Final[float] = 0.5

#: Tool-call/return pairs ``ClearToolResults`` leaves intact so the newest
#: reads stay reviewable by the model while only superseded content is
#: blanked.
_CLEAR_KEEP_PAIRS: Final[int] = 5

#: Default recent tail, in message-content tokens, the relevance judge may
#: never touch. Mirrors ``CompactionConfig.keep_tokens``.
_DEFAULT_KEEP_TOKENS: Final[int] = 8_000

#: Default token growth that re-arms the tiered stack. Mirrors
#: ``CompactionConfig.min_growth_tokens``.
_DEFAULT_MIN_GROWTH_TOKENS: Final[int] = 2_048

#: Read-only tools whose result for identical arguments is identical, so an
#: earlier read is safely blanked once a later read of the same key exists.
#: Queues and writes are absent on purpose: their results differ per call.
#: Tool names are code identifiers shared with ``kompend.compilation``'s
#: tool builders, not prompt text.
_IDEMPOTENT_READ_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "get_claim",  # verification
        "get_figures",  # query
        "get_group",  # distill
        "get_tables",  # query
        "list_known_titles",  # generation
        "list_pages",  # overview
        "list_sources",  # write
        "read_page",  # query / overview
        "read_related",  # write
        "read_segment",  # generation
        "read_source",  # write
        "read_source_window",  # generation
        "search_sources",  # generation / verification
        "search_wiki",  # query
    }
)

logger = get_logger(__name__)


@dataclass
class GrowthGatedCompaction(AbstractCapability[AgentDepsT]):
    """Run the wrapped tiered stack only when the context grew.

    Compaction rewrites older messages, which invalidates the provider's
    prompt cache from the rewrite point onward. Re-firing on a static
    context would re-fold the same material on every request and pay that
    cache-write each time, reclaiming nothing. This wrapper tracks the
    token count at the last compaction (``_baseline_tokens``) and skips the
    wrapped strategy — returning the request context unchanged, raising
    nothing — until the history grew by ``min_growth_tokens`` again. A skip
    is not an error, so a surrounding ``FallbackCompaction`` still reserves
    its fallback for genuine strategy failures.

    The wrapper marks the first user turn as pinned before delegating (see
    ``_pin_first_user_turn``), so the tiers and the sliding fallback keep
    the run's task assignment verbatim.
    """

    strategy: TieredCompaction[AgentDepsT]
    """The tiered stack to gate. Its own trigger decides when it would fire."""

    min_growth_tokens: int = _DEFAULT_MIN_GROWTH_TOKENS
    """Token growth since the last compaction that re-arms the stack."""

    _baseline_tokens: int | None = field(default=None, repr=False, compare=False)
    """Estimated tokens after the last gated compaction; ``None`` until then."""

    def with_focus(self, focus: str) -> GrowthGatedCompaction[AgentDepsT]:
        """Forward the focus to the wrapped stack's focus-capable tiers."""
        inner = self.strategy.with_focus(focus) if isinstance(self.strategy, SupportsFocus) else self.strategy
        focused: GrowthGatedCompaction[AgentDepsT] = GrowthGatedCompaction(
            strategy=inner,
            min_growth_tokens=self.min_growth_tokens,
        )
        focused._baseline_tokens = self._baseline_tokens
        return focused

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Pin then gate — the path ``CompactOnRequest`` takes.

        The model's explicit request still passes the growth gate: a request
        made right after a fold would otherwise re-fold the same material and
        pay the cache-write twice for nothing.
        """
        pinned = _pin_first_user_turn(messages)
        if not self._gate_open(pinned):
            return pinned
        compacted = await self.strategy.compact(pinned, ctx)
        self._baseline_tokens = estimate_context_tokens(compacted)
        return compacted

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Pin the task assignment, then gate the wrapped stack's trigger."""
        pinned = _pin_first_user_turn(list(ctx.messages))
        ctx.messages[:] = pinned
        if not self._gate_open(pinned):
            self._baseline_tokens = estimate_context_tokens(pinned)
            return request_context
        request_context = await self.strategy.before_model_request(ctx, request_context)
        self._baseline_tokens = estimate_context_tokens(ctx.messages)
        return request_context

    def _gate_open(self, pinned: list[ModelMessage]) -> bool:
        """Whether enough new material arrived since the last compaction."""
        if self._baseline_tokens is None:
            return True
        growth = estimate_context_tokens(pinned) - self._baseline_tokens
        if growth < self.min_growth_tokens:
            logger.debug(
                "Compaction skipped: context grew %d token(s) since the last pass (< %d)",
                growth,
                self.min_growth_tokens,
            )
            return False
        return True


class _LoggedSummarizingCompaction(SummarizingCompaction):
    """``SummarizingCompaction`` that logs its compression events."""

    # pydantic_ai_harness 0.62 hands ``_summarize`` a ``WrittenSummary``
    # record (``.text``/``.created_at``) instead of the plain ``str`` of
    # 0.61 — the return type follows whichever harness is installed.
    async def _summarize(self, messages: list[ModelMessage], ctx: Any, previous_summary: str | None = None) -> Any:
        logger.info("Context compression starting: %d message(s)", len(messages))
        summary = await super()._summarize(messages, ctx, previous_summary=previous_summary)
        text = summary if isinstance(summary, str) else summary.text
        logger.info("Context compression produced %d-char summary", len(text))
        return summary


class _LoggedSlidingWindowCompaction(SlidingWindowCompaction):
    """``SlidingWindowCompaction`` that logs when it trims as a fallback.

    It only ever runs after the summarizer failed, so the warning is the
    record of that failure — the run continues, the context is degraded.
    """

    async def compact(self, messages: list[ModelMessage], ctx: RunContext[Any]) -> list[ModelMessage]:
        logger.warning(
            "Summarizer failed; falling back to sliding-window trimming of %d message(s)",
            len(messages),
        )
        compacted = await super().compact(messages, ctx)
        logger.warning("Sliding-window fallback kept %d of %d message(s)", len(compacted), len(messages))
        return compacted


class _PinningFallbackCompaction(FallbackCompaction):
    """``FallbackCompaction`` that pins the task assignment before the chain runs.

    The harness re-injects pinned parts from the *original* history after a
    chain entry succeeds (``_compact_pinned``), so the pin has to be on
    ``ctx.messages`` before the chain is entered — pinning inside a chain
    entry is invisible to that re-injection and the sliding fallback would
    drop the task assignment along with the trimmed history.
    """

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        pinned = _pin_first_user_turn(list(ctx.messages))
        if pinned is not ctx.messages:
            ctx.messages[:] = pinned
        return await super().before_model_request(ctx, request_context)


def _pin_first_user_turn(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """Return *messages* with the first user turn marked pinned (idempotent).

    Uses the harness pin contract — model-invisible metadata, no content
    change — so every shipped strategy keeps that turn verbatim, the sliding
    fallback included. Only string user prompts are marked; structured
    content keeps its parts untouched (the strategies' own head protection
    still applies).
    """
    for index, message in enumerate(messages):
        if not (isinstance(message, ModelRequest) and any(isinstance(part, UserPromptPart) for part in message.parts)):
            continue
        parts = [
            pin(part.content) if isinstance(part, UserPromptPart) and not is_pinned(part) and isinstance(part.content, str) else part for part in message.parts
        ]
        if parts == list(message.parts):
            return list(messages)
        pinned = list(messages)
        pinned[index] = replace(message, parts=parts)
        return pinned
    return list(messages)


def build_fallback_compaction(
    *,
    context_window: int | None,
    compress_threshold: float,
    keep_messages: int,
    model: Any = None,
    judge: Model | None = None,
    keep_tokens: int | None = None,
    min_growth_tokens: int | None = None,
    instructions: str | None = None,
) -> FallbackCompaction:
    """Build the rank-based, tiered compaction capability kompend attaches.

    Tier order, cheap to expensive: zero-cost clearing of old tool
    results, zero-cost deduplication of superseded file reads, the optional
    judge-ranked ``RelevanceCompaction`` (only when *judge* is given), and
    the summarizer last. ``TieredCompaction`` re-measures between tiers and
    stops once the history fits, so the expensive tiers run only when the
    cheap passes cannot reclaim enough. The whole stack is wrapped in
    ``GrowthGatedCompaction`` (see its docstring) and backed by the
    zero-cost sliding window for summarizer failures (see the module
    docstring).

    Parameters
    ----------
    context_window:
        The model's input window; every trigger is a fraction of it.
    compress_threshold:
        Fraction of the window at which compaction runs. The single
        fraction the whole stack derives from: tier target, judge trigger
        and fallback trigger all read it.
    keep_messages:
        How many recent messages survive a *successful* summary verbatim.
    model:
        The agent's model, reused for the summary so both run on the same
        failover chain.
    judge:
        Decision model that ranks steps keep/reduce/drop; ``None`` skips
        the relevance tier.
    keep_tokens:
        Recent tail the judge may never touch (``CompactionConfig.keep_tokens``).
    min_growth_tokens:
        Token growth that re-arms the tiered stack
        (``CompactionConfig.min_growth_tokens``).
    instructions:
        The summarizer's doctrine; ``None`` keeps the harness default.
    """
    summarizer = build_summarizing_compaction(
        context_window=context_window,
        compress_threshold=compress_threshold,
        keep_messages=keep_messages,
        model=model,
        instructions=instructions,
    )
    tiers: list[CompactionStrategy[Any]] = [
        ClearToolResults(
            max_fraction=compress_threshold,
            context_window=context_window,
            keep_pairs=_CLEAR_KEEP_PAIRS,
        ),
        DeduplicateFileReads(
            file_key=_file_key,
            max_fraction=compress_threshold,
            context_window=context_window,
        ),
    ]
    if judge is not None:
        tiers.append(
            RelevanceCompaction(
                model=judge,
                summarizer=summarizer,
                max_fraction=compress_threshold,
                target_fraction=compress_threshold,
                context_window=context_window,
                keep_tokens=keep_tokens if keep_tokens is not None else _DEFAULT_KEEP_TOKENS,
            ),
        )
    tiers.append(summarizer)
    gated = GrowthGatedCompaction(
        strategy=TieredCompaction(
            tiers=tiers,
            target_fraction=compress_threshold,
            context_window=context_window,
        ),
        min_growth_tokens=min_growth_tokens if min_growth_tokens is not None else _DEFAULT_MIN_GROWTH_TOKENS,
    )
    return _PinningFallbackCompaction(
        fallback_chain=[
            gated,
            _LoggedSlidingWindowCompaction(
                max_fraction=compress_threshold,
                context_window=context_window,
                keep_tokens=int((context_window or DEFAULT_CONTEXT_WINDOW) * compress_threshold * _FALLBACK_KEEP_SHARE),
            ),
        ],
        fallback_on=(UnexpectedModelBehavior,),
        max_fraction=compress_threshold,
        context_window=context_window,
    )


def _file_key(call: ToolCallPart) -> str | None:
    """Stable identity of an idempotent read tool call, else ``None``.

    The key includes the arguments, so windowed reads (``offset``) of the
    same source keep distinct results and only a true re-read of the same
    window is deduplicated.
    """
    if call.tool_name not in _IDEMPOTENT_READ_TOOLS:
        return None
    try:
        args = call.args_as_dict()
    except ValidationError:  # unparseable args cannot be identified safely
        return None
    if INVALID_JSON_KEY in args:  # the harness's marker for raw, unparsed args
        return None
    return f"{call.tool_name}:{sorted(args.items())!r}"


def build_summarizing_compaction(
    *,
    context_window: int | None,
    compress_threshold: float,
    keep_messages: int,
    model: Any = None,
    instructions: str | None = None,
) -> SummarizingCompaction:
    """Build the LLM compaction capability from kompend's config values.

    Parameters
    ----------
    context_window:
        The model's input window; the trigger is a fraction of it.
    compress_threshold:
        Fraction of the window at which compaction runs (``max_fraction``).
    keep_messages:
        How many recent messages survive compaction verbatim.
    model:
        The agent's model, reused for the summary so both run on the same
        failover chain.
    instructions:
        Doctrine for the summarizer agent; ``None`` keeps the harness
        default. kompend passes the rendered ``compaction/doctrine.md.j2``.
    """
    if instructions is None:
        return _LoggedSummarizingCompaction(
            model=model,
            max_fraction=compress_threshold,
            keep_messages=keep_messages,
            context_window=context_window,
        )
    return _LoggedSummarizingCompaction(
        model=model,
        max_fraction=compress_threshold,
        keep_messages=keep_messages,
        context_window=context_window,
        instructions=instructions,
    )
