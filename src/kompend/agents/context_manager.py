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

The ``sliding`` preset is the one exception, and it is still a
capability built here: ``build_sliding_compaction`` returns the zero-cost
``SlidingWindowCompaction`` with a *token* trigger (a fraction of the
window) and the message cap as a second bound, wrapped in
``GrowthGatedCompaction`` so the run's first user turn is pinned and an
unchanged context is not re-trimmed; every trim it performs is logged.
A message count alone cannot protect a token budget: a p863 vault set this
preset and the provider rejected one request at 90 154 actual tokens
against a 65 536-token window.

How the estimate is defended (``_conservative_tokens``): the harness anchors
``estimate_context_tokens`` on the newest provider-reported usage, so most of
any history is measured exactly and only the messages after that anchor are
estimated. Those messages are the dense ones — tool returns, tables, retry
echoes — and the harness charges them 4 characters per token. Measured on the
p863 corpus (733 request tails) the true density is a median 3.86 chars per
token with a p01 of 2.65, so kompend passes its own callable at 2.5 chars per
token: 99.0% of measured tails come out at or above their real cost, and the
seven that do not are off by at most 1.27x, inside the trigger's headroom.
Every strategy kompend builds gets it, so a tier's trigger, its reclaim
accounting and its logged figures cannot disagree about how big the context
is.

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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from math import ceil
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
    estimate_message_tokens,
    is_pinned,
    pin,
    resolve_context_window,
)
from pydantic_core import ValidationError

from kompend.logging import get_logger

#: Sliding-window defaults (zero-cost, no LLM call). Owned here so the
#: ``sliding`` preset and the summarizer fallback cannot drift apart: the
#: fallback is defined as "do what the sliding preset does".
_SLIDING_MAX_MESSAGES: Final[int] = 100
_SLIDING_KEEP_MESSAGES: Final[int] = 50

#: Token trigger of the ``sliding`` preset: the fraction of the window at
#: which the trim runs. The estimate itself is now conservative (see
#: ``_CONSERVATIVE_CHARS_PER_TOKEN``), so this headroom no longer carries the
#: whole error budget: it also absorbs the drift between what a server
#: advertises and what it enforces — one p863 server served 64 000 tokens,
#: enforced 65 536, and the run was configured for 128 000.
_SLIDING_COMPRESS_THRESHOLD: Final[float] = 0.8

#: Characters per token kompend assumes where the provider's own count is not
#: available. The harness defaults to 4, which is honest for English prose and
#: optimistic for everything else a run puts in a request.
#:
#: Measured on the p863 corpus (733 request tails: the context growth between
#: two responses that each reported usage, minus the later response's output),
#: reproducible with ``.kompend/tmp/measure_token_density.py``.
#: p01 2.65, p05 2.95, p10 3.18, median 3.86 chars per token. Tool returns —
#: JSON, tables, standard clauses — carry the dense end. At the harness's 4
#: the estimate under-counts 560 of those 733 tails (76%), worst by 2.02x; at
#: this value it covers 726 of 733 (99.0%) and the seven it leaves under-count
#: by at most 1.27x, which the trigger's headroom absorbs.
#:
#: Erring high is the cheap failure — a trim slightly early costs prompt-cache
#: writes; a trim late is a provider 400 after minutes of upstream work, and
#: the anchored part of the estimate (below) is exact, so only the tail can
#: drift. The cost of the margin: a ``keep_tokens`` tail is counted in these
#: units, so a named 8 000-token tail retains roughly 5 200 real tokens of the
#: same text (it held ~8 300 at 4 chars/token).
_CONSERVATIVE_CHARS_PER_TOKEN: Final[float] = 2.5

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


def _conservative_tokens(text: str) -> int:
    """Count *text* at kompend's measured density floor, never the harness's 4.

    The harness calls this only for the part of a request no provider count
    covers — the messages after the newest usage-carrying response, and a
    history with no usage at all. That is exactly where dense content lives,
    and where under-counting lets a history blow the window.
    """
    return ceil(len(text) / _CONSERVATIVE_CHARS_PER_TOKEN)


@dataclass
class GrowthGatedCompaction(AbstractCapability[AgentDepsT]):
    """Run the wrapped tiered stack only when the context grew.

    Compaction rewrites older messages, which invalidates the provider's
    prompt cache from the rewrite point onward. Re-firing on a static
    context would re-fold the same material on every request and pay that
    cache-write each time, reclaiming nothing. This wrapper tracks the
    token count after the last pass that was allowed through the gate
    (``_baseline_tokens``) and skips the wrapped strategy — returning the
    request context unchanged, raising nothing — until the history grew by
    ``min_growth_tokens`` since that pass. A skipped request never moves the
    baseline: growth accumulates across skips, so a history that gains less
    than the threshold per request still re-arms the stack once the total
    since the last pass reaches it. A skip is not an error, so a surrounding
    ``FallbackCompaction`` still reserves its fallback for genuine strategy
    failures.

    The wrapper marks the first user turn as pinned before delegating (see
    ``_pin_first_user_turn``), so the tiers and the sliding fallback keep
    the run's task assignment verbatim.
    """

    strategy: TieredCompaction[AgentDepsT]
    """The tiered stack to gate. Its own trigger decides when it would fire."""

    min_growth_tokens: int = _DEFAULT_MIN_GROWTH_TOKENS
    """Token growth since the last pass through the gate that re-arms the stack."""

    _baseline_tokens: int | None = field(default=None, repr=False, compare=False)
    """Estimated tokens after the last pass allowed through the gate.

    ``None`` means no pass has run yet, so the first request always passes.
    A skipped request leaves it alone so growth keeps accumulating.
    """

    tokenizer: Callable[[str], int] = _conservative_tokens
    """Counts *text* for every estimate this gate compares.

    Both operands come from the same callable, so the growth this gate measures
    and the ``min_growth_tokens`` it compares against stay on one scale — the
    conservative one, not the harness's optimistic four-chars-per-token default.
    """

    def with_focus(self, focus: str) -> GrowthGatedCompaction[AgentDepsT]:
        """Forward the focus to the wrapped stack's focus-capable tiers."""
        inner = self.strategy.with_focus(focus) if isinstance(self.strategy, SupportsFocus) else self.strategy
        focused: GrowthGatedCompaction[AgentDepsT] = GrowthGatedCompaction(
            strategy=inner,
            min_growth_tokens=self.min_growth_tokens,
            tokenizer=self.tokenizer,
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
        self._baseline_tokens = estimate_context_tokens(compacted, self.tokenizer)
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
            return request_context
        request_context = await self.strategy.before_model_request(ctx, request_context)
        self._baseline_tokens = estimate_context_tokens(ctx.messages, self.tokenizer)
        return request_context

    def _gate_open(self, pinned: list[ModelMessage]) -> bool:
        """Whether the context grew enough since the last pass through the gate."""
        if self._baseline_tokens is None:
            return True
        growth = estimate_context_tokens(pinned, self.tokenizer) - self._baseline_tokens
        if growth < self.min_growth_tokens:
            logger.debug(
                "Compaction skipped: context grew %d token(s) since the last pass (< %d)",
                growth,
                self.min_growth_tokens,
            )
            return False
        return True


@dataclass
class _GuardedSlidingCompaction(SlidingWindowCompaction):
    """The ``sliding`` preset: a token-triggered trim that keeps the task.

    ``SlidingWindowCompaction`` alone fires on a message count, which cannot
    protect a token budget: a long tool-call history reaches many times the
    window in far fewer than 100 messages. The p863 vault ran this preset and
    one request was rejected at 90 154 actual tokens against a 65 536-token
    window. This subclass adds the three halves of that fix:

    * **Token trigger.** ``max_fraction`` trims at a fraction of the
      request's window (``_SLIDING_COMPRESS_THRESHOLD``); the message count
      stays as a second, coarser bound.
    * **Pinned task assignment.** The first user turn is marked with the pin
      contract before the trim (``_pin_first_user_turn``), so the trimmed
      history keeps the run's task verbatim.
    * **A record.** Every trim logs kept/dropped messages and the token
      estimate before/after; a trim that reclaims nothing says so, and a trim
      that leaves more than the trigger behind says that too — the p863 tail
      held one 380 538-char retry-feedback part, so dropping every older
      message could never have brought it under budget.

    Every count this class compares comes from ``_conservative_tokens``, so
    the trigger, the recorded figures and ``keep_tokens`` all speak the same
    (measured, pessimistic) density instead of the harness's four
    characters-per-token default.

    ``min_growth_tokens`` gates repeats exactly as in
    ``GrowthGatedCompaction``: a trim normally lands the history far below
    the trigger, but a protected tail close to the trigger leaves it above,
    and re-trimming the same material every request would pay the prompt
    cache write for nothing.
    """

    min_growth_tokens: int = _DEFAULT_MIN_GROWTH_TOKENS
    """Token growth since the last trim that lets the next one run."""

    _baseline_tokens: int | None = field(default=None, repr=False, compare=False)
    """Estimated tokens after the last trim; ``None`` until one ran."""

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Trim, then record what was kept, what was dropped, and the cost."""
        before_messages = len(messages)
        before_tokens = estimate_context_tokens(messages, self.tokenizer)
        compacted = await super().compact(messages, ctx)
        after_tokens = estimate_context_tokens(compacted, self.tokenizer)
        dropped = before_messages - len(compacted)
        if dropped > 0:
            # Only an effective trim moves the baseline: a no-op trim rewrote
            # nothing, so the next request must be free to try again.
            self._baseline_tokens = after_tokens
            logger.info(
                "Sliding-window trim: dropped %d of %d message(s), kept %d; ~%d → ~%d token(s)",
                dropped,
                before_messages,
                len(compacted),
                before_tokens,
                after_tokens,
            )
            self._warn_if_still_over(compacted, after_tokens, ctx)
        else:
            logger.warning(
                "Sliding-window trim reclaimed nothing: %d message(s), ~%d token(s) survive the %s-token tail",
                len(compacted),
                after_tokens,
                self.keep_tokens,
            )
        return compacted

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Pin the task, apply the growth gate, then let the trim decide."""
        pinned = _pin_first_user_turn(list(ctx.messages))
        if pinned is not ctx.messages:
            ctx.messages[:] = pinned
        fraction = self.max_fraction
        if fraction is None:  # not the shape this build produces; stay passive
            return await super().before_model_request(ctx, request_context)
        window = self.context_window or resolve_context_window(request_context.model) or self.fallback_context_window
        trigger = max(1, int(window * fraction))
        tokens = estimate_context_tokens(
            pinned,
            self.tokenizer,
            model_request_parameters=request_context.model_request_parameters,
        )
        over_messages = self.max_messages is not None and len(pinned) > self.max_messages
        if tokens <= trigger and not over_messages:
            return request_context
        if self._baseline_tokens is not None and tokens - self._baseline_tokens < self.min_growth_tokens:
            logger.debug(
                "Sliding-window trim skipped: history grew %d token(s) since the last trim (< %d)",
                tokens - self._baseline_tokens,
                self.min_growth_tokens,
            )
            return request_context
        return await super().before_model_request(ctx, request_context)

    def _warn_if_still_over(
        self,
        compacted: list[ModelMessage],
        after_tokens: int,
        ctx: RunContext[AgentDepsT],
    ) -> None:
        """Say so when the retained tail alone still exceeds the trigger.

        A trim that drops messages yet leaves the history over the trigger
        cannot do better on the next request: the tail is what it is. That is
        the p863 shape — one 380 538-char retry-feedback part made up 39% of
        the request — and without this line the run looks trimmed and healthy
        right up to the provider's 400.
        """
        fraction = self.max_fraction
        if fraction is None:
            return
        window = self.context_window or resolve_context_window(ctx.model) or self.fallback_context_window
        trigger = max(1, int(window * fraction))
        if after_tokens <= trigger:
            return
        largest = max(estimate_message_tokens(compacted, self.tokenizer), default=0)
        logger.warning(
            "Sliding-window trim kept %d message(s) still at ~%d token(s) against a %d-token trigger "
            "(largest ~%d token(s)); the retained tail cannot reclaim more",
            len(compacted),
            after_tokens,
            trigger,
            largest,
        )


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
            tokenizer=_conservative_tokens,
        ),
        DeduplicateFileReads(
            file_key=_file_key,
            max_fraction=compress_threshold,
            context_window=context_window,
            tokenizer=_conservative_tokens,
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
                tokenizer=_conservative_tokens,
            ),
        )
    tiers.append(summarizer)
    gated = GrowthGatedCompaction(
        strategy=TieredCompaction(
            tiers=tiers,
            target_fraction=compress_threshold,
            context_window=context_window,
            tokenizer=_conservative_tokens,
        ),
        min_growth_tokens=min_growth_tokens if min_growth_tokens is not None else _DEFAULT_MIN_GROWTH_TOKENS,
        tokenizer=_conservative_tokens,
    )
    return _PinningFallbackCompaction(
        fallback_chain=[
            gated,
            _LoggedSlidingWindowCompaction(
                max_fraction=compress_threshold,
                context_window=context_window,
                keep_tokens=int((context_window or DEFAULT_CONTEXT_WINDOW) * compress_threshold * _FALLBACK_KEEP_SHARE),
                tokenizer=_conservative_tokens,
            ),
        ],
        fallback_on=(UnexpectedModelBehavior,),
        max_fraction=compress_threshold,
        context_window=context_window,
    )


def build_sliding_compaction(
    *,
    context_window: int | None,
    compress_threshold: float,
    keep_messages: int,
    keep_tokens: int | None = None,
    min_growth_tokens: int | None = None,
) -> SlidingWindowCompaction:
    """Build the zero-cost ``sliding`` preset: a token-triggered trim.

    The trim keeps *message content* tokens, never a message count alone, and
    is gated so a history the tail cannot shrink is not re-trimmed on every
    request (see ``_GuardedSlidingCompaction``). Every trigger, tail and log
    figure it reports is counted with ``_conservative_tokens``, kompend's
    measured density floor, never the harness's four-chars-per-token default.

    Parameters
    ----------
    context_window:
        The model's input window; the trigger is a fraction of it.
    compress_threshold:
        Fraction of the window at which the trim runs. Well below 1.0: it
        absorbs a server that enforces a smaller window than kompend was
        configured for, so a trim that lands near the limit still arrives
        before the provider rejects.
    keep_messages:
        Messages retained when no token tail applies (``keep_tokens`` of 0).
        It also stays the *retention* default; the trigger cap is
        ``_SLIDING_MAX_MESSAGES``.
    keep_tokens:
        Recent-tail budget (``CompactionConfig.keep_tokens``). Capped at half
        the trigger, so a tail sized for a large window cannot exceed the
        budget of a small one and make the trim a no-op. ``0`` disables the
        token tail and retains *keep_messages* instead.
    min_growth_tokens:
        Token growth that re-arms the trim
        (``CompactionConfig.min_growth_tokens``).
    """
    window = context_window or DEFAULT_CONTEXT_WINDOW
    headroom = int(window * compress_threshold * _FALLBACK_KEEP_SHARE)
    tail = min(keep_tokens, headroom) if keep_tokens else 0
    return _GuardedSlidingCompaction(
        max_messages=_SLIDING_MAX_MESSAGES,
        max_fraction=compress_threshold,
        context_window=context_window,
        keep_messages=keep_messages,
        keep_tokens=tail or None,
        min_growth_tokens=min_growth_tokens if min_growth_tokens is not None else _DEFAULT_MIN_GROWTH_TOKENS,
        tokenizer=_conservative_tokens,
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
            tokenizer=_conservative_tokens,
        )
    return _LoggedSummarizingCompaction(
        model=model,
        max_fraction=compress_threshold,
        keep_messages=keep_messages,
        context_window=context_window,
        instructions=instructions,
        tokenizer=_conservative_tokens,
    )
