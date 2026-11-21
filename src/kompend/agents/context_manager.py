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

How the budget is derived (``usable_window_tokens``): a request does not own
the whole window — it also carries the answer reserve and the safety margin.
A trigger or a target is therefore a fraction of the *usable* window (window
minus answer reserve minus safety margin), not of the window itself: at
64 000 tokens the usable share is 50 688, at 1 000 000 tokens it is 911 808.
The formula is the contract; the 64 000-token case is only one instance of it.
Every bound kompend derives this way scales with the discovered window, so a
1M-context model gets a proportionally larger budget instead of a fixed one.

How the estimate is defended (``conservative_tokens``): the harness anchors
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
from typing import Any, Final

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    INVALID_JSON_KEY,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import AbstractModel, Model, ModelRequestContext
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

from kompend.core.text.density import conservative_tokens
from kompend.llm.context_window import answer_reserve_tokens
from kompend.llm.request_budget import usable_window_tokens
from kompend.logging import get_logger

#: Sliding-window defaults (zero-cost, no LLM call). Owned here so the
#: ``sliding`` preset and the summarizer fallback cannot drift apart: the
#: fallback is defined as "do what the sliding preset does".
_SLIDING_MAX_MESSAGES: Final[int] = 100
_SLIDING_KEEP_MESSAGES: Final[int] = 50

#: Token trigger of the ``sliding`` preset: the fraction of the window at
#: which the trim runs. The estimate itself is now conservative (see
#: ``CONSERVATIVE_CHARS_PER_TOKEN``), so this headroom no longer carries the
#: whole error budget: it also absorbs the drift between what a server
#: advertises and what it enforces — one p863 server served 64 000 tokens,
#: enforced 65 536, and the run was configured for 128 000.
_SLIDING_COMPRESS_THRESHOLD: Final[float] = 0.8


#: Characters per token kompend assumes where the provider's own count is not
#: available. The number, the measurement behind it and the counting function
#: live in :mod:`kompend.core.text.density`, so the planner, the request-fit
#: guard and this estimator count in one unit: a request planned against the
#: window cannot arrive over it.

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


@dataclass(frozen=True)
class _ClampOutcome:
    """What a clamp pass left behind, and what it says about the budget.

    ``messages`` is the surviving history, ``dropped`` names every oversized
    message that went (for one log line, not one line each), and ``pinned_over``
    records that the task assignment itself exceeds the trigger — the case no
    later trim can rescue.
    """

    messages: list[ModelMessage]
    """The history after the clamp."""

    dropped: list[str]
    """Role and size of each dropped message, e.g. ``request ~152215 token(s) [retry-feedback]``."""

    pinned_over: bool
    """Whether a pinned part alone exceeds the trigger."""

    def merged_with(self, prior: _ClampOutcome) -> _ClampOutcome:
        """Fold an earlier clamp pass into this one, for a single log line.

        A wrapper clamps twice — before the strategy runs, so an un-fittable
        part cannot skew what the strategy keeps, and after it, so the
        guarantee holds whatever the strategy returned. Both passes feed one
        line, or the log would report two partial stories.
        """
        return _ClampOutcome(
            messages=self.messages,
            dropped=[*prior.dropped, *self.dropped],
            pinned_over=self.pinned_over or prior.pinned_over,
        )


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

    tokenizer: Callable[[str], int] = field(default_factory=lambda: conservative_tokens)
    """Counts *text* for every estimate this gate compares.

    Both operands come from the same callable, so the growth this gate measures
    and the ``min_growth_tokens`` it compares against stay on one scale — the
    conservative one, not the harness's optimistic four-chars-per-token default.

    A factory, not a plain default: pyreorder orders module sections, so the
    formatter places ``conservative_tokens`` after the dataclasses. A plain
    default would evaluate the name while the class body runs, before the
    function exists, and the whole package would stop importing.
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
        trigger = _tiered_target(self.strategy, ctx.model)
        cap = _per_message_cap(_strategy_window(self.strategy, ctx.model))
        before = _clamp_oversized(pinned, trigger, self.tokenizer, cap)
        compacted = await self.strategy.compact(before.messages, ctx)
        outcome = _clamp_oversized(compacted, trigger, self.tokenizer, cap).merged_with(before)
        outcome = _escalate(outcome, trigger, self.tokenizer)
        _warn_if_still_over(outcome, trigger, self.tokenizer, label="Context compaction")
        self._baseline_tokens = estimate_context_tokens(outcome.messages, self.tokenizer)
        return outcome.messages

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
        # The request's model, not the run's: the wrapped tiered stack resolves its
        # own escalation target from the request context's model, so a capability
        # that swaps in a smaller-window model must shrink the clamp budget too.
        trigger = _tiered_target(self.strategy, request_context.model)
        cap = _per_message_cap(_strategy_window(self.strategy, request_context.model))
        before = _clamp_oversized(pinned, trigger, self.tokenizer, cap)
        if before.dropped:
            ctx.messages[:] = before.messages
        request_context = await self.strategy.before_model_request(ctx, request_context)
        outcome = _clamp_oversized(list(ctx.messages), trigger, self.tokenizer, cap).merged_with(before)
        outcome = _escalate(outcome, trigger, self.tokenizer)
        if outcome.dropped:
            ctx.messages[:] = outcome.messages
            request_context = replace(request_context, messages=outcome.messages)
        _warn_if_still_over(outcome, trigger, self.tokenizer, label="Context compaction")
        self._baseline_tokens = estimate_context_tokens(outcome.messages, self.tokenizer)
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
class _UsableTargetCompaction(TieredCompaction[AgentDepsT]):
    """``TieredCompaction`` whose target is a fraction of the *usable* window.

    The harness resolves ``target_fraction`` against the whole window, so a
    history at its target already asks for more than the window holds: the
    request also carries the answer reserve and the safety margin. This
    subclass resolves the same fraction against the usable window instead
    (window − answer reserve − safety margin), so the target holds at every
    window size — the formula, not the 64 000-token case, is the contract.
    """

    def _target(self, model: AbstractModel | str) -> int | None:
        """The absolute token target for *model*, or ``None`` for a realtime one."""
        fraction = self.target_fraction
        if fraction is None:
            return super()._target(model)
        window = self.context_window or resolve_context_window(model) or self.fallback_context_window
        return max(1, int(usable_window_tokens(window) * fraction))


@dataclass
class _UsableRelevanceCompaction(RelevanceCompaction[AgentDepsT]):
    """``RelevanceCompaction`` whose target is a fraction of the *usable* window.

    The relevance tier judges the history down to its own target; that target
    has to leave room for the answer and the safety margin exactly like the
    tiered stack's, or the judged history is still too big for the request.
    """

    def _target(self, model: AbstractModel | str) -> int | None:
        """The size to compact down to for *model*, or ``None`` for a realtime one."""
        window = self.context_window or resolve_context_window(model) or self.fallback_context_window
        if self.target_fraction is not None:
            return max(1, int(usable_window_tokens(window) * self.target_fraction))
        if self.max_fraction is not None:
            return max(1, int(usable_window_tokens(window) * self.max_fraction) // 2)
        return super()._target(model)


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
    * **A clamp.** A retained message that alone exceeds the trigger is
      dropped: it is an echo of a response the provider already counted, and
      trimming around it can never bring the request under budget. The p863
      tail held one 380 538-char retry-feedback part, so dropping every older
      message left the request at 90 154 actual tokens. The pinned task
      assignment and either half of a tool call/return pair are never dropped.
    * **A record.** Every trim logs kept/dropped messages and the token
      estimate before/after; a trim that reclaims nothing says so, and one
      line names both what the clamp dropped and the largest survivor, so a
      run log explains the ranking the provider saw. A tail that cannot
      shrink otherwise logs both lines on every request, so each says it
      once: the no-op warning once per streak (an effective trim re-arms
      it), the still-over line once per distinct (kept, largest) tail. Both lines hold one
      situation for many requests when the tail cannot shrink, so the
      no-op warning fires once per streak and the still-over line once per
      distinct ``(kept, largest)`` pair; repeats drop to debug.

    Every count this class compares comes from ``conservative_tokens``, so
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

    _noop_streak: int = field(default=0, repr=False, compare=False)
    """Consecutive no-op trims; only the first of a streak warns."""

    _still_over_seen: set[tuple[int, int]] = field(default_factory=set, repr=False, compare=False)
    """(kept messages, largest survivor) tails already reported at warning level."""

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Trim, clamp the tail, then record what was kept, what was dropped, and the cost."""
        trigger = self._trigger_tokens(ctx.model)
        cap = _per_message_cap(self._window(ctx.model))
        before = _clamp_oversized(messages, trigger, self.tokenizer, cap)
        # The trim is measured against the post-clamp list: a clamp drop charged
        # to the trim would misname the mechanism, and a clamp-only pass — where
        # the trim itself reclaimed nothing — must still hit the warning below.
        trim_from = len(before.messages)
        before_tokens = estimate_context_tokens(before.messages, self.tokenizer)
        compacted = await super().compact(before.messages, ctx)
        outcome = _clamp_oversized(compacted, trigger, self.tokenizer, cap).merged_with(before)
        compacted = outcome.messages
        after_tokens = estimate_context_tokens(compacted, self.tokenizer)
        dropped = trim_from - len(compacted)
        if dropped > 0:
            # Only an effective trim moves the baseline: a no-op trim rewrote
            # nothing, so the next request must be free to try again.
            self._baseline_tokens = after_tokens
            self._noop_streak = 0
            logger.info(
                "Sliding-window trim: dropped %d of %d message(s), kept %d; ~%d → ~%d token(s)",
                dropped,
                trim_from,
                len(compacted),
                before_tokens,
                after_tokens,
            )
        else:
            # The situation is the same until an effective trim lands, so the
            # warning fires once per streak; a repeat is the same fact and
            # drops to debug, keeping the run log readable.
            self._noop_streak += 1
            repeat = f" (repeat {self._noop_streak})" if self._noop_streak > 1 else ""
            log = logger.warning if self._noop_streak == 1 else logger.debug
            log(
                "Sliding-window trim reclaimed nothing%s: %d message(s); an estimated ~%d token(s) survive the %s-token tail",
                repeat,
                len(compacted),
                after_tokens,
                self.keep_tokens,
            )
        _warn_if_still_over(outcome, trigger, self.tokenizer, label="Sliding-window trim", seen=self._still_over_seen)
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
        trigger = max(1, int(usable_window_tokens(window) * fraction))
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
        # The trim is driven here, not by the harness: the harness resolves its
        # own fraction against the whole window, while the budget this trim
        # protects is the usable window (window − answer reserve − safety).
        request_ctx = replace(ctx, model=request_context.model)
        compacted = await self.compact(list(ctx.messages), request_ctx)
        ctx.messages[:] = compacted
        return replace(request_context, messages=compacted)

    def _trigger_tokens(self, model: AbstractModel | str) -> int | None:
        """The token budget this trim compares against, ``None`` when it has none."""
        return _fraction_trigger(
            fraction=self.max_fraction,
            context_window=self.context_window,
            fallback_context_window=self.fallback_context_window,
            model=model,
        )

    def _window(self, model: AbstractModel | str) -> int:
        """The window this trim's fractions are resolved against."""
        return self.context_window or resolve_context_window(model) or self.fallback_context_window


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
        trigger = _fraction_trigger(
            fraction=self.max_fraction,
            context_window=self.context_window,
            fallback_context_window=self.fallback_context_window,
            model=ctx.model,
        )
        cap = _per_message_cap(self.context_window or resolve_context_window(ctx.model) or self.fallback_context_window)
        before = _clamp_oversized(messages, trigger, self.tokenizer, cap)
        compacted = await super().compact(before.messages, ctx)
        outcome = _clamp_oversized(compacted, trigger, self.tokenizer, cap).merged_with(before)
        logger.warning("Sliding-window fallback kept %d of %d message(s)", len(outcome.messages), len(messages))
        _warn_if_still_over(outcome, trigger, self.tokenizer, label="Sliding-window fallback")
        return outcome.messages


class _PinningFallbackCompaction(FallbackCompaction):
    """``FallbackCompaction`` that pins the task assignment before the chain runs.

    The harness re-injects pinned parts from the *original* history after a
    chain entry succeeds (``_compact_pinned``), so the pin has to be on
    ``ctx.messages`` before the chain is entered — pinning inside a chain
    entry is invisible to that re-injection and the sliding fallback would
    drop the task assignment along with the trimmed history.

    The gate is kompend's own as well: the harness resolves its trigger
    against the whole window, while a request also carries the answer reserve
    and the safety margin. The chain therefore runs once the history passes
    ``(window − answer reserve − safety margin) x fraction`` — the budget the
    request actually has.
    """

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        pinned = _pin_first_user_turn(list(ctx.messages))
        if pinned is not ctx.messages:
            ctx.messages[:] = pinned
        trigger = _fraction_trigger(
            fraction=self.max_fraction,
            context_window=self.context_window,
            fallback_context_window=self.fallback_context_window,
            model=request_context.model,
        )
        if trigger is None:
            return request_context
        messages = list(ctx.messages)
        tokens = estimate_context_tokens(
            messages,
            self.tokenizer,
            model_request_parameters=request_context.model_request_parameters,
        )
        if tokens <= trigger:
            return request_context
        request_ctx = replace(ctx, model=request_context.model)
        compacted = await self._run_chain(messages, request_ctx)
        ctx.messages[:] = compacted
        return replace(request_context, messages=compacted)

    async def _run_chain(self, messages: list[ModelMessage], ctx: RunContext[Any]) -> list[ModelMessage]:
        """Run the fallback chain and re-inject the pinned parts.

        ``_compact_pinned`` is the harness's own entry point for this: it runs
        the chain (a summary that does not shrink the history falls through to
        the next entry) and puts the pinned parts back. A renamed private API
        must fail loudly rather than silently skip the compaction.
        """
        compact_pinned = getattr(self, "_compact_pinned", None)
        if compact_pinned is None:
            raise AttributeError(
                f"{type(self).__name__} has no `_compact_pinned(messages, ctx, held)` method, so the "
                "fallback chain cannot run; the harness FallbackCompaction private API changed; "
                "update `_PinningFallbackCompaction` to match"
            )
        return await compact_pinned(messages, ctx, None)


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
        Fraction of the *usable* window at which compaction runs (the window
        minus the answer reserve and the safety margin). The single
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
            tokenizer=conservative_tokens,
        ),
        DeduplicateFileReads(
            file_key=_file_key,
            max_fraction=compress_threshold,
            context_window=context_window,
            tokenizer=conservative_tokens,
        ),
    ]
    if judge is not None:
        tiers.append(
            _UsableRelevanceCompaction(
                model=judge,
                summarizer=summarizer,
                max_fraction=compress_threshold,
                target_fraction=compress_threshold,
                context_window=context_window,
                keep_tokens=keep_tokens if keep_tokens is not None else _DEFAULT_KEEP_TOKENS,
                tokenizer=conservative_tokens,
            ),
        )
    tiers.append(summarizer)
    gated = GrowthGatedCompaction(
        strategy=_UsableTargetCompaction(
            tiers=tiers,
            target_fraction=compress_threshold,
            context_window=context_window,
            tokenizer=conservative_tokens,
        ),
        min_growth_tokens=min_growth_tokens if min_growth_tokens is not None else _DEFAULT_MIN_GROWTH_TOKENS,
        tokenizer=conservative_tokens,
    )
    window = context_window or DEFAULT_CONTEXT_WINDOW
    return _PinningFallbackCompaction(
        fallback_chain=[
            gated,
            _LoggedSlidingWindowCompaction(
                max_fraction=compress_threshold,
                context_window=context_window,
                keep_tokens=int(usable_window_tokens(window) * compress_threshold * _FALLBACK_KEEP_SHARE),
                tokenizer=conservative_tokens,
            ),
        ],
        fallback_on=(UnexpectedModelBehavior,),
        max_fraction=compress_threshold,
        context_window=context_window,
        tokenizer=conservative_tokens,
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
    figure it reports is counted with ``conservative_tokens``, kompend's
    measured density floor, never the harness's four-chars-per-token default.

    Parameters
    ----------
    context_window:
        The model's input window; the trigger is a fraction of it.
    compress_threshold:
        Fraction of the *usable* window at which the trim runs. Well below 1.0: it
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
    headroom = int(usable_window_tokens(window) * compress_threshold * _FALLBACK_KEEP_SHARE)
    tail = min(keep_tokens, headroom) if keep_tokens else 0
    return _GuardedSlidingCompaction(
        max_messages=_SLIDING_MAX_MESSAGES,
        max_fraction=compress_threshold,
        context_window=context_window,
        keep_messages=keep_messages,
        keep_tokens=tail or None,
        min_growth_tokens=min_growth_tokens if min_growth_tokens is not None else _DEFAULT_MIN_GROWTH_TOKENS,
        tokenizer=conservative_tokens,
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
        Fraction of the *usable* window at which compaction runs
        (``max_fraction``).
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
            tokenizer=conservative_tokens,
        )
    return _LoggedSummarizingCompaction(
        model=model,
        max_fraction=compress_threshold,
        keep_messages=keep_messages,
        context_window=context_window,
        instructions=instructions,
        tokenizer=conservative_tokens,
    )


def _fraction_trigger(
    *,
    fraction: float | None,
    context_window: int | None,
    fallback_context_window: int,
    model: AbstractModel | str,
) -> int | None:
    """Resolve a fractional trigger to tokens, or ``None`` when there is none.

    The fraction is a share of the *usable* window — the window minus the
    answer reserve and the safety margin — because a request also carries
    those two: a history at the whole window already asks for more than the
    window holds. Mirrors the harness's own resolution so kompend's clamp
    measures against exactly the budget the strategy measured against.
    """
    if fraction is None:
        return None
    window = context_window or resolve_context_window(model) or fallback_context_window
    return max(1, int(usable_window_tokens(window) * fraction))


def _tiered_target(strategy: Any, model: AbstractModel | str) -> int | None:
    """The token target a ``TieredCompaction`` escalates toward.

    Read through ``_target`` — the harness method the tiered stack itself
    calls — so the clamp cannot disagree with the escalation loop about the
    budget, and so a stack configured with an absolute target works too.
    ``None`` only ever means "realtime model, nothing to clamp against"; a
    strategy without the method is a changed harness contract, and raising
    beats silently disabling the clamp and the still-over warning.
    """
    target = getattr(strategy, "_target", None)
    if target is None:
        raise AttributeError(
            f"{type(strategy).__name__} has no `_target(model)` method, so the clamp cannot resolve its budget; "
            "the harness TieredCompaction private API changed; update `_tiered_target` to match"
        )
    resolved: int | None = target(model)
    return resolved


def _clamp_oversized(
    messages: list[ModelMessage],
    trigger: int | None,
    tokenizer: Callable[[str], int] | None,
    per_message_cap: int | None = None,
) -> _ClampOutcome:
    """Drop retained messages that alone exceed *trigger*; report what went.

    A message the provider already counted can be larger than the budget for
    the whole request. No amount of trimming around it makes the request fit,
    and re-sending it can only end in a provider 400 after minutes of upstream
    work — the p863 run sent one 380 538-char retry-feedback part inside a
    65 536-token window and was refused. Such a message is an echo of a
    response whose cost is already in that response's usage, so it is dropped
    rather than split: a truncated echo would still be sent as prompt text.

    *per_message_cap* is the second, tighter bound: the largest a single
    message may be before it is dropped whatever the trigger says. The
    compaction tiers drop or summarise whole messages, so a message above the
    cap can never be folded into the request — it is dropped at the first
    pass that sees it, which is where it would otherwise enter the history.

    Two things are never dropped, because a request without them answers the
    wrong question:

    * pinned messages — the task assignment first, then any later part the
      pin contract marks: every strategy re-injects each one, so the clamp
      may not drop them either. If one alone is over budget the request
      cannot be saved, and the caller says so loudly.
    * either half of a tool call/return pair — a provider rejects a return
      whose call is gone, so the pair outlives the budget instead.
    """
    if trigger is None or not messages:
        return _ClampOutcome(messages, [], False)
    counts = estimate_message_tokens(messages, tokenizer)
    if len(counts) != len(messages):  # pragma: no cover - the harness pairs them
        return _ClampOutcome(messages, [], False)
    limit = trigger if per_message_cap is None else min(trigger, per_message_cap)
    oversized = [index for index, count in enumerate(counts) if count > limit]
    if not oversized:
        return _ClampOutcome(messages, [], False)
    pinned = _pinned_indexes(messages)
    paired = _paired_indexes(messages)
    dropped: list[str] = []
    survivors: list[ModelMessage] = []
    for index, message in enumerate(messages):
        if index not in oversized or index in pinned or index in paired:
            survivors.append(message)
            continue
        dropped.append(_describe(message, counts[index]))
    return _ClampOutcome(survivors, dropped, bool(pinned.intersection(oversized)))


def _escalate(
    outcome: _ClampOutcome,
    target: int | None,
    tokenizer: Callable[[str], int] | None,
) -> _ClampOutcome:
    """Drop the turns the tiers could not shrink, and name every drop.

    The last resort of the compaction stack: the tiers drop or summarise
    whole messages, so a history that still exceeds its target after every
    tier ran needs a message to go. The drop is logged by name — role, size
    and part kinds — so a run log says which turn the escalation gave up.
    """
    escalated = _drop_largest_to_target(outcome.messages, target, tokenizer)
    if not escalated.dropped:
        return outcome
    logger.warning(
        "Compaction escalation: dropped %d message(s) the tiers could not shrink (%s)",
        len(escalated.dropped),
        "; ".join(escalated.dropped),
    )
    return escalated.merged_with(outcome)


def _drop_largest_to_target(
    messages: list[ModelMessage],
    target: int | None,
    tokenizer: Callable[[str], int] | None,
) -> _ClampOutcome:
    """Drop the largest droppable turns until the history fits *target*.

    The escalation the compaction tiers cannot do: they drop or summarise
    whole messages, so one message the summarizer cannot shrink leaves the
    history above its target no matter how many tiers ran. The p863 run
    reached that state — every tier ran and the history stayed at an
    estimated 101 849 tokens against a 57 600-token target, because one
    message was 50 502 tokens.

    A *turn* is a request and the response that answered it, so dropping one
    keeps the provider's alternation intact. The pinned task assignment, both
    halves of a tool call/return pair and the request being answered now are
    never dropped: a history without them answers the wrong question.
    """
    if target is None or not messages:
        return _ClampOutcome(list(messages), [], False)
    survivors = list(messages)
    protected = _protected_message_ids(survivors)
    dropped: list[str] = []
    while survivors and estimate_context_tokens(survivors, tokenizer) > target:
        counts = estimate_message_tokens(survivors, tokenizer)
        turn = _largest_droppable_turn(survivors, counts, protected)
        if turn is None:
            break
        for index in turn:
            dropped.append(_describe(survivors[index], counts[index]))
        removed = set(turn)
        survivors = [message for index, message in enumerate(survivors) if index not in removed]
    return _ClampOutcome(survivors, dropped, False)


def _protected_message_ids(messages: Sequence[ModelMessage]) -> set[int]:
    """Identity of every message the compaction may not drop.

    The pinned parts (the task assignment first), both halves of every tool
    call/return pair, and the request being answered now — the last message
    is the one the run is about to send, and a history without it would
    answer a different question.
    """
    protected = _pinned_indexes(messages) | _paired_indexes(messages)
    if messages:
        protected.add(len(messages) - 1)
    return {id(messages[index]) for index in protected}


def _largest_droppable_turn(
    messages: Sequence[ModelMessage],
    counts: Sequence[int],
    protected: set[int],
) -> list[int] | None:
    """Indexes of the largest turn that may be dropped, or ``None``.

    A turn is a request together with the response that answered it, so the
    provider's request/response alternation survives the drop. A turn holding
    a protected message is not droppable at all.
    """
    turns = _turns(messages)
    best: list[int] | None = None
    best_size = 0
    for turn in turns:
        if any(id(messages[index]) in protected for index in turn):
            continue
        size = sum(counts[index] for index in turn)
        if size > best_size:
            best, best_size = turn, size
    return best


def _turns(messages: Sequence[ModelMessage]) -> list[list[int]]:
    """Group the messages into turns: a request and the response answering it."""
    turns: list[list[int]] = []
    current: list[int] = []
    for index, message in enumerate(messages):
        if isinstance(message, ModelRequest) and current:
            turns.append(current)
            current = []
        current.append(index)
        if isinstance(message, ModelResponse) and current:
            turns.append(current)
            current = []
    if current:
        turns.append(current)
    return turns


def _per_message_cap(window: int) -> int:
    """The largest a single message may be before it leaves the history.

    Twice the answer reserve — the window's own history budget. It scales
    with the discovered window, so a 1M-context model allows a proportionally
    larger message than a 64k one; the number is never fixed.
    """
    return 2 * answer_reserve_tokens(window)


def _strategy_window(strategy: Any, model: AbstractModel | str) -> int:
    """The window a strategy resolves its fractions against."""
    window = getattr(strategy, "context_window", None)
    if window:
        return int(window)
    resolved = resolve_context_window(model)
    if resolved:
        return int(resolved)
    return int(getattr(strategy, "fallback_context_window", DEFAULT_CONTEXT_WINDOW))


def _pinned_indexes(messages: Sequence[ModelMessage]) -> set[int]:
    """Indexes of every message holding a pinned part, not just the first.

    The harness pin contract (``reinject_pinned``) protects each pinned
    part, so a clamp that spared only the first would drop a later pin the
    strategies would faithfully re-inject.
    """
    return {index for index, message in enumerate(messages) if isinstance(message, ModelRequest) and any(is_pinned(part) for part in message.parts)}


def _paired_indexes(messages: Sequence[ModelMessage]) -> set[int]:
    """Indexes of messages on either side of a tool call/return pair.

    A return whose call was dropped is rejected by the provider, so the pair
    is kept whole rather than half-dropped. Call ids are only unique within a
    run, and a provider may reuse one, so a return is matched to the nearest
    preceding call.
    """
    calls: dict[str, int] = {}
    pairs: set[int] = set()
    for index, message in enumerate(messages):
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart) and part.tool_call_id:
                    calls[part.tool_call_id] = index
            continue
        for part in message.parts:
            tool_call_id = getattr(part, "tool_call_id", None)
            call_index = calls.get(tool_call_id) if isinstance(tool_call_id, str) else None
            if call_index is not None:
                pairs.update((call_index, index))
    return pairs


def _describe(message: ModelMessage, tokens: int) -> str:
    """Name a dropped message by role and size, for the trim's log line."""
    role = "response" if isinstance(message, ModelResponse) else "request"
    kinds = ",".join(sorted({part.part_kind for part in message.parts}))
    return f"{role} ~{tokens} token(s) [{kinds}]"


def _warn_if_still_over(
    outcome: _ClampOutcome,
    trigger: int | None,
    tokenizer: Callable[[str], int] | None,
    *,
    label: str,
    seen: set[tuple[int, int]] | None = None,
) -> None:
    """One line naming what the clamp dropped and the largest survivor.

    Two facts belong together because either alone misleads: a trim that
    dropped the oversized part and now fits is a success, while one that kept
    it is a request about to be refused, and only the pair says which happened.
    The largest survivor is also the number to look at when the request is
    still over budget — the tail is what it is — so the line keeps the wording
    an operator greps for and takes the severity that matches the outcome.

    A pinned part that alone exceeds the trigger is the one case no
    later trim can rescue: it is kept, and the warning says the request
    cannot be saved rather than implying another pass would help.

    A request whose tail cannot shrink logs this line every turn, so
    *seen* dedupes it: a tail described by the same kept count and largest
    survivor reports once at warning level and exact repeats at debug, and
    only a changed tail warns again.
    """
    if trigger is None:
        return
    counts = estimate_message_tokens(outcome.messages, tokenizer)
    largest = max(counts, default=0)
    estimated = estimate_context_tokens(outcome.messages, tokenizer)
    over = estimated > trigger
    if not outcome.dropped and not over:
        return
    clamp = ""
    if outcome.dropped:
        clamp = f"dropped the oversized part(s) no trim could fit ({'; '.join(outcome.dropped)}); "
    if outcome.pinned_over:
        verdict = "a pinned part alone exceeds it, so this request cannot be saved"
    elif over:
        verdict = "the retained tail cannot reclaim more"
    else:
        verdict = "it fits now"
    emit = logger.warning if (over or outcome.pinned_over) else logger.info
    if seen is not None:
        signature = (len(outcome.messages), largest)
        if signature in seen:
            emit = logger.debug
        else:
            seen.add(signature)
    emit(
        "%s %skept %d message(s) at ~%d token(s) against an estimated trigger of %d token(s) (largest ~%d token(s)); %s",
        label,
        clamp,
        len(outcome.messages),
        estimated,
        trigger,
        largest,
        verdict,
    )
