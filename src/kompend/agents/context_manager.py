"""Context compaction for kompend agents, on ``pydantic_ai_harness.compaction``.

The functionality this module used started in the community package
``summarization-pydantic-ai``; it was upstreamed into pydantic-ai-harness
(``pydantic_ai_harness.compaction``), which kompend now uses directly:
``SummarizingCompaction`` for the LLM path, ``SlidingWindowCompaction`` for
the zero-cost path and ``ReportContextUsage`` for the token accounting.

Kompend adds logging and one fallback. The summarizer runs in a nested,
toolless agent that never gets an explicit retry budget (harness 0.64,
``_summarizing_compaction.py``), so it inherits ``ToolManager``'s default of
one retry; since it is fed a transcript of the run's own tool calls, a small
model imitates one, the single unknown-tool refusal is consumed, and
``UnexpectedModelBehavior`` escapes ``before_model_request`` and kills the
parent run (pydantic/pydantic-ai#12307). ``FallbackCompaction`` keeps that
from being fatal: when the summarizer cannot answer, the oldest messages are
dropped instead and the run continues.

That fallback is not the hand-crafted stand-in ``.agents/CODE.md`` forbids
under "LLM calls": nothing is fabricated and no prompt, source or model
answer is cropped. It is the harness's own zero-cost strategy — the one
kompend already exposes as ``compile.llm.context_manager = "sliding"`` —
applied to whole messages, exactly as the summarizer would have. It also
stays narrow: only ``UnexpectedModelBehavior`` (retry exhaustion, unusable
model behaviour) falls through. Provider errors and programming errors
propagate unchanged, so an outage still burns loudly instead of degrading
the context silently.
"""

from __future__ import annotations

from typing import Any, Final

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.compaction import (
    DEFAULT_CONTEXT_WINDOW,
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
)

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

logger = get_logger(__name__)


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


def build_fallback_compaction(
    *,
    context_window: int | None,
    compress_threshold: float,
    keep_messages: int,
    model: Any = None,
) -> FallbackCompaction:
    """Build the compaction capability kompend agents actually attach.

    Summarizes at *compress_threshold* of the window; if the summarizer
    cannot answer (see the module docstring), falls back to the zero-cost
    sliding window trimmed on the same trigger, keeping a tail worth
    ``_FALLBACK_KEEP_SHARE`` of it.

    Parameters
    ----------
    context_window:
        The model's input window; the trigger is a fraction of it.
    compress_threshold:
        Fraction of the window at which compaction runs (``max_fraction``).
    keep_messages:
        How many recent messages survive a *successful* summary verbatim.
    model:
        The agent's model, reused for the summary so both run on the same
        failover chain.
    """
    return FallbackCompaction(
        fallback_chain=[
            build_summarizing_compaction(
                context_window=context_window,
                compress_threshold=compress_threshold,
                keep_messages=keep_messages,
                model=model,
            ),
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


def build_summarizing_compaction(
    *,
    context_window: int | None,
    compress_threshold: float,
    keep_messages: int,
    model: Any = None,
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
    """
    return _LoggedSummarizingCompaction(
        model=model,
        max_fraction=compress_threshold,
        keep_messages=keep_messages,
        context_window=context_window,
    )
