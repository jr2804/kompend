"""Context compaction for kompend agents, on ``pydantic_ai_harness.compaction``.

The functionality this module used started in the community package
``summarization-pydantic-ai``; it was upstreamed into pydantic-ai-harness
(``pydantic_ai_harness.compaction``), which kompend now uses directly:
``SummarizingCompaction`` for the LLM path, ``SlidingWindowCompaction`` for
the zero-cost path and ``ReportContextUsage`` for the token accounting.

Kompend adds logging only. When the summarizer model cannot answer, the
compaction error propagates: a hand-crafted fallback summary would silently
degrade the agent's context, and cropping the history to a character tail is
forbidden (`.agents/CODE.md`, "LLM calls") — in doubt, let it burn.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.compaction import SummarizingCompaction

from kompend.logging import get_logger

logger = get_logger(__name__)


class _LoggedSummarizingCompaction(SummarizingCompaction):
    """``SummarizingCompaction`` that logs its compression events."""

    # pydantic_ai_harness 0.62 hands ``_summarize`` a ``WrittenSummary``
    # record (``.text``/``.created_at``) instead of the plain ``str`` of
    # 0.61 — the return type follows whichever harness is installed.
    async def _summarize(self, messages: list[ModelMessage], ctx: Any, previous_summary: str | None = None) -> Any:
        logger.info("Context compression starting: %d message(s)", len(messages))
        summary = await super()._summarize(messages, ctx, previous_summary=previous_summary)
        text = getattr(summary, "text", summary)
        logger.info("Context compression produced %d-char summary", len(text or ""))
        return summary


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
