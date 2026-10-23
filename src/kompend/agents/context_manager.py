"""Context compaction for kompend agents, on ``pydantic_ai_harness.compaction``.

The functionality this module used started in the community package
``summarization-pydantic-ai``; it was upstreamed into pydantic-ai-harness
(``pydantic_ai_harness.compaction``), which kompend now uses directly:
``SummarizingCompaction`` for the LLM path, ``SlidingWindowCompaction`` for the
zero-cost path and ``ReportContextUsage`` for the token accounting.

Kompend adds one guarantee on top: **compaction must always happen**. The
summarizer runs on the agent's own model, and a model that hallucinates a tool
call (MiniMax mirrors ``final_result`` from the embedded history) makes the
summary fail; the harness then leaves the history untouched and the next
request overflows the model window. ``_HardenedSummarizingCompaction`` falls
back to a deterministic extractive summary when the model cannot answer.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.compaction import SummarizingCompaction

from kompend.logging import get_logger

#: Characters kept by the extractive fallback summary.
_FALLBACK_SUMMARY_CHARS: int = 4000

logger = get_logger(__name__)


class _HardenedSummarizingCompaction(SummarizingCompaction):
    """``SummarizingCompaction`` that never leaves the history uncompacted."""

    async def _summarize(self, messages: list[ModelMessage], ctx: Any, previous_summary: str | None = None) -> str:
        logger.info("Context compression starting: %d message(s)", len(messages))
        try:
            summary = await super()._summarize(messages, ctx, previous_summary=previous_summary)
        except Exception as exc:  # ponytail: any summarizer failure must not disable compaction
            logger.warning(
                "Summarizer model failed (%s) - compacting with an extractive summary of %d message(s)",
                exc,
                len(messages),
            )
            return _extractive_summary(messages)
        logger.info("Context compression produced %d-char summary", len(summary or ""))
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
    return _HardenedSummarizingCompaction(
        model=model,
        max_fraction=compress_threshold,
        keep_messages=keep_messages,
        context_window=context_window,
    )


def _extractive_summary(messages: list[ModelMessage]) -> str:
    """Keep the tail of the conversation as plain text (model-free compaction)."""
    chunks: list[str] = []
    for message in messages:
        for part in getattr(message, "parts", ()):
            content = getattr(part, "content", None)
            if isinstance(content, str) and content.strip():
                chunks.append(content.strip())
    rendered = "\n".join(chunks).strip()
    if not rendered:
        return "No previous conversation history."
    if len(rendered) <= _FALLBACK_SUMMARY_CHARS:
        return rendered
    return "… " + rendered[-_FALLBACK_SUMMARY_CHARS:]
