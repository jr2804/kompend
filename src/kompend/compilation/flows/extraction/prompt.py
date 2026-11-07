"""Extraction context assembly — threaded message history for the incremental path.

History estimation, in-place compaction, and the synthetic response that stands in
for a cached batch so the next uncached batch keeps the document thread.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

from kompend.compilation.two_phase import (
    Extraction,
)
from kompend.llm.request_budget import chars_to_tokens
from kompend.logging import LOG_SAMPLE_LIMIT, get_logger

logger = get_logger(__name__)


def _compact_history_inplace(history: list[ModelMessage], max_tokens: int, *, chars_per_token: float) -> None:
    """Drop the oldest messages from ``history`` until under budget.

    Mutates the list in place. Messages come in (request, response) pairs
    in pydantic-ai, so dropping the two oldest preserves pairing. Keeps
    at least the last (request, response) pair so the agent always has
    some context to continue from.
    """
    if not history:
        return
    while _estimate_history_tokens(history, chars_per_token=chars_per_token) > max_tokens and len(history) > 2:
        # Drop the oldest request or response. Remove two at a time when
        # possible to keep request/response pairing intact.
        drop_n = 2 if len(history) > 2 else 1
        del history[:drop_n]
    if _estimate_history_tokens(history, chars_per_token=chars_per_token) > max_tokens and len(history) >= 2:
        logger.debug(
            "TwoPhase: history still over budget after compaction (%d tokens, %d messages) — keeping tail",
            _estimate_history_tokens(history, chars_per_token=chars_per_token),
            len(history),
        )


def _estimate_history_tokens(history: list[ModelMessage], *, chars_per_token: float) -> int:
    """Estimate total tokens across all messages in ``history``.

    Walks every ``ModelRequest`` / ``ModelResponse`` and sums the string
    length of each part's ``content`` field. Non-string parts (images,
    tool calls) are ignored — they're rare in the extraction path.

    The conversion uses the configured chars-per-token
    (``compile.llm.default_chars_per_token``), the same heuristic the batch
    budget is built from: measuring the history with a different ratio than
    the one the window was sized with is how a compacted history still
    overflows.
    """
    total = 0
    for msg in history:
        parts = getattr(msg, "parts", [])
        for part in parts:
            content = getattr(part, "content", None)
            if isinstance(content, str):
                total += chars_to_tokens(len(content), chars_per_token=chars_per_token)
    return total


def _synthetic_history_response(extraction: Extraction) -> ModelResponse:
    """Build a synthetic ``ModelResponse`` summarising a cached extraction.

    When a batch is served from the per-batch cache, we still want the
    next uncached batch to see what concepts/entities were extracted
    upstream. Rather than replaying the original (potentially huge)
    assistant turn, we inject a compact one-line summary.
    """
    concepts = ", ".join(c.name or "?" for c in extraction.concepts[:LOG_SAMPLE_LIMIT])
    entities = ", ".join(e.name or "?" for e in extraction.entities[:LOG_SAMPLE_LIMIT])
    summary_bits = [f"summary: {extraction.summary}"]
    if concepts:
        summary_bits.append(f"concepts: {concepts}")
    if entities:
        summary_bits.append(f"entities: {entities}")
    return ModelResponse(parts=[TextPart(content=" | ".join(summary_bits))])
