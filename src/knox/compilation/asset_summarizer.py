"""Deterministic one-line summaries for separated structured assets.

Used in extraction-text references (``¶ tab-a1f3 (summary: ...)``) and in
asset footnotes. No LLM call, no embeddings — pure formatting. Rich
LLM-generated summaries are Phase 3 (on-demand asset tools).
"""

from __future__ import annotations

from knox.core.models.structured import (
    StructuredData,
    StructuredEquation,
    StructuredFigure,
    StructuredTable,
)

__all__ = ["ASSET_SUMMARY_MAX_CHARS", "summarize_asset"]

#: Maximum caption/alt-text length kept verbatim in a summary.
ASSET_SUMMARY_MAX_CHARS = 80


def summarize_asset(data: StructuredData) -> str:
    """Return a one-line deterministic summary of a structured asset.

    - Table: ``"{cols} cols × {rows} rows" + (" — {caption}" if caption)``
    - Figure: ``"{caption or alt_text or 'figure'}"`` (truncated)
    - Equation: not an asset in Phase 1 (caller never passes equations here);
      returns the caption or a fallback if one ever reaches this path.
    """
    if isinstance(data, StructuredTable):
        dims = f"{len(data.columns)} cols × {len(data.rows)} rows"
        if data.caption:
            return f"{dims} — {_truncate(data.caption)}"
        return dims
    if isinstance(data, StructuredFigure):
        text = data.caption or data.alt_text or "figure"
        return _truncate(text)
    if isinstance(data, StructuredEquation):
        # Deferred (Phase 1 leaves equations inlined). Handle defensively.
        return _truncate(data.caption) if data.caption else "equation"
    return ""


def _truncate(text: str) -> str:
    """Truncate to :data:`ASSET_SUMMARY_MAX_CHARS`, adding an ellipsis if cut."""
    if len(text) <= ASSET_SUMMARY_MAX_CHARS:
        return text
    return text[: ASSET_SUMMARY_MAX_CHARS - 1].rstrip() + "…"
