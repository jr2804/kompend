"""Deterministic one-line summaries for separated structured assets.

Used in extraction-text references (``¶ tab-a1f3 (summary: ...)``) and in
asset footnotes. No LLM call, no embeddings — pure formatting. Rich
LLM-generated summaries are Phase 3 (on-demand asset tools).
"""

from __future__ import annotations

from kompend.core.models.structured import (
    StructuredData,
    StructuredEquation,
    StructuredFigure,
    StructuredTable,
)

__all__ = ["summarize_asset"]


def summarize_asset(data: StructuredData) -> str:
    """Return a one-line deterministic summary of a structured asset.

    - Table: ``"{cols} cols × {rows} rows" + (" — {caption}" if caption)``
    - Figure: ``"{caption or alt_text or 'figure'}"``
    - Equation: not an asset in Phase 1 (caller never passes equations here);
      returns the caption or a fallback if one ever reaches this path.
    """
    if isinstance(data, StructuredTable):
        dims = f"{len(data.columns)} cols × {len(data.rows)} rows"
        if data.caption:
            return f"{dims} — {_one_line(data.caption)}"
        return dims
    if isinstance(data, StructuredFigure):
        text = data.caption or data.alt_text or "figure"
        return _one_line(text)
    if isinstance(data, StructuredEquation):
        # Deferred (Phase 1 leaves equations inlined). Handle defensively.
        return _one_line(data.caption) if data.caption else "equation"
    return ""


def _one_line(text: str) -> str:
    """Collapse line breaks so the reference stays one line. No content is cut."""
    return " ".join(text.split())
