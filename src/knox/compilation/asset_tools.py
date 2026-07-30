"""PydanticAI Tool factory for on-demand asset inspection.

Provides ``get_table`` and ``get_figure_description`` tools that the
extraction agent may call to retrieve full structured asset data on
demand, instead of relying only on the compact ``\u00b6 <id>`` reference
and summary injected during ref-mode extraction.

Phase 3a --- deterministic lookups only (no VLM generation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import Tool

from knox.chunking.summarizer import structured_to_text
from knox.core.models.assets import AssetRecord
from knox.core.models.enums import AssetKind
from knox.core.models.structured import StructuredFigure

if TYPE_CHECKING:
    from knox.core.pipeline.assets import AssetStore


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_asset_tools(store: AssetStore) -> list[Tool]:
    """Build the list of on-demand asset-inspection tools.

    Args:
        store: The :class:`AssetStore` used to look up asset records.

    Returns:
        A list of PydanticAI ``Tool`` objects --- always includes
        ``get_table`` and ``get_figure_description``.
    """
    return [
        _make_get_table(store),
        _make_get_figure(store),
    ]


# ---------------------------------------------------------------------------
# Tool implementations (closure-captured ``store`` --- no RunContext needed)
# ---------------------------------------------------------------------------


def _make_get_table(store: AssetStore) -> Tool:
    """Build a ``get_table`` tool that returns pipe-delimited table text."""

    def get_table(asset_id: str) -> str:
        """Retrieve the full text representation of a table asset.

        Use this when you need to inspect the complete table data ---
        columns, rows, and values --- rather than just the compact
        ``\u00b6`` reference or summary. Call it whenever the summary
        suggests the table may contain relevant information.
        """
        record: AssetRecord | None = store.get(asset_id)
        if record is None:
            return f"[unknown asset: {asset_id}]"
        if record.kind != AssetKind.TABLE:
            return f"[{asset_id} is a {record.kind.value}, not a table]"
        try:
            return structured_to_text(record.data)
        except Exception as exc:
            return f"[error reconstructing table {asset_id}: {exc}]"

    return Tool(get_table)


def _make_get_figure(store: AssetStore) -> Tool:
    """Build a ``get_figure_description`` tool for figure assets."""

    def get_figure_description(asset_id: str) -> str:
        """Return the description of a figure asset.

        Use this to learn what a figure depicts --- its caption, alt
        text, and any available description. The data is deterministic
        (no model invocation). For figures stored as images (not yet
        implemented), the description may be empty.
        """
        record: AssetRecord | None = store.get(asset_id)
        if record is None:
            return f"[unknown asset: {asset_id}]"
        if record.kind != AssetKind.FIGURE:
            return f"[{asset_id} is a {record.kind.value}, not a figure]"
        if not isinstance(record.data, StructuredFigure):
            return f"[{asset_id} is a {record.kind.value}, not a figure]"
        figure = record.data
        parts = []
        if figure.caption:
            parts.append(f"Caption: {figure.caption}")
        if figure.alt_text:
            parts.append(f"Alt text: {figure.alt_text}")
        if figure.description:
            parts.append(f"Description: {figure.description}")
        return "\n".join(parts) if parts else "[figure --- no caption or description available]"

    return Tool(get_figure_description)
