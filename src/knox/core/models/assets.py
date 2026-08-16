"""Separated structured-asset records (tables, figures).

Phase 1 of the agentic-extraction series: tables and figures are extracted
from the chunk tree, stored under content-addressed ids, and referenced by
id + summary in the extraction input (instead of being inlined as
pipe-delimited text). See ``.agents/plans/structured-asset-separation.md``.

Equations are deferred (Phase 1 leaves them inlined via
:func:`knox.core.models.structured.structured_to_text`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from knox.core.models.enums import AssetField, AssetKind
from knox.core.models.structured import StructuredData, structured_from_dict, structured_to_dict

__all__ = ["AssetRecord"]


@dataclass
class AssetRecord:
    """A separated structured asset (table/figure) stored by id.

    The id is content-addressed (``<prefix>-<hash>``), so identical
    tables in different sources dedup to the same id. ``ref_count`` tracks
    how many nodes reference it (for prune/prune-floor heuristics later).
    """

    id: str
    kind: AssetKind
    source: str
    data: StructuredData
    summary: str = ""
    ref_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            AssetField.ID: self.id,
            AssetField.KIND: self.kind.value,
            AssetField.SOURCE: self.source,
            AssetField.DATA: structured_to_dict(self.data),
            AssetField.SUMMARY: self.summary,
            AssetField.REF_COUNT: self.ref_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AssetRecord:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            id=d[AssetField.ID],
            kind=AssetKind(d[AssetField.KIND]),
            source=d.get(AssetField.SOURCE, ""),
            data=structured_from_dict(d[AssetField.DATA]),
            summary=d.get(AssetField.SUMMARY, ""),
            ref_count=d.get(AssetField.REF_COUNT, 0),
        )
