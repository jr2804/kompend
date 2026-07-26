"""Persistent storage for separated structured assets (tables, figures).

Mirrors the :class:`~knox.core.pipeline.references.ReferenceStore` pattern:
a JSON file (``assets.json``) keyed by asset id, where each entry is an
:class:`~knox.core.models.assets.AssetRecord`.

The store is loaded once at pipeline start and saved after asset ids are
assigned. Content-addressed ids mean identical tables/figures across sources
dedup to a single record; ``ref_count`` is incremented for each referencing
node, so the store is cumulative across compilations.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from knox.core.models.assets import AssetRecord
from knox.logging import get_logger

logger = get_logger(__name__)

__all__ = ["AssetStore"]


class AssetStore:
    """JSON-backed store of separated structured assets, keyed by id.

    File shape::

        {
          "tab-a1f3": {"id": "tab-a1f3", "kind": "table", ...},
          "fig-9c2e": {"id": "fig-9c2e", "kind": "figure", ...}
        }

    Parameters
    ----------
    assets_file:
        File to read/write. Created lazily on first save.
    """

    def __init__(self, assets_file: Path) -> None:
        self._file = assets_file
        self._data: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Read the assets file into memory.

        A missing file means a fresh start (empty store). A corrupt file
        is logged as a warning and treated as empty rather than crashing
        the pipeline.
        """
        if not self._file.exists():
            self._data = {}
            return
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Asset store corrupt (%s) — starting fresh: %s", exc, self._file)
            self._data = {}
            return
        if not isinstance(raw, dict):
            logger.warning("Asset store not a dict (%r) — starting fresh", type(raw).__name__)
            self._data = {}
            return
        self._data = raw

    def save(self) -> None:
        """Write the store to disk.

        Creates the parent directory lazily. A failed write is logged but
        does not crash the pipeline (asset separation is best-effort).
        """
        self._file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._file.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover — filesystem-dependent
            logger.warning("Could not write asset store (%s): %s", self._file, exc)

    def get(self, asset_id: str) -> AssetRecord | None:
        """Return the record for ``asset_id``, or ``None`` if absent."""
        raw = self._data.get(asset_id)
        return AssetRecord.from_dict(raw) if raw is not None else None

    def put(self, record: AssetRecord) -> None:
        """Insert or replace a record (keyed by ``record.id``)."""
        self._data[record.id] = record.to_dict()

    def increment_ref(self, asset_id: str) -> None:
        """Increment the ``ref_count`` of ``asset_id`` (no-op if absent)."""
        record = self.get(asset_id)
        if record is not None:
            record.ref_count += 1
            self.put(record)

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[AssetRecord]:
        for raw in self._data.values():
            yield AssetRecord.from_dict(raw)

    def total_count(self) -> int:
        """Total number of distinct asset records."""
        return len(self._data)
