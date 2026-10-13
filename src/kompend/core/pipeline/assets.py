"""Persistent storage for separated structured assets (tables, figures).

Backed by the ``assets`` table of the single ``.kompend/kompend.db`` store, keyed by
asset id, where each entry is an
:class:`~kompend.core.models.assets.AssetRecord`.

The store is loaded once at pipeline start and saved after asset ids are
assigned. Content-addressed ids mean identical tables/figures across sources
dedup to a single record; ``ref_count`` is incremented for each referencing
node, so the store is cumulative across compilations.
"""

# ruff: noqa: S608 — SQL is built from Final[str] table-name constants only.

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from kompend.core.models.assets import AssetField, AssetRecord
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import ASSETS_TABLE
from kompend.logging import get_logger
from kompend.types import JSONDict

__all__ = ["AssetStore"]

logger = get_logger(__name__)


class AssetStore:
    """Store of separated structured assets, keyed by id.

    Row shape (``assets`` table): ``id``, ``kind``, ``source``, ``data`` (JSON),
    ``summary``, ``ref_count``.

    Parameters
    ----------
    assets_file:
        Legacy artifact path (``.kompend/assets.json``) used to locate the vault
        root, or the vault ``base_dir`` itself.
    base_dir:
        Vault root. When given, *assets_file* is ignored.
    """

    def __init__(self, assets_file: Path, base_dir: Path | None = None) -> None:
        self._file = assets_file
        self._db_path = vault_db_path(base_dir if base_dir is not None else assets_file)
        self._data: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Read the asset records into memory.

        A missing store means a fresh start (empty). A corrupt record is logged
        and skipped rather than crashing the pipeline.
        """
        if not self._db_path.exists():
            self._data = {}
            return
        with open_store(self._db_path) as conn:
            rows = conn.execute(f"SELECT id, kind, source, data, summary, ref_count FROM {ASSETS_TABLE}").fetchall()
        data: dict[str, dict[str, Any]] = {}
        for row in rows:
            try:
                record: dict[str, Any] = {
                    AssetField.ID: row["id"],
                    AssetField.KIND: row["kind"],
                    AssetField.SOURCE: row["source"],
                    AssetField.DATA: json.loads(row["data"]),
                    AssetField.SUMMARY: row["summary"],
                    AssetField.REF_COUNT: int(row["ref_count"]),
                }
            except json.JSONDecodeError as exc:
                logger.warning("Asset record %s corrupt (%s) — skipping", row["id"], exc)
                continue
            data[row["id"]] = record
        self._data = data

    def save(self) -> None:
        """Persist the in-memory records (full overwrite). Best-effort."""
        try:
            with open_store(self._db_path) as conn:
                conn.execute(f"DELETE FROM {ASSETS_TABLE}")
                conn.executemany(
                    f"INSERT INTO {ASSETS_TABLE}(id, kind, source, summary, ref_count, data) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            record[AssetField.ID],
                            record[AssetField.KIND],
                            record[AssetField.SOURCE],
                            record[AssetField.SUMMARY],
                            record[AssetField.REF_COUNT],
                            json.dumps(record[AssetField.DATA], ensure_ascii=False),
                        )
                        for record in self._data.values()
                    ],
                )
        except Exception as exc:  # pragma: no cover — DB-dependent
            logger.warning("Could not write asset store (%s): %s", self._db_path, exc)

    def get(self, asset_id: str) -> AssetRecord | None:
        """Return the record for ``asset_id``, or ``None`` if absent."""
        raw = self._data.get(asset_id)
        return AssetRecord.from_dict(raw) if raw is not None else None

    def put(self, record: AssetRecord) -> None:
        """Insert or replace a record (keyed by ``record.id``)."""
        self._data[record.id] = cast("JSONDict", record.to_dict())

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
            yield AssetRecord.from_dict(cast("JSONDict", raw))

    def total_count(self) -> int:
        """Total number of distinct asset records."""
        return len(self._data)
