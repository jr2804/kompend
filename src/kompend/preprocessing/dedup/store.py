"""Persistent store for dedup fingerprints with automatic invalidation.

Rows live in the ``dedup_fingerprints`` table of the single ``.kompend/kompend.db``
store. The bookkeeping record
(``{stamp, store_version}``) is kept under the reserved slug ``"_meta"`` — its
JSON payload is stored in the ``embedding`` column.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import RowMapping

from kompend.core.config import DEFAULT_MATCHERS, parse_matchers
from kompend.core.keys import (
    COMPILE_EMBEDDING_MODEL,
    DEDUP_MATCHERS,
    DEDUP_MINHASH_LSH_THRESHOLD,
    DEDUP_MODEL,
)
from kompend.core.store import core_connection, open_store, vault_db_path
from kompend.core.store.tables import dedup_fingerprints
from kompend.logging import get_logger
from kompend.types import JSONDict, JSONValue

# Current store schema version — bump when fingerprint format changes.
_STORE_VERSION = "1"

#: Reserved primary-key slug for the store bookkeeping row.
_META_SLUG = "_meta"

logger = get_logger(__name__)


class StoreMeta(TypedDict):
    """Schema for the store bookkeeping record."""

    stamp: str
    store_version: str


class StoreEntry(TypedDict):
    """Schema for one slug entry in the persistent dedup store."""

    verdict: str
    closest_match: str | None
    closest_score: float
    closest_method: str
    embedding_score: float | None
    minhash_score: float | None
    embedding: list[JSONValue] | None
    minhash: list[JSONValue] | None


class DedupStore:
    """Persistent store of dedup fingerprints (+ closest-match metadata).

    Backed by the ``dedup_fingerprints`` table in the vault store.
    Auto-invalidates when the config that produced the fingerprints changes:
    :meth:`stamp` fingerprints the relevant dedup config keys, and
    :meth:`is_fresh` compares the stored stamp against the current one.
    """

    def __init__(self, store_file: Path, config: JSONDict | None = None) -> None:
        self.store_file = store_file
        self.config = config or {}
        self._db_path = vault_db_path(store_file)
        self._meta: StoreMeta = {"stamp": "", "store_version": _STORE_VERSION}
        self._entries: dict[str, StoreEntry] = {}

    @property
    def stamp(self) -> str:
        """Fingerprint over all config keys that affect fingerprint computation."""
        parts = [
            self._canonical_matchers(),
            _cfg_str(self.config, DEDUP_MODEL, ""),
            str(self.config.get(DEDUP_MINHASH_LSH_THRESHOLD, 0.8)),
            _cfg_str(self.config, COMPILE_EMBEDDING_MODEL, ""),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def is_fresh(self) -> bool:
        """True iff the persisted stamp matches the current config fingerprint."""
        return self._meta["stamp"] == self.stamp and self._meta["store_version"] == _STORE_VERSION

    def load(self) -> None:
        """Load data from the vault store. Missing store = empty."""
        self._meta = {"stamp": "", "store_version": _STORE_VERSION}
        self._entries = {}
        if not self._db_path.exists():
            return
        with open_store(self._db_path) as conn, core_connection(conn) as core:
            rows = core.execute(select(dedup_fingerprints)).mappings().all()
        for row in rows:
            slug = str(row["slug"])
            if slug == _META_SLUG:
                self._meta = _decode_meta(row["embedding"])
                continue
            self._entries[slug] = _decode_entry(row)

    def save(self) -> None:
        """Persist data to the vault store with the current stamp."""
        with open_store(self._db_path) as conn, core_connection(conn) as core:
            core.execute(delete(dedup_fingerprints))
            meta = sqlite_insert(dedup_fingerprints).values(
                slug=_META_SLUG,
                embedding=json.dumps({"stamp": self.stamp, "store_version": _STORE_VERSION}),
            )
            core.execute(meta)
            rows = [_entry_row(slug, entry) for slug, entry in self._entries.items()]
            if rows:
                core.execute(sqlite_insert(dedup_fingerprints), rows)

    def get(self, slug: str) -> StoreEntry | None:
        """Return fingerprint data for *slug*, or ``None``."""
        return self._entries.get(slug)

    def __setitem__(self, slug: str, value: StoreEntry) -> None:
        self._entries[slug] = value

    def __contains__(self, slug: str) -> bool:
        return slug in self._entries

    @property
    def slugs(self) -> list[str]:
        """All document slugs in the store (excluding bookkeeping)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def update_closest(
        self,
        slug: str,
        closest_match: str | None,
        closest_score: float,
        closest_method: str,
        *,
        embedding_score: float | None = None,
        minhash_score: float | None = None,
    ) -> None:
        """Update the closest-match metadata for an existing entry.

        Only replaces if *closest_score* is higher than the current value.
        """
        existing = self._entries.get(slug)
        current_score = existing["closest_score"] if existing else -1.0
        if closest_score > current_score:
            base = existing or StoreEntry(
                verdict="",
                closest_match=None,
                closest_score=-1.0,
                closest_method="",
                embedding_score=None,
                minhash_score=None,
                embedding=None,
                minhash=None,
            )
            self._entries[slug] = StoreEntry(
                verdict=base["verdict"],
                closest_match=closest_match,
                closest_score=closest_score,
                closest_method=closest_method,
                embedding_score=embedding_score,
                minhash_score=minhash_score,
                embedding=base["embedding"],
                minhash=base["minhash"],
            )

    def _canonical_matchers(self) -> str:
        """Canonical ``dedup.matchers`` spelling — member values, comma-joined.

        Parsing through :func:`~kompend.core.config.parse_matchers` makes the
        stamp insensitive to case and whitespace while still changing when
        the matcher *set* (or its order) changes. An unknown name
        raises here rather than stamping a config the orchestrator would
        later reject.
        """
        raw = self.config.get(DEDUP_MATCHERS, ",".join(m.value for m in DEFAULT_MATCHERS))
        return ",".join(m.value for m in parse_matchers(raw))


def _decode_meta(raw: object) -> StoreMeta:
    if isinstance(raw, str) and raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Corrupt dedup store bookkeeping row; starting fresh", exc_info=True)
        else:
            if isinstance(data, dict):
                return {"stamp": str(data.get("stamp", "")), "store_version": str(data.get("store_version", ""))}
    return {"stamp": "", "store_version": _STORE_VERSION}


def _decode_entry(row: RowMapping) -> StoreEntry:
    return StoreEntry(
        verdict=str(row["verdict"] or ""),
        closest_match=None if row["closest_match"] is None else str(row["closest_match"]),
        closest_score=float(row["closest_score"] or 0.0),
        closest_method=str(row["closest_method"] or ""),
        embedding_score=None if row["embedding_score"] is None else float(row["embedding_score"]),
        minhash_score=None if row["minhash_score"] is None else float(row["minhash_score"]),
        embedding=_decode_json_list(row["embedding"]),
        minhash=_decode_json_list(row["minhash"]),
    )


def _decode_json_list(raw: object) -> list[JSONValue] | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def _entry_row(slug: str, entry: StoreEntry) -> dict[str, object]:
    return {
        "slug": slug,
        "verdict": entry["verdict"],
        "closest_match": entry["closest_match"],
        "closest_score": entry["closest_score"],
        "closest_method": entry["closest_method"],
        "embedding_score": entry["embedding_score"],
        "minhash_score": entry["minhash_score"],
        "embedding": _encode_json_list(entry["embedding"]),
        "minhash": _encode_json_list(entry["minhash"]),
    }


def _encode_json_list(value: list[JSONValue] | None) -> str | None:
    return None if value is None else json.dumps(value)


def _cfg_str(config: JSONDict, key: str, default: str) -> str:
    value = config.get(key, default)
    return value if isinstance(value, str) else str(value)
