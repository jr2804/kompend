"""Persistent store for dedup fingerprints with automatic invalidation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict

from knox.logging import get_logger
from knox.types import JSONDict, JSONValue

# Current store schema version — bump when fingerprint format changes.
_STORE_VERSION = "1"

_META_KEY = "_meta"

logger = get_logger(__name__)


class StoreMeta(TypedDict):
    """Schema for the store bookkeeping record."""

    stamp: str
    store_version: str


class StoreEntry(TypedDict):
    """Schema for one slug entry in the persistent dedup store (JSON)."""

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

    Lives at ``.knox/dedup/fingerprints.json``.
    Auto-invalidates when the config that produced the fingerprints changes:
    :meth:`stamp` fingerprints the relevant dedup config keys, and
    :meth:`is_fresh` compares the stored stamp against the current one.

    The file layout is one JSON object per slug plus a ``_meta`` record;
    :attr:`_entries` and :attr:`_meta` keep those two schemas apart.
    """

    def __init__(self, store_file: Path, config: JSONDict | None = None) -> None:
        self.store_file = store_file
        self.config = config or {}
        self._meta: StoreMeta = {"stamp": "", "store_version": _STORE_VERSION}
        self._entries: dict[str, StoreEntry] = {}

    @property
    def stamp(self) -> str:
        """Fingerprint over all config keys that affect fingerprint computation."""
        parts = [
            _cfg_str(self.config, "dedup.matchers", "embedding,minhash"),
            _cfg_str(self.config, "dedup.model", ""),
            str(self.config.get("dedup.minhash_lsh_threshold", 0.8)),
            _cfg_str(self.config, "compile.embedding.model", ""),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def is_fresh(self) -> bool:
        """True iff the persisted stamp matches the current config fingerprint."""
        return self._meta["stamp"] == self.stamp and self._meta["store_version"] == _STORE_VERSION

    def load(self) -> None:
        """Load data from disk. Missing file = empty store."""
        if not self.store_file.exists():
            return
        try:
            raw_text = self.store_file.read_text(encoding="utf-8")
            raw: JSONValue = json.loads(raw_text)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt store at %s, starting fresh", self.store_file, exc_info=True)
            return
        if not isinstance(raw, dict):
            raise ValueError(f"Corrupt dedup store {self.store_file}: top level must be a JSON object")
        meta = raw.pop(_META_KEY, {})
        if not isinstance(meta, dict):
            raise ValueError(f"Corrupt dedup store {self.store_file}: {_META_KEY!r} must be a JSON object")
        for slug, entry in raw.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Corrupt dedup store {self.store_file}: entry {slug!r} is not a JSON object")
        self._meta = StoreMeta(
            stamp=str(meta.get("stamp", "")),
            store_version=str(meta.get("store_version", "")),
        )
        self._entries = dict(raw.items())

    def save(self) -> None:
        """Persist data to disk with the current stamp."""
        self._meta = StoreMeta(stamp=self.stamp, store_version=_STORE_VERSION)
        payload = {**self._entries, _META_KEY: self._meta}
        self.store_file.parent.mkdir(parents=True, exist_ok=True)
        self.store_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def get(self, slug: str) -> StoreEntry | None:
        """Return fingerprint data for *slug*, or ``None``."""
        return self._entries.get(slug)

    def __setitem__(self, slug: str, value: StoreEntry) -> None:
        self._entries[slug] = value

    def __contains__(self, slug: str) -> bool:
        return slug in self._entries

    @property
    def slugs(self) -> list[str]:
        """All document slugs in the store (excluding metadata keys)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self.slugs)

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


def _cfg_str(config: JSONDict, key: str, default: str) -> str:
    """Read a string config value, rejecting other types."""
    raw = config.get(key, default)
    if not isinstance(raw, str):
        raise TypeError(f"Config key {key!r} must be a string, got {type(raw).__name__}")
    return raw
