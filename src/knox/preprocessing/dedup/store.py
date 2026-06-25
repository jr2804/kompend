"""Persistent store for dedup fingerprints with automatic invalidation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from knox.logging import get_logger

logger = get_logger(__name__)

# Current store schema version — bump when fingerprint format changes.
_STORE_VERSION = "1"


class DedupStore:
    """Persistent store of dedup fingerprints (+ closest-match metadata).

    Lives at ``.knox/dedup/fingerprints.json``.
    Auto-invalidates when the config that produced the fingerprints changes:
    :meth:`stamp` fingerprints the relevant dedup config keys, and
    :meth:`is_fresh` compares the stored stamp against the current one.

    The ``_meta`` dict holds the version stamp. All other entries are
    keyed by document slug and hold per-document fingerprint data.
    """

    def __init__(
        self,
        store_file: Path,
        config: dict[str, Any] | None = None,
        _data: dict[str, Any] | None = None,
    ) -> None:
        self.store_file = store_file
        self.config = config or {}
        self._data = _data or {}

    @property
    def stamp(self) -> str:
        """Fingerprint over all config keys that affect fingerprint computation."""
        cfg = self.config
        parts = [
            cfg.get("dedup.matchers", "embedding,minhash"),
            cfg.get("dedup.model", ""),
            str(cfg.get("dedup.minhash_lsh_threshold", 0.8)),
            cfg.get("compile.embedding.model", ""),
            str(cfg.get("compile.embedding.dim", 384)),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def is_fresh(self) -> bool:
        """True iff the persisted stamp matches the current config fingerprint."""
        meta = self._data.get("_meta", {})
        return meta.get("stamp") == self.stamp and meta.get("store_version") == _STORE_VERSION

    def load(self) -> None:
        """Load data from disk. Missing file = empty store."""
        if self.store_file.exists():
            try:
                raw = self.store_file.read_text(encoding="utf-8")
                self._data = json.loads(raw)
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt store at %s, starting fresh", self.store_file, exc_info=True)
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        """Persist data to disk with the current stamp."""
        self._data["_meta"] = {
            "stamp": self.stamp,
            "store_version": _STORE_VERSION,
        }
        self.store_file.parent.mkdir(parents=True, exist_ok=True)
        self.store_file.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    def get(self, slug: str, default: Any = None) -> Any:
        """Return fingerprint data for *slug*, or *default*."""
        return self._data.get(slug, default)

    def __setitem__(self, slug: str, value: Any) -> None:
        self._data[slug] = value

    def __contains__(self, slug: str) -> bool:
        return slug in self._data

    @property
    def slugs(self) -> list[str]:
        """All document slugs in the store (excluding metadata keys)."""
        return [k for k in self._data if not k.startswith("_")]

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
        existing = self._data.get(slug, {})
        current_score = existing.get("closest_score", -1.0)
        if closest_score > current_score:
            self._data[slug] = {
                **existing,
                "closest_match": closest_match,
                "closest_score": closest_score,
                "closest_method": closest_method,
                "embedding_score": embedding_score,
                "minhash_score": minhash_score,
            }
