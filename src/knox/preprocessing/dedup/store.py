"""Persistent store for dedup fingerprints with automatic invalidation."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Current store schema version — bump when fingerprint format changes.
_STORE_VERSION = "1"


@dataclass
class ContentStore:
    """Persistent store for derived artifacts with automatic invalidation.

    Subclasses define :meth:`stamp` to return a fingerprint of the
    configuration that produced this data. If the stored stamp doesn't
    match, the caller knows to rebuild.

    The ``_meta`` dict holds the version stamp. All other entries are
    keyed by document slug and hold per-document fingerprint data.
    """

    path: Path
    _data: dict[str, Any] = field(default_factory=dict)

    @property
    def stamp(self) -> str:
        """Fingerprint of the config that produced this artifact.

        Subclasses must override this to include all relevant config keys.
        """
        raise NotImplementedError

    def is_fresh(self) -> bool:
        """True iff the persisted stamp matches the current config fingerprint."""
        meta = self._data.get("_meta", {})
        return meta.get("stamp") == self.stamp and meta.get("store_version") == _STORE_VERSION

    def load(self) -> None:
        """Load data from disk. Missing file = empty store."""
        if self.path.exists():
            try:
                raw = self.path.read_text(encoding="utf-8")
                self._data = json.loads(raw)
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt store at %s, starting fresh", self.path, exc_info=True)
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        """Persist data to disk with the current stamp."""
        self._data["_meta"] = {
            "stamp": self.stamp,
            "store_version": _STORE_VERSION,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

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


class DedupStore(ContentStore):
    """Persistent store of dedup fingerprints (+ closest-match metadata).

    Lives at ``.knox/dedup/fingerprints.json``.
    """

    def __init__(self, path: Path, *, config: dict[str, Any] | None = None) -> None:
        super().__init__(path=path)
        self._config = config or {}

    @property
    def stamp(self) -> str:
        """Fingerprint over all config keys that affect fingerprint computation."""
        cfg = self._config
        parts = [
            cfg.get("dedup.matchers", "embedding,minhash"),
            cfg.get("dedup.model", ""),
            str(cfg.get("dedup.minhash_lsh_threshold", 0.8)),
            cfg.get("compile.embedding.model", ""),
            str(cfg.get("compile.embedding.dim", 384)),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

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
