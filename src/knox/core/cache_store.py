"""Content-addressed cache store with dependency tracking.

Each cached output is stored as a pickle file in
``<cache_dir>/<layer>/<sha256-key>.pkl``.  The key is derived from the
dependency hashes, so identical inputs always map to the same file.

On lookup the store verifies that *every* stored dependency hash still
matches the caller-provided current hash.  If any dependency changed,
the entry is treated as a miss and the caller re-generates the value.

Thread safety
-------------
Different inputs write to different cache files (different keys), so
workers never contend on the same file.  Writes use an atomic
write-to-temp-then-rename pattern.  Two workers that compute the same
key produce the same value, so the race is harmless (idempotent write).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def make_cache_key(deps: dict[str, str]) -> str:
    """Deterministic 16-char hex key from a sorted dependency dict.

    Two calls with the same deps (same keys and values, regardless of
    insertion order) produce the same key.
    """
    blob = json.dumps(deps, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class CacheStore:
    """Content-addressed cache with dependency validation.

    Parameters
    ----------
    cache_dir:
        Root directory for cache files (typically ``.knox/cache/``).
        Created lazily on first write.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, layer: str, deps: dict[str, str]) -> Any | None:
        """Look up a cached value.

        Parameters
        ----------
        layer:
            Cache layer name (e.g. ``"chunk"``, ``"resolve"``,
            ``"extract"``).
        deps:
            Current dependency hashes for *every* input that the
            cached value depends on.  If any hash differs from the
            stored entry the cache is considered stale.

        Returns:
        -------
        The cached value, or ``None`` on miss / stale / error.
        """
        key = make_cache_key(deps)
        path = self._resolve(layer, key)
        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                entry: dict[str, Any] = pickle.load(f)  # noqa: S301 — cache dir is not an untrusted source
        except Exception:
            logger.debug("Cache read error for %s — treating as miss", path)
            return None

        # Validate every stored dependency against the caller's current deps.
        stored_deps = entry.get("deps", {})
        for dep_name, stored_hash in stored_deps.items():
            if deps.get(dep_name) != stored_hash:
                logger.debug(
                    "Cache dep mismatch %s/%s: %s changed", layer, key, dep_name
                )
                return None

        return entry.get("value")

    def set(self, layer: str, deps: dict[str, str], value: Any) -> None:
        """Store a value with its dependency hashes.

        Uses an atomic write-to-temp-then-rename so concurrent writers
        never see a partial file.

        Parameters
        ----------
        layer:
            Cache layer name.
        deps:
            Every input hash that the value depends on.  Must be the
            same dict that will be passed to :meth:`get` for lookup.
        value:
            Arbitrary Python object (pickle-serialised).
        """
        key = make_cache_key(deps)
        path = self._resolve(layer, key)
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "key": key,
            "layer": layer,
            "deps": deps,
            "value": value,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }

        # Atomic write to avoid partial reads from concurrent workers.
        try:
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                pickle.dump(entry, f)
            os.replace(tmp, path)
        except Exception:
            logger.debug("Cache write error for %s", path, exc_info=True)

    def has(self, layer: str, deps: dict[str, str]) -> bool:
        """Quick existence check — returns ``True`` if the file exists.

        .. note::

           This does **not** validate dependencies.  Use :meth:`get`
           when you need a correctness check.
        """
        path = self._resolve(layer, make_cache_key(deps))
        return path.exists()

    def invalidate(self, layer: str | None = None) -> None:
        """Delete cached entries.

        Parameters
        ----------
        layer:
            If given, only that layer is cleared.  ``None`` wipes the
            entire cache directory.
        """
        if layer is not None:
            target = self._cache_dir / layer
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
                logger.debug("Cache layer %s invalidated", layer)
        elif self._cache_dir.is_dir():
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            logger.debug("Entire cache invalidated")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self, layer: str, key: str) -> Path:
        """Resolve the on-disk path for a cache entry."""
        return self._cache_dir / layer / f"{key}.pkl"
