"""Content-addressed cache store with dependency tracking.

Each cached output is stored as a JSON file in
``<cache_dir>/<layer>/<sha256-key>.json``.  The key is derived from the
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
import shutil
import tempfile
from base64 import b64decode, b64encode
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from knox.core.models import (
    Chunk,
    CompiledWiki,
    DocumentElement,
    ElementType,
    ParsedDocument,
    ResolvedChunk,
    WikiNode,
)

logger = logging.getLogger(__name__)


_CACHE_DATACLASSES: dict[str, type] = {
    "DocumentElement": DocumentElement,
    "ParsedDocument": ParsedDocument,
    "Chunk": Chunk,
    "ResolvedChunk": ResolvedChunk,
    "WikiNode": WikiNode,
    "CompiledWiki": CompiledWiki,
}

_CACHE_ENUMS: dict[str, type[StrEnum]] = {
    "ElementType": ElementType,
}


def _encode_json(obj: Any) -> Any:  # noqa: PLR0911
    """Recursively encode cache values for JSON serialization.

    Dataclasses, ``Path``, ``StrEnum``, ``bytes`` and tuples are tagged so
    they can be reconstructed safely without ``pickle``.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        data = {f.name: _encode_json(getattr(obj, f.name)) for f in fields(obj)}
        data["__dataclass__"] = type(obj).__name__
        return data
    if isinstance(obj, Path):
        return {"__path__": str(obj)}
    if isinstance(obj, StrEnum):
        return {"__enum__": type(obj).__name__, "value": str(obj)}
    if isinstance(obj, bytes):
        return {"__bytes__": b64encode(obj).decode("ascii")}
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode_json(v) for v in obj]}
    if isinstance(obj, dict):
        return {k: _encode_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode_json(v) for v in obj]
    return obj


def _decode_json(obj: Any) -> Any:  # noqa: PLR0911
    """Recursively decode JSON values produced by ``_encode_json``."""
    if isinstance(obj, dict):
        if "__dataclass__" in obj:
            cls_name = obj.pop("__dataclass__")
            cls = _CACHE_DATACLASSES.get(cls_name)
            if cls is None:
                raise ValueError(f"Unknown dataclass tag: {cls_name}")
            decoded = {k: _decode_json(v) for k, v in obj.items()}
            return cls(**decoded)
        if "__path__" in obj:
            return Path(obj["__path__"])
        if "__enum__" in obj:
            enum_cls = _CACHE_ENUMS.get(obj["__enum__"])
            if enum_cls is None:
                raise ValueError(f"Unknown enum tag: {obj['__enum__']}")
            return enum_cls(obj["value"])
        if "__bytes__" in obj:
            return b64decode(obj["__bytes__"])
        if "__tuple__" in obj:
            return tuple(_decode_json(v) for v in obj["__tuple__"])
        return {k: _decode_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_json(v) for v in obj]
    return obj


def make_cache_key(deps: dict[str, str]) -> str:
    """Deterministic 16-char hex key from a sorted dependency dict.

    Two calls with the same deps (same keys and values, regardless of
    insertion order) produce the same key.
    """
    blob = json.dumps(deps, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class CacheStore:
    """Content-addressed cache with dependency validation.

    Cached values are serialized to JSON with explicit type tags for
    dataclasses, ``Path``, ``StrEnum``, ``bytes`` and ``tuple``.  This
    avoids ``pickle`` entirely while preserving the KNOX model graph.

    Parameters
    ----------
    cache_dir:
        Root directory for cache files (typically ``.knox/cache/``).
        Created lazily on first write.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

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
            entry: dict[str, Any] = _decode_json(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            logger.debug("Cache read error for %s - treating as miss", path, exc_info=True)
            return None

        # Validate every stored dependency against the caller's current deps.
        stored_deps = entry.get("deps", {})
        for dep_name, stored_hash in stored_deps.items():
            if deps.get(dep_name) != stored_hash:
                logger.debug("Cache dep mismatch %s/%s: %s changed", layer, key, dep_name)
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
            Arbitrary Python object (JSON-serialised).
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
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp", text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(_encode_json(entry), f)
            os.replace(tmp, path)
            logger.debug("Cache write OK layer=%s key=%s size=%d", layer, key, path.stat().st_size)
        except OSError:
            # Cache write failure is non-fatal (next read is a miss), but make it visible.
            logger.warning("Cache write failed for %s", path, exc_info=True)

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

    def _resolve(self, layer: str, key: str) -> Path:
        """Resolve the on-disk path for a cache entry."""
        return self._cache_dir / layer / f"{key}.json"
