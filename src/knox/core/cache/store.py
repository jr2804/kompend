"""Content-addressed cache store with dependency tracking.

Entries live in the single ``.knox/knox.db`` store, in the ``chunks`` table
(the ``chunk`` layer) or the ``extractions`` table (every other layer, keyed by
``(layer, cache_key)``). The key is derived from the dependency hashes, so
identical inputs always map to the same row.

On lookup the store verifies that *every* stored dependency hash still matches
the caller-provided current hash.  If any dependency changed, the entry is
treated as a miss and the caller re-generates the value.

Concurrency
-----------
Different inputs write to different rows (different keys), so workers never
contend on the same entry.  Writes are idempotent upserts; the DB is opened
with WAL, so readers do not block the writer (see ``knox.core.store``).
"""

# ruff: noqa: S608 — SQL is built from Final[str] table-name constants only.

from __future__ import annotations

import hashlib
import json
from base64 import b64decode, b64encode
from dataclasses import fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

from knox.core.models import (
    Chunk,
    CompiledWiki,
    DocumentElement,
    ElementType,
    ParsedDocument,
    ResolvedChunk,
    WikiNode,
)
from knox.core.store import open_store, vault_db_path
from knox.core.store.schema import CHUNKS_TABLE, EXTRACTIONS_TABLE
from knox.core.utils.time import iso_local
from knox.logging import get_logger
from knox.types import CacheValue, JSONDict, JSONValue

# ponytail: registry-boundary (C2) — cache round-trips known dataclasses via
# JSON tags. Add new cacheable types here, or store to_dict() payloads instead.
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

#: Cache layer whose payloads are keyed only by ``cache_key`` (no layer column).
_CHUNK_LAYER: Final[str] = "chunk"

logger = get_logger(__name__)


class CacheStore:
    """Content-addressed cache with dependency validation.

    Cached values are serialized to JSON with explicit type tags for
    dataclasses, ``Path``, ``StrEnum``, ``bytes`` and ``tuple``.  This
    avoids ``pickle`` entirely while preserving the KNOX model graph.

    Parameters
    ----------
    cache_dir:
        Cache directory used to locate the vault root (typically
        ``.knox/cache/``), or the vault ``base_dir`` itself.
    base_dir:
        Vault root. When given (needed if *cache_dir* is overridden outside
        ``.knox/``) it is used directly.
    """

    def __init__(self, cache_dir: Path, base_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir
        self._base_dir = base_dir
        self._db_path = vault_db_path(base_dir if base_dir is not None else cache_dir)

    def get(self, layer: str, deps: dict[str, str]) -> CacheValue | None:
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
        payload = self._read_payload(layer, key)
        if payload is None:
            return None

        try:
            decoded = decode_json(json.loads(payload))
            if not isinstance(decoded, dict):
                raise ValueError(f"Cache entry {layer}/{key} is not a JSON object")
        except Exception:
            logger.debug("Cache read error for %s/%s - treating as miss", layer, key, exc_info=True)
            return None
        entry = cast("JSONDict", decoded)

        # Validate every stored dependency against the caller's current deps.
        stored_deps = entry.get("deps")
        if not isinstance(stored_deps, dict):
            raise ValueError(f"Cache entry {layer}/{key} has missing or malformed deps block")
        for dep_name, stored_hash in stored_deps.items():
            if deps.get(dep_name) != stored_hash:
                logger.debug("Cache dep mismatch %s/%s: %s changed", layer, key, dep_name)
                return None

        return entry.get("value")

    def get_dict(self, layer: str, deps: dict[str, str]) -> JSONDict | None:
        """Cached JSON-object payload, or ``None`` on miss.

        Raises ValueError when an entry exists but its payload is not a
        JSON object — that is a corrupt cache, not a miss.
        """
        value = self.get(layer, deps)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"Corrupt cache entry {layer}: payload is {type(value).__name__}, expected object")
        return cast("JSONDict", value)

    def get_as[T: object](self, layer: str, deps: dict[str, str], expected: type[T]) -> T | None:
        """Cached value validated against *expected*, or ``None`` on miss.

        Raises ValueError when an entry exists but holds a different type —
        that is a corrupt cache, not a miss.
        """
        value = self.get(layer, deps)
        if value is None:
            return None
        if not isinstance(value, expected):
            raise ValueError(f"Corrupt cache entry {layer}: payload is {type(value).__name__}, expected {expected.__name__}")
        return value

    def set(self, layer: str, deps: dict[str, str], value: CacheValue) -> None:
        """Store a value with its dependency hashes (idempotent upsert).

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
        entry = {
            "key": key,
            "layer": layer,
            "deps": deps,
            "value": value,
            "created_at": iso_local(),
        }
        try:
            payload = json.dumps(encode_json(entry))
        except (TypeError, ValueError):
            # Cache write failure is non-fatal (next read is a miss), but make it visible.
            logger.warning("Cache write failed for %s/%s", layer, key, exc_info=True)
            return
        self._write_payload(layer, key, payload)
        logger.debug("Cache write OK layer=%s key=%s size=%d", layer, key, len(payload))

    def has(self, layer: str, deps: dict[str, str]) -> bool:
        """Quick existence check — ``True`` if the row exists.

        .. note::

           This does **not** validate dependencies.  Use :meth:`get`
           when you need a correctness check.
        """
        return self._read_payload(layer, make_cache_key(deps)) is not None

    def invalidate(self, layer: str | None = None) -> None:
        """Delete cached entries.

        Parameters
        ----------
        layer:
            If given, only that layer is cleared.  ``None`` wipes every
            layer owned by this store (the ``chunk`` layer and all
            ``extractions`` layers).
        """
        if layer is None:
            sqls = [
                (f"DELETE FROM {CHUNKS_TABLE}", ()),
                (f"DELETE FROM {EXTRACTIONS_TABLE}", ()),
            ]
        elif layer == _CHUNK_LAYER:
            sqls = [(f"DELETE FROM {CHUNKS_TABLE}", ())]
        else:
            sqls = [(f"DELETE FROM {EXTRACTIONS_TABLE} WHERE layer = ?", (layer,))]
        with open_store(self._db_path) as conn:
            for sql, params in sqls:
                conn.execute(sql, params)
        logger.debug("Cache invalidated: layer=%s", layer or "*")

    def _read_payload(self, layer: str, key: str) -> str | None:
        """Return the stored payload JSON for a layer/key, or ``None``."""
        if not self._db_path.exists():
            return None
        if layer == _CHUNK_LAYER:
            sql = f"SELECT payload FROM {CHUNKS_TABLE} WHERE cache_key = ?"
            params = (key,)
        else:
            sql = f"SELECT payload FROM {EXTRACTIONS_TABLE} WHERE layer = ? AND cache_key = ?"
            params = (layer, key)
        with open_store(self._db_path) as conn:
            row = conn.execute(sql, params).fetchone()
        return None if row is None else cast("str", row["payload"])

    def _write_payload(self, layer: str, key: str, payload: str) -> None:
        """Upsert the payload JSON for a layer/key."""
        with open_store(self._db_path) as conn:
            if layer == _CHUNK_LAYER:
                conn.execute(
                    f"INSERT INTO {CHUNKS_TABLE}(cache_key, payload) VALUES (?, ?) ON CONFLICT(cache_key) DO UPDATE SET payload = excluded.payload",
                    (key, payload),
                )
            else:
                conn.execute(
                    f"INSERT INTO {EXTRACTIONS_TABLE}(layer, cache_key, payload) VALUES (?, ?, ?) "
                    f"ON CONFLICT(layer, cache_key) DO UPDATE SET payload = excluded.payload",
                    (layer, key, payload),
                )


def encode_json(obj: object) -> JSONValue:  # noqa: PLR0911
    """Recursively encode cache values for JSON serialization.

    Dataclasses, ``Path``, ``StrEnum``, ``bytes`` and tuples are tagged so
    they can be reconstructed safely without ``pickle``.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        data = {f.name: encode_json(getattr(obj, f.name)) for f in fields(obj)}
        data["__dataclass__"] = type(obj).__name__
        return data
    if isinstance(obj, Path):
        return {"__path__": str(obj)}
    if isinstance(obj, StrEnum):
        return {"__enum__": type(obj).__name__, "value": str(obj)}
    if isinstance(obj, bytes):
        return {"__bytes__": b64encode(obj).decode("ascii")}
    if isinstance(obj, tuple):
        return {"__tuple__": [encode_json(v) for v in obj]}
    if isinstance(obj, dict):
        encoded: JSONDict = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"Cannot encode cache dict with {type(k).__name__} key")
            encoded[k] = encode_json(v)
        return encoded
    if isinstance(obj, list):
        return [encode_json(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError(f"Cannot encode cache value of type {type(obj).__name__}")


def decode_json(obj: JSONValue) -> CacheValue:  # noqa: PLR0911
    """Recursively decode JSON values produced by ``_encode_json``."""
    if isinstance(obj, dict):
        if "__dataclass__" in obj:
            cls_name = obj.pop("__dataclass__")
            cls = _CACHE_DATACLASSES.get(cls_name)
            if cls is None:
                raise ValueError(f"Unknown dataclass tag: {cls_name}")
            decoded = {k: decode_json(v) for k, v in obj.items()}
            # Dataclass instances are first-class cache values by design; the
            # registry lookup erases the concrete type at the boundary.
            return cast("CacheValue", cls(**decoded))
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
            return tuple(decode_json(v) for v in obj["__tuple__"])
        return {k: decode_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_json(v) for v in obj]
    return obj


def make_cache_key(deps: dict[str, str]) -> str:
    """Deterministic 16-char hex key from a sorted dependency dict.

    Two calls with the same deps (same keys and values, regardless of
    insertion order) produce the same key.
    """
    blob = json.dumps(deps, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
