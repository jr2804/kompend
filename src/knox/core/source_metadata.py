"""Read and mutate source-file metadata in the parse-cache store.

Source metadata (``tags``, ``attributes``) lives in the ``metadata`` JSON of
each ``documents`` row — the DB-backed parse cache. Both ``knox sources
tag/attr`` (CLI) and the MCP source tools operate on it, so the logic lives
here once rather than being globbed from a directory nothing writes.

All entry points take the vault ``cache_dir`` (a handle the store resolves to
``.knox/knox.db``); callers keep their existing signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from knox.core.models import MetadataField
from knox.core.pipeline.parse_cache import (
    extract_text_from_data,
    iter_documents,
    read_document,
    write_document,
)
from knox.core.store import vault_db_path
from knox.types import JSONDict, JSONValue


@dataclass(frozen=True)
class SourceChange:
    """Outcome of a metadata mutation."""

    changed: bool
    message: str


def load_source_entries(cache_dir: Path) -> dict[str, JSONDict]:
    """Return every parse-cache record keyed by its ``source_file``."""
    entries: dict[str, JSONDict] = {}
    for _key, record in iter_documents(vault_db_path(cache_dir)):
        source_file = record.get(MetadataField.SOURCE_FILE)
        if isinstance(source_file, str) and source_file:
            entries[source_file] = record
    return entries


def cached_text(cache_dir: Path, source: str) -> str | None:
    """Concatenated parse-cache text for *source*, or ``None`` when uncached."""
    found = _resolve(cache_dir, source)
    return None if found is None else extract_text_from_data(found[1])


def find_entry(cache_dir: Path, source: str) -> JSONDict | None:
    """Return the parse-cache record matching *source*, or ``None``.

    *source* may be a ``source_file`` string (``sources/doc.pdf``) or a bare
    stem; when several same-stem entries exist the exact ``source_file`` wins.
    """
    found = _resolve(cache_dir, source)
    return None if found is None else found[1]


def apply_tag(cache_dir: Path, file: str, tag: str, *, add: bool) -> SourceChange:
    """Add or remove *tag* on *file*'s parse-cache record.

    Raises ``LookupError`` when *file* has no cache entry.
    """
    found = _resolve(cache_dir, file)
    if found is None:
        raise LookupError(file)
    key, record = found
    meta = _meta(record)
    existing = source_tags(record)

    if add:
        if tag in existing:
            return SourceChange(False, f"Tag '{tag}' already present on {file}")
        existing.append(tag)
        message = f"Added tag '{tag}' on {file}"
    else:
        if tag not in existing:
            return SourceChange(False, f"Tag '{tag}' not found on {file}")
        existing.remove(tag)
        message = f"Removed tag '{tag}' on {file}"

    meta["tags"] = cast("JSONValue", list(existing))
    write_document(vault_db_path(cache_dir), key, record)
    return SourceChange(True, message)


def source_tags(record: JSONDict) -> list[str]:
    """Tags recorded in *record*'s metadata (``[]`` when absent/malformed)."""
    meta = record.get("metadata")
    tags = meta.get("tags") if isinstance(meta, dict) else None
    return [t for t in tags if isinstance(t, str)] if isinstance(tags, list) else []


def apply_attribute(cache_dir: Path, file: str, key_value: str, *, add: bool) -> SourceChange:
    """Set or remove a ``key=value`` attribute on *file*'s parse-cache record.

    Raises ``LookupError`` when *file* has no cache entry, ``ValueError`` when
    *add* is requested with a malformed ``key=value``.
    """
    found = _resolve(cache_dir, file)
    if found is None:
        raise LookupError(file)
    key, record = found
    meta = _meta(record)
    existing = source_attributes(record)

    if add:
        if "=" not in key_value:
            raise ValueError(key_value)
        name, value = key_value.split("=", 1)
        if existing.get(name) == value:
            return SourceChange(False, f"Attribute '{name}={value}' already present on {file}")
        existing[name] = value
        message = f"Set {name}={value}"
    else:
        if key_value not in existing:
            return SourceChange(False, f"Attribute key '{key_value}' not found on {file}")
        del existing[key_value]
        message = f"Removed {key_value}"

    meta["attributes"] = cast("JSONValue", dict(existing))
    write_document(vault_db_path(cache_dir), key, record)
    return SourceChange(True, message)


def source_attributes(record: JSONDict) -> dict[str, str]:
    """Key=value attributes recorded in *record*'s metadata."""
    meta = record.get("metadata")
    attrs = meta.get("attributes") if isinstance(meta, dict) else None
    if not isinstance(attrs, dict):
        return {}
    return {k: v for k, v in attrs.items() if isinstance(k, str) and isinstance(v, str)}


def _resolve(cache_dir: Path, source: str) -> tuple[str, JSONDict] | None:
    """Return ``(key, record)`` for *source*, or ``None`` when uncached."""
    db_path = vault_db_path(cache_dir)
    stem = Path(source).stem
    keys = sorted(key for key, _ in iter_documents(db_path) if key == stem or key.startswith(f"{stem}-"))
    if not keys:
        return None
    if len(keys) > 1:
        for key in keys:
            record = read_document(db_path, key)
            if record is not None and record.get(MetadataField.SOURCE_FILE) == source:
                return key, record
    record = read_document(db_path, keys[0])
    return None if record is None else (keys[0], record)


def _meta(record: JSONDict) -> JSONDict:
    raw = record.get("metadata")
    if isinstance(raw, dict):
        return raw
    fresh: JSONDict = {}
    record["metadata"] = cast("JSONValue", fresh)
    return fresh
