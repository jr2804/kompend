"""DB-backed parse cache for ``ParsedDocument`` entries.

Replaces the former ``.knox/cache/<stem>-<hash>.json`` files. Entries live in
the ``documents`` table of the single store, keyed by ``<stem>-<hash8>`` so
same-stem files with different content never collide.

The stored record keeps the same JSON shape the file cache used, so callers
(``_parse`` / ``ingest``) only swap their persistence calls:

``source_file``, ``_parser_fingerprint``, ``title``, ``author``, ``language``,
``page_count``, ``metadata``, ``elements``.
"""

# ruff: noqa: S608 — SQL is built from Final[str] table-name constants only.

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final, cast

from knox.core.models import MetadataField
from knox.core.store import open_store, vault_db_path
from knox.core.store.schema import DOCUMENTS_TABLE
from knox.types import JSONDict

#: Key under which the parser fingerprint is stored (cache-validity marker).
PARSE_CACHE_PARSER_KEY: Final[str] = "_parser_fingerprint"

_CACHE_HASH_LEN: Final[int] = 8


def cache_key(source_file: Path, content_hash: str) -> str:
    """Content-discriminated cache key for *source_file* (``<stem>-<hash8>``)."""
    return f"{source_file.stem}-{content_hash[:_CACHE_HASH_LEN]}"


def write_document(db_path: Path, key: str, data: JSONDict) -> None:
    """Upsert a parse-cache record from the caller's JSON dict *data*."""
    source_file = str(data.get(MetadataField.SOURCE_FILE) or "")
    fingerprint = str(data.get(PARSE_CACHE_PARSER_KEY) or "")
    metadata = json.dumps(data.get("metadata") or {}, default=str)
    elements = json.dumps(data.get("elements") or [], default=str)
    with open_store(db_path) as conn:
        conn.execute(
            f"INSERT INTO {DOCUMENTS_TABLE}"
            f"(cache_key, source_file, parser_fingerprint, title, author, language, page_count, metadata, elements) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            f"ON CONFLICT(cache_key) DO UPDATE SET "
            f"source_file = excluded.source_file, parser_fingerprint = excluded.parser_fingerprint, "
            f"title = excluded.title, author = excluded.author, language = excluded.language, "
            f"page_count = excluded.page_count, metadata = excluded.metadata, elements = excluded.elements",
            (
                key,
                source_file,
                fingerprint,
                str(data.get("title") or ""),
                str(data.get("author") or ""),
                str(data.get("language") or "en"),
                cast("int", data.get("page_count") or 0),
                metadata,
                elements,
            ),
        )


def clear_documents(db_path: Path) -> None:
    """Delete every parse-cache record (used by ``--force``)."""
    with open_store(db_path) as conn:
        conn.execute(f"DELETE FROM {DOCUMENTS_TABLE}")


def iter_documents(db_path: Path) -> Iterator[tuple[str, JSONDict]]:
    """Yield ``(key, record)`` for every stored parse-cache entry."""
    if not db_path.exists():
        return
    with open_store(db_path) as conn:
        keys = [row["cache_key"] for row in conn.execute(f"SELECT cache_key FROM {DOCUMENTS_TABLE}").fetchall()]
    for key in keys:
        data = read_document(db_path, key)
        if data is not None:
            yield key, data


def extract_text_from_cached(json_file: Path) -> str:
    """Extract text for the entry named by a legacy ``<key>.json`` path.

    DB-backed: resolves the vault from *json_file* and reads the ``documents``
    row. Returns ``""`` when the entry is absent (clean cut — no legacy-file
    read).
    """
    db_path = vault_db_path(json_file)
    data = read_document(db_path, normalize_key(json_file.name))
    if data is None:
        return ""
    return extract_text_from_data(data)


def normalize_key(name: str) -> str:
    """Accept either a bare key or a legacy ``.json`` filename."""
    return name.removesuffix(".json")


def read_document(db_path: Path, key: str) -> JSONDict | None:
    """Read a parse-cache record, or ``None`` when absent.

    Returns the same JSON shape the file cache stored (columns re-assembled).
    """
    if not db_path.exists():
        return None
    with open_store(db_path) as conn:
        row = conn.execute(
            f"SELECT source_file, parser_fingerprint, title, author, language, page_count, metadata, elements FROM {DOCUMENTS_TABLE} WHERE cache_key = ?",
            (key,),
        ).fetchone()
    if row is None:
        return None
    try:
        elements = json.loads(row["elements"])
        metadata = json.loads(row["metadata"])
    except json.JSONDecodeError:
        return None
    return {
        MetadataField.SOURCE_FILE: row["source_file"],  # nocheck: right side is a knox.db column
        PARSE_CACHE_PARSER_KEY: row["parser_fingerprint"],
        "title": row["title"],
        "author": row["author"],
        "language": row["language"],
        "page_count": int(row["page_count"]),
        "metadata": metadata,
        "elements": elements,
    }


def extract_text_from_data(data: JSONDict) -> str:
    """Concatenate string element content from a parse-cache record."""
    text = ""
    elements = data.get("elements")
    if isinstance(elements, list):
        for element in elements:
            if isinstance(element, dict):
                content = element.get("content")
                if isinstance(content, str) and content.strip():
                    text += content + "\n\n"
    return text


def content_hash_prefix(data: bytes) -> str:
    """SHA-256 prefix of *data*, matching :func:`cache_key`'s discriminator."""
    return hashlib.sha256(data).hexdigest()[:_CACHE_HASH_LEN]


__all__ = [
    "PARSE_CACHE_PARSER_KEY",
    "cache_key",
    "clear_documents",
    "content_hash_prefix",
    "extract_text_from_cached",
    "extract_text_from_data",
    "iter_documents",
    "normalize_key",
    "read_document",
    "write_document",
]
