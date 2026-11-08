"""DB-backed parse cache for ``ParsedDocument`` entries.

Replaces the former ``.kompend/cache/<stem>-<hash>.json`` files. Entries live in
the ``documents`` table of the single store, keyed by ``<stem>-<hash8>`` so
same-stem files with different content never collide.

The stored record keeps the same JSON shape the file cache used, so callers
(``stages/parse_store`` / ``ingest``) only swap their persistence calls:

``source_file``, ``_parser_fingerprint``, ``title``, ``author``, ``language``,
``page_count``, ``metadata``, ``elements``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final, cast

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from kompend.core.identity import IDENTITY_HASH_LEN, source_key
from kompend.core.models import MetadataField
from kompend.core.store import core_connection, open_store, vault_db_path
from kompend.core.store.tables import documents
from kompend.types import JSONDict

#: Key under which the parser fingerprint is stored (cache-validity marker).
PARSE_CACHE_PARSER_KEY: Final[str] = "_parser_fingerprint"

_CACHE_HASH_LEN: Final[int] = IDENTITY_HASH_LEN


def write_document(db_path: Path, key: str, data: JSONDict) -> None:
    """Upsert a parse-cache record from the caller's JSON dict *data*."""
    source_file = str(data.get(MetadataField.SOURCE_FILE) or "")
    fingerprint = str(data.get(PARSE_CACHE_PARSER_KEY) or "")
    metadata = json.dumps(data.get("metadata") or {}, default=str)
    elements = json.dumps(data.get("elements") or [], default=str)
    values = {
        "cache_key": key,
        MetadataField.SOURCE_FILE: source_file,
        "parser_fingerprint": fingerprint,
        "title": str(data.get("title") or ""),
        "author": str(data.get("author") or ""),
        "language": str(data.get("language") or "en"),
        "page_count": cast("int", data.get("page_count") or 0),
        "metadata": metadata,
        "elements": elements,
    }
    with open_store(db_path) as conn, core_connection(conn) as core:
        insert_stmt = sqlite_insert(documents).values(**values)
        core.execute(
            insert_stmt.on_conflict_do_update(
                index_elements=[documents.c.cache_key],
                set_={
                    MetadataField.SOURCE_FILE: insert_stmt.excluded.source_file,
                    "parser_fingerprint": insert_stmt.excluded.parser_fingerprint,
                    "title": insert_stmt.excluded.title,
                    "author": insert_stmt.excluded.author,
                    "language": insert_stmt.excluded.language,
                    "page_count": insert_stmt.excluded.page_count,
                    "metadata": insert_stmt.excluded.metadata,
                    "elements": insert_stmt.excluded.elements,
                },
            )
        )


def clear_documents(db_path: Path) -> None:
    """Delete every parse-cache record (used by ``--force``)."""
    with open_store(db_path) as conn, core_connection(conn) as core:
        core.execute(delete(documents))


def iter_documents(db_path: Path) -> Iterator[tuple[str, JSONDict]]:
    """Yield ``(key, record)`` for every stored parse-cache entry."""
    if not db_path.exists():
        return
    with open_store(db_path) as conn, core_connection(conn) as core:
        keys = [row.cache_key for row in core.execute(select(documents.c.cache_key)).fetchall()]
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


def parse_cache_is_current(data: JSONDict, parser_identity: str) -> bool:
    """True when a parse-cache entry was produced by *parser_identity*.

    The identity is :meth:`kompend.preprocessing.parsers.base.DocumentParser.backend_identity`,
    which composes the backends that can read here with the routing inputs that
    change their output. An entry written before the identity existed stores an
    empty string and returns ``False``: one re-parse that self-heals the cache,
    which is why an empty identity never counts as current.
    """
    return bool(parser_identity) and data.get(PARSE_CACHE_PARSER_KEY) == parser_identity


def read_document(db_path: Path, key: str) -> JSONDict | None:
    """Read a parse-cache record, or ``None`` when absent.

    Returns the same JSON shape the file cache stored (columns re-assembled).
    """
    if not db_path.exists():
        return None
    with open_store(db_path) as conn, core_connection(conn) as core:
        row = core.execute(
            select(
                documents.c.source_file,
                documents.c.parser_fingerprint,
                documents.c.title,
                documents.c.author,
                documents.c.language,
                documents.c.page_count,
                documents.c.metadata,
                documents.c.elements,
            ).where(documents.c.cache_key == key)
        ).fetchone()
    if row is None:
        return None
    try:
        elements = json.loads(row.elements)
        metadata = json.loads(row.metadata)
    except json.JSONDecodeError as exc:
        # A corrupt row is not a cache miss: raise where it is read, naming the
        # row key, so the cause surfaces instead of a silent re-parse.
        raise ValueError(f"Corrupt parse-cache document {key!r} at {db_path}: {exc}") from exc
    if not isinstance(elements, list):
        raise ValueError(f"Corrupt parse-cache document {key!r} at {db_path}: elements is {type(elements).__name__}, expected a list")
    if not isinstance(metadata, dict):
        raise ValueError(f"Corrupt parse-cache document {key!r} at {db_path}: metadata is {type(metadata).__name__}, expected an object")
    return {
        MetadataField.SOURCE_FILE: row.source_file,
        PARSE_CACHE_PARSER_KEY: row.parser_fingerprint,
        "title": row.title,
        "author": row.author,
        "language": row.language,
        "page_count": int(row.page_count),
        "metadata": metadata,
        "elements": elements,
    }


def cache_key(source_file: Path, content_hash: str) -> str:
    """Content-discriminated cache key for *source_file* (``<stem>-<hash8>``).

    Same identity rule as the tier state — see :func:`kompend.core.identity.source_key`.
    """
    return source_key(source_file.stem, content_hash)


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
    "parse_cache_is_current",
    "read_document",
    "write_document",
]
