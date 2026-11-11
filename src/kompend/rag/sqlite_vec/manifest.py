"""The index manifest — what the last build recorded, and why a store is stale.

The manifest is a single row in ``kompend.db`` beside the vector file: the
indexed id set, the per-node content hashes, the embed-text version, the
embedding model and the backend that wrote the vectors. It is the record
:meth:`SqliteVecWikiStore.is_stale` judges the loaded wiki against, and the
record ``kompend doctor`` reads with :func:`recorded_embedding_model`.

Statements go through SQLAlchemy Core against the shared ``vector_manifest``
table, so the table name and the column identifiers live in one place. Rows
are materialised inside the connection block — the Core wrapper is torn down
at block exit, and a lazy mapping view would fail in the caller instead.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError

from kompend.core.models import WikiNode
from kompend.core.paths import KOMP_DB_FILE
from kompend.core.store import connect, core_connection, open_store
from kompend.core.store.tables import vector_manifest
from kompend.core.utils.time import iso_local
from kompend.logging import LOG_SAMPLE_LIMIT, get_logger
from kompend.rag.sqlite_vec.schema import BACKEND, BACKEND_COLUMN, EMBEDDING_MODEL_COLUMN
from kompend.rag.vector_store import EMBED_TEXT_VERSION, content_hash

#: The manifest is one row, keyed by a fixed id.
_MANIFEST_ROW_ID = 1

logger = get_logger(__name__)


def write_manifest(db_file: Path, nodes: list[WikiNode], model_id: str) -> None:
    """Record the indexed id set, content hashes, embed-text version, model and backend."""
    ids = sorted(node.id for node in nodes)
    hashes = {node.id: content_hash(node) for node in nodes}
    values = {
        "id": _MANIFEST_ROW_ID,
        "embed_version": EMBED_TEXT_VERSION,
        EMBEDDING_MODEL_COLUMN: model_id,
        BACKEND_COLUMN: BACKEND,
        "ids": json.dumps(ids),
        "content_hashes": json.dumps(hashes, sort_keys=True),
        "written_at": iso_local(),
    }
    try:
        with open_store(manifest_db_path(db_file)) as conn, core_connection(conn) as core:
            statement = sqlite_insert(vector_manifest).values(**values)
            core.execute(
                statement.on_conflict_do_update(
                    index_elements=[vector_manifest.c.id],
                    set_={column: statement.excluded[column] for column in values if column != "id"},
                )
            )
    except (sqlite3.Error, SQLAlchemyError, OSError) as exc:
        logger.warning(
            "Could not record the vector manifest in %s: %s",
            manifest_db_path(db_file),
            exc,
        )


def manifest_db_path(db_file: Path) -> Path:
    """``kompend.db`` beside the vector store — home of the index manifest row.

    The vectors file always sits in the vault's ``.kompend/`` next to the
    single non-vector store, so the manifest is a row there instead of a
    loose JSON sidecar: ``.kompend/`` holds one state file, not a scattering
    of state files.  The vector branch itself (the store file) stays separate
    and regenerable.
    """
    return db_file.parent / KOMP_DB_FILE


def read_manifest(manifest_db: Path) -> Mapping[str, Any] | None:
    """The manifest row of the last build, or ``None`` when there is none.

    The row is materialised here: ``core_connection`` tears its wrapper down
    at block exit, so a mapping view handed to the caller would already be
    dead. The columns the caller reads are the store's own names.
    """
    try:
        with open_store(manifest_db) as conn, core_connection(conn) as core:
            row = core.execute(select(vector_manifest).where(vector_manifest.c.id == _MANIFEST_ROW_ID)).mappings().first()
    except (sqlite3.Error, SQLAlchemyError) as exc:
        logger.warning("Vector manifest unreadable — treating store as stale: %s", exc)
        return None
    return None if row is None else dict(row)


def manifest_mismatch(row: Mapping[str, Any], model_id: str) -> str | None:
    """Why the manifest does not describe the current vectors, or ``None``.

    A row written before the hashes were tracked is not a mismatch here —
    the content comparison catches that.
    """
    if row[BACKEND_COLUMN] != BACKEND:
        return f"backend {row[BACKEND_COLUMN]!r} != {BACKEND!r}"
    if row["embed_version"] != EMBED_TEXT_VERSION:
        return f"embed-text version {row['embed_version']!r} != {EMBED_TEXT_VERSION!r}"
    if row[EMBEDDING_MODEL_COLUMN] != model_id:
        return f"embedding model {row[EMBEDDING_MODEL_COLUMN]!r} != {model_id!r}"
    return None


def content_changed(recorded: str, nodes: list[WikiNode], db_file: Path) -> bool:
    """Compare recorded content hashes against *nodes*.

    The id check alone cannot see an edited page: a `lint --fix`, a hand
    edit, or a recompile that produced different prose leaves the id set
    unchanged while the vectors describe text search no longer returns.
    Hashing the indexed payload closes that blind spot.

    A row written before hashes were tracked (empty mapping) is stale by
    definition — one rebuild records them.
    """
    stored: dict[str, str] = json.loads(recorded) if recorded else {}
    if not stored:
        logger.info("Vector manifest has no content hashes (pre-tracking build) — rebuilding: %s", db_file)
        return True
    expected = {node.id: content_hash(node) for node in nodes}
    changed = sorted(node_id for node_id, digest in expected.items() if stored.get(node_id) != digest)
    if not changed:
        return False
    logger.info(
        "Vector index content changed for %d of %d node(s), e.g. %s — rebuilding: %s",
        len(changed),
        len(expected),
        changed[:LOG_SAMPLE_LIMIT],
        db_file,
    )
    return True


def recorded_embedding_model(manifest_db: Path) -> str | None:
    """The ``provider/model`` string the current vectors were built with.

    The manifest row in ``kompend.db`` records the embedding provider's
    ``model_id`` at build time. The doctor resolves this tag's serving digest
    and compares it with the configured embedder's: equal digests under
    different tags mean a pending rebuild that would produce byte-identical
    vectors. ``None`` when there is no manifest row (never indexed, unreadable)
    or the recorded model is empty.

    The read goes through :func:`kompend.core.store.connect`, deliberately
    *not* :func:`~kompend.core.store.open_store`: the doctor calls this on
    stores written by any kompend generation, and ``open_store``'s schema
    ensure would clean-cut-recreate a version-mismatched ``kompend.db`` — a
    destructive write a diagnostic must never perform.

    A store that predates vectors has no manifest table; the missing table is an
    answer here, not a bug, so its error is one of the cases that yields ``None``.
    """
    if not manifest_db.exists():
        return None
    try:
        conn = connect(manifest_db)
        try:
            with core_connection(conn) as core:
                row = core.execute(select(vector_manifest.c[EMBEDDING_MODEL_COLUMN]).where(vector_manifest.c.id == _MANIFEST_ROW_ID)).first()
        finally:
            conn.close()
    except (sqlite3.Error, SQLAlchemyError):
        return None
    if row is None:
        return None
    recorded = str(row[0]).strip()
    return recorded or None
