"""Open / create the single ``.kompend/kompend.db`` SQLite store.

One place owns connection setup, PRAGMAs, and the clean-cut version policy, so
no caller opens a raw ``sqlite3`` handle against the store.

Concurrency choices (measured, not guessed):

* ``PRAGMA journal_mode=WAL`` — one writer concurrent with many readers; the
  mode is persistent, so it is set (and verified) on every open.
* ``PRAGMA foreign_keys=ON`` — per-connection in SQLite, so it must be set here.
* ``PRAGMA busy_timeout`` — a blocked writer waits instead of raising
  ``SQLITE_BUSY`` immediately.
* ``check_same_thread=False`` — this CPython's ``sqlite3`` is built in
  serialized mode (``sqlite3.threadsafety == 3``), so a connection may be used
  from multiple threads. We still recommend one connection per operation via
  :func:`open_store` rather than sharing; the flag only removes the Python-side
  guard, it does not add app-level locking.

Clean-cut policy (B3): a version mismatch drops every user object and recreates
the schema. There is **no** importer and no legacy reader — old ``.kompend/*.json*``
files are simply not read. See :mod:`kompend.core.store.legacy` for detection only.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from kompend.core.store.schema import SCHEMA_VERSION, apply_schema, current_version, reconcile_columns
from kompend.logging import get_logger

_BUSY_TIMEOUT_MS = 5000

logger = get_logger(__name__)


@contextmanager
def open_store(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Context manager: open, ensure the schema, commit on success, close.

    On entry a missing database is created and a version mismatch triggers a
    clean-cut recreate. The connection is committed on clean exit and always
    closed.
    """
    conn = connect(db_path)
    try:
        ensure_schema(conn, db_path)
        yield conn
        conn.commit()
    finally:
        conn.close()


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with the required PRAGMAs applied.

    Creates parent directories. The caller owns closing. Prefer
    :func:`open_store` unless a long-lived handle is genuinely needed.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    return conn


def ensure_schema(conn: sqlite3.Connection, db_path: Path) -> None:
    """Create the schema, recreating from scratch only on an incompatible version.

    The DDL is fully idempotent (``IF NOT EXISTS`` throughout), so the schema
    is applied on every open: tables/triggers added after a store was first
    created are picked up without a version bump and without touching existing
    rows. A ``SCHEMA_VERSION`` mismatch is reserved for genuinely incompatible
    changes and triggers the clean-cut recreate.
    """
    version = current_version(conn)
    if version is not None and version != SCHEMA_VERSION:
        logger.warning(
            "kompend.db schema version %d != %d — recreating store at %s. .kompend/ state is NOT migrated; run `kompend compile --force` to rebuild.",
            version,
            SCHEMA_VERSION,
            db_path,
        )
        _drop_all(conn)
    apply_schema(conn, SCHEMA_VERSION)
    added = reconcile_columns(conn)
    if added:
        logger.warning(
            "kompend.db schema drift reconciled at %s: added missing columns %s (no version bump, no data loss)",
            db_path,
            ", ".join(added),
        )
    conn.commit()


def _drop_all(conn: sqlite3.Connection) -> None:
    """Drop every user object (triggers, virtual tables, tables, views).

    Ordering matters: triggers first (they reference the tables), then virtual
    tables (dropping an FTS5 table removes its shadow tables), then plain tables
    and views.
    """
    rows = conn.execute("SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
    triggers = [row["name"] for row in rows if row["type"] == "trigger"]
    virtual = [row["name"] for row in rows if row["type"] == "table" and (row["sql"] or "").upper().startswith("CREATE VIRTUAL")]
    tables = [row["name"] for row in rows if row["type"] == "table" and row["name"] not in virtual]
    views = [row["name"] for row in rows if row["type"] == "view"]
    for name in triggers:
        conn.execute(f'DROP TRIGGER IF EXISTS "{name}"')
    for name in virtual:
        conn.execute(f'DROP TABLE IF EXISTS "{name}"')
    for name in views:
        conn.execute(f'DROP VIEW IF EXISTS "{name}"')
    for name in tables:
        conn.execute(f'DROP TABLE IF EXISTS "{name}"')
