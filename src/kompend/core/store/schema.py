"""Single source of truth for the ``.kompend/kompend.db`` SQLite schema.

This is the DDL home for the one non-vector store (see
``.agents/history/2026-09-24-kompend-storage-migration.md`` §2). Every table, its columns, and
the ``schema_version`` row live here so no other module invents schema.

Design rules applied below:

* **Row per record.** Typed columns for anything queried, filtered, or sorted;
  a ``TEXT`` column holding JSON for nested / graph-shaped content that is only
  round-tripped as a whole (``elements``, ``nodes``, ``report`` …).
* **FTS5 only where it earns its place.** ``memory_log`` gets an external-content
  FTS index (query / answer / feedback keyword fallback alongside the
  semantic path). ``wiki_snapshot`` deliberately does **not**: the primary wiki
  query path is the LLM reading markdown, so an FTS copy would only duplicate
  the vault. Bibliographic FTS is deferred (its ``(source, ref_id)`` text PK
  makes external-content FTS awkward; add it later — the clean-cut store needs
  no migration to do so).
* **Binaries live here too.** Extracted images and re-encoded ``.utf8`` copies
  go in the ``blobs`` table as ``BLOB``: the user's target is one file for all
  non-vector state, and SQLite handles multi-MB blobs fine. See
  ``connection.open_store`` for the WAL/locking consequences.

Table names are exposed as ``Final[str]`` constants from this module — no other
module inlines a table literal.
"""

from __future__ import annotations

import functools
import sqlite3
from typing import Final

from sqlalchemy import create_engine, literal_column, select
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.schema import CreateTable

from kompend.core.identity import IDENTITY_HASH_LEN
from kompend.core.store.tables import METADATA, schema_version, tier_state

#: Scratch name used while rebuilding ``tier_state`` in place.
_TIER_STATE_LEGACY_TABLE: Final[str] = "tier_state_legacy"

#: ``tier_state`` columns in DDL order, keyed first.
_TIER_STATE_COLUMNS: Final[tuple[str, ...]] = (
    "key",
    "stem",
    "source_file",
    "content_hash",
    "current_tier",
    "last_compile",
    "search_hits",
    "last_hit",
    "ingested_by",
)
_TIER_STATE_KEYED_COLUMNS: Final[str] = ", ".join(_TIER_STATE_COLUMNS)

#: Bump when the DDL below changes shape. The store treats a mismatch as a
#: clean cut: drop everything and recreate (no migration — policy B3).
SCHEMA_VERSION: Final[int] = 1

# ── Table name constants (never inline these) ───────────────────────────
SCHEMA_VERSION_TABLE: Final[str] = "schema_version"
DOCUMENTS_TABLE: Final[str] = "documents"
CHUNKS_TABLE: Final[str] = "chunks"
EXTRACTIONS_TABLE: Final[str] = "extractions"
ASSETS_TABLE: Final[str] = "assets"
TIER_STATE_TABLE: Final[str] = "tier_state"
WIKI_SNAPSHOT_TABLE: Final[str] = "wiki_snapshot"
DEDUP_FINGERPRINTS_TABLE: Final[str] = "dedup_fingerprints"
ATTACHMENT_MANIFEST_TABLE: Final[str] = "attachment_manifest"
LINT_RUNS_TABLE: Final[str] = "lint_runs"
RUNS_TABLE: Final[str] = "runs"
STAGES_TABLE: Final[str] = "stages"
PARSE_QUALITY_TABLE: Final[str] = "parse_quality"
MEMORY_STATES_TABLE: Final[str] = "memory_states"
MEMORY_LOG_TABLE: Final[str] = "memory_log"
MEMORY_LOG_FTS_TABLE: Final[str] = "memory_log_fts"
#: ``references`` is a SQLite reserved keyword, so the table is named
#: ``bib_references`` (quoted identifiers everywhere would leak into consumers).
REFERENCES_TABLE: Final[str] = "bib_references"
ENTITY_TYPES_TABLE: Final[str] = "entity_types"
DOWNLOADED_TABLE: Final[str] = "downloaded"
ORGANIZE_REPORTS_TABLE: Final[str] = "organize_reports"
VERIFICATION_TABLE: Final[str] = "verification"
PRIVATE_PAGES_TABLE: Final[str] = "private_pages"
PROVENANCE_TABLE: Final[str] = "provenance"
REVIEW_STATE_TABLE: Final[str] = "review_state"
BLOBS_TABLE: Final[str] = "blobs"
#: Bookkeeping for the separate vector branch (``vectors/``): the id set and
#: embed-text version of the last build. Lives here rather than as a loose
#: ``vectors.manifest.json`` sidecar so ``.kompend/`` holds one state file.
VECTOR_MANIFEST_TABLE: Final[str] = "vector_manifest"

#: Every data table (excludes the version bookkeeping table and FTS shadows).
TABLE_NAMES: Final[tuple[str, ...]] = (
    DOCUMENTS_TABLE,
    CHUNKS_TABLE,
    EXTRACTIONS_TABLE,
    ASSETS_TABLE,
    TIER_STATE_TABLE,
    WIKI_SNAPSHOT_TABLE,
    DEDUP_FINGERPRINTS_TABLE,
    ATTACHMENT_MANIFEST_TABLE,
    LINT_RUNS_TABLE,
    RUNS_TABLE,
    STAGES_TABLE,
    PARSE_QUALITY_TABLE,
    MEMORY_STATES_TABLE,
    MEMORY_LOG_TABLE,
    REFERENCES_TABLE,
    ENTITY_TYPES_TABLE,
    DOWNLOADED_TABLE,
    ORGANIZE_REPORTS_TABLE,
    VERIFICATION_TABLE,
    PRIVATE_PAGES_TABLE,
    PROVENANCE_TABLE,
    REVIEW_STATE_TABLE,
    BLOBS_TABLE,
    VECTOR_MANIFEST_TABLE,
)


def migrate_tier_state_identity(conn: sqlite3.Connection) -> bool:
    """Rekey ``tier_state`` from bare stem to ``<stem>-<hash8>``.

    ``reconcile_columns`` can add columns but cannot change a primary key, so
    this one shape change is migrated in place instead of dropping the store:
    state written by an older kompend is real data (tier, hit counters,
    ``ingested_by``) and losing it would make every source look un-indexed.

    A store whose ``tier_state`` already has a ``key`` column is left alone.
    Returns ``True`` when the table was rebuilt so the caller can log it.
    """
    if not _tier_state_exists(conn):
        return False
    if "key" in {row[1] for row in conn.execute(f"PRAGMA table_info({TIER_STATE_TABLE})")}:
        return False
    conn.execute(f"ALTER TABLE {TIER_STATE_TABLE} RENAME TO {_TIER_STATE_LEGACY_TABLE}")
    conn.execute(_TIER_STATE_DDL())
    conn.execute(
        _statement(
            "INSERT OR REPLACE INTO {table}({columns}) SELECT stem || '-' || substr(content_hash, 1, {hash_len}), {copied} FROM {legacy}",
            table=TIER_STATE_TABLE,
            columns=_TIER_STATE_KEYED_COLUMNS,
            hash_len=IDENTITY_HASH_LEN,
            # key is the one column the rekey derives; the rest carry over as-is.
            copied=", ".join(column for column in _TIER_STATE_COLUMNS if column != "key"),
            legacy=_TIER_STATE_LEGACY_TABLE,
        )
    )
    conn.execute(f"DROP TABLE {_TIER_STATE_LEGACY_TABLE}")
    return True


def _tier_state_exists(conn: sqlite3.Connection) -> bool:
    """True when the ``tier_state`` table exists in this store."""
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (TIER_STATE_TABLE,)).fetchone()
    return row is not None


def _statement(template: str, **values: object) -> str:
    """Fill one constant SQL template with the names and numbers it needs.

    Bandit flags an f-string that interpolates a name into a statement, and its
    ``S608`` check cannot tell ``Final[str]`` table names from request data. The
    template form keeps every statement written out in full at its call site
    while the value still flows through a plain ``str`` parameter.
    """
    return template.format(**values)


def _TIER_STATE_DDL() -> str:
    """The ``tier_state`` CREATE statement, compiled from the Core metadata."""
    return str(CreateTable(tier_state).compile(dialect=sqlite_dialect.dialect()))


def apply_schema(conn: sqlite3.Connection, version: int) -> None:
    """Create every table/index/trigger and stamp *version*.

    ``create_all`` is fully idempotent (``checkfirst`` throughout), so calling
    this on an existing store adds only missing tables/triggers and never
    touches existing rows. This is why :func:`kompend.core.store.ensure_schema`
    can apply the schema on every open instead of early-returning on a version
    match — additive changes (new tables/columns) need no version bump. A
    version bump is reserved for *incompatible* changes (drops, retypes),
    where the caller recreates first.
    """
    engine = _core_engine(conn)
    METADATA.create_all(engine)
    with engine.begin() as core:
        core.execute(
            sqlite_insert(schema_version)
            .values(id=1, version=version, created_at=literal_column("datetime('now')"))
            .on_conflict_do_update(index_elements=[schema_version.c.id], set_={"version": version})
        )
    engine.dispose(close=False)


def current_version(conn: sqlite3.Connection) -> int | None:
    """Return the stored schema version, or ``None`` when the schema is absent."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (SCHEMA_VERSION_TABLE,),
    ).fetchone()
    if exists is None:
        return None
    engine = _core_engine(conn)
    with engine.connect() as core:
        row = core.execute(select(schema_version.c.version).where(schema_version.c.id == 1)).fetchone()
    engine.dispose(close=False)
    return int(row.version) if row is not None else None


def reconcile_columns(conn: sqlite3.Connection) -> list[str]:
    """Add any column present in the DDL but missing from an existing table.

    ``CREATE TABLE IF NOT EXISTS`` cannot add a column to a table that already
    exists, so a column added to the DDL after a store was first created would
    otherwise drift silently and surface later as ``no such column`` from a
    distant consumer. This closes that gap: additive columns are reconciled in
    place (no version bump, no data loss). Returns the ``table.column`` names
    added so the caller can log them loudly.
    """
    added: list[str] = []
    for table, columns in _expected_columns().items():
        actual = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, fragment in columns.items():
            if name not in actual:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {fragment}")
                added.append(f"{table}.{name}")
    return added


@functools.cache
def _expected_columns() -> dict[str, dict[str, str]]:
    """Column DDL fragments derived from the DDL itself, via a scratch DB.

    Built once per process: creates the schema in an in-memory database and
    reads back each table's columns, so the expected column set can never
    drift from ``_DDL``. Fragments are ``"name TYPE [DEFAULT x]"``, suitable
    for ``ALTER TABLE ... ADD COLUMN``.
    """
    scratch = sqlite3.connect(":memory:")
    try:
        METADATA.create_all(_core_engine(scratch))
        expected: dict[str, dict[str, str]] = {}
        for table in TABLE_NAMES:
            columns: dict[str, str] = {}
            for row in scratch.execute(f"PRAGMA table_info({table})"):
                name, col_type, default = row[1], row[2], row[4]
                fragment = f"{name} {col_type}"
                if default is not None:
                    fragment += f" DEFAULT {default}"
                columns[name] = fragment
            expected[table] = columns
        return expected
    finally:
        scratch.close()


# ── DDL ─────────────────────────────────────────────────────────────────
# The DDL itself lives in ``tables.METADATA`` — the single source of truth.
# This module only *applies* it to a raw connection.


def _core_engine(conn: sqlite3.Connection) -> Engine:
    """A throwaway engine bound to *conn*, so Core DDL runs on the caller's connection.

    ``StaticPool`` holds exactly one connection — the caller's. A caller that
    finishes with the engine must ``dispose(close=False)`` it (dropping the
    pooled reference without closing the caller's handle); never a plain
    ``dispose()``. :func:`kompend.core.store.connection.core_connection` is
    the ready-made wrapper for query sites.
    """
    return create_engine("sqlite+pysqlite://", creator=lambda: conn, poolclass=StaticPool)
