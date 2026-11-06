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

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants;
# no runtime value ever enters an identifier or statement.

from __future__ import annotations

import functools
import sqlite3
from typing import Final

from sqlalchemy import create_engine
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.schema import CreateTable

from kompend.core.identity import IDENTITY_HASH_LEN
from kompend.core.store.tables import METADATA, tier_state

#: Scratch name used while rebuilding ``tier_state`` in place.
_TIER_STATE_LEGACY_TABLE: Final[str] = "tier_state_legacy"

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
        f"INSERT OR REPLACE INTO {TIER_STATE_TABLE}(key, stem, source_file, content_hash, current_tier, "
        f"last_compile, search_hits, last_hit, ingested_by) "
        f"SELECT stem || '-' || substr(content_hash, 1, {IDENTITY_HASH_LEN}), stem, source_file, content_hash, "
        f"current_tier, last_compile, search_hits, last_hit, ingested_by FROM {_TIER_STATE_LEGACY_TABLE}"
    )
    conn.execute(f"DROP TABLE {_TIER_STATE_LEGACY_TABLE}")
    return True


def _tier_state_exists(conn: sqlite3.Connection) -> bool:
    """True when the ``tier_state`` table exists in this store."""
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (TIER_STATE_TABLE,)).fetchone()
    return row is not None


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
    METADATA.create_all(_core_engine(conn))
    conn.execute(
        f"INSERT INTO {SCHEMA_VERSION_TABLE}(id, version, created_at) VALUES (1, ?, datetime('now')) ON CONFLICT(id) DO UPDATE SET version = excluded.version",
        (version,),
    )


def current_version(conn: sqlite3.Connection) -> int | None:
    """Return the stored schema version, or ``None`` when the schema is absent."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (SCHEMA_VERSION_TABLE,),
    ).fetchone()
    if exists is None:
        return None
    row = conn.execute(f"SELECT version FROM {SCHEMA_VERSION_TABLE} WHERE id = 1").fetchone()
    return int(row["version"]) if row is not None else None


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

    ``StaticPool`` holds exactly one connection — the caller's — and the
    engine is never ``dispose()``d: disposing closes pooled connections, which
    here would close the caller's handle. Dropping the engine is enough; the
    caller's reference keeps the connection open.
    """
    return create_engine("sqlite+pysqlite://", creator=lambda: conn, poolclass=StaticPool)
