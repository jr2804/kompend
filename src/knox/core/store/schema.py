"""Single source of truth for the ``.knox/knox.db`` SQLite schema.

This is the DDL home for the one non-vector store (see
``.agents/plans/knox-storage-migration.md`` §2). Every table, its columns, and
the ``schema_version`` row live here so no other module invents schema.

Design rules applied below:

* **Row per record.** Typed columns for anything queried, filtered, or sorted;
  a ``TEXT`` column holding JSON for nested / graph-shaped content that is only
  round-tripped as a whole (``elements``, ``nodes``, ``report`` …).
* **FTS5 only where it earns its place.** ``memory_log`` gets an external-content
  FTS index (query / answer / feedback keyword fallback alongside the zvec
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
)

# ── DDL ─────────────────────────────────────────────────────────────────
_DDL: Final[tuple[str, ...]] = (
    f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA_VERSION_TABLE} (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        version INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    # Parsed-document cache (was cache/<stem>-<content-hash>.json). Keyed by the
    # discriminator so same-stem files with different content never collide.
    f"""
    CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE} (
        cache_key TEXT PRIMARY KEY,
        source_file TEXT NOT NULL,
        parser_fingerprint TEXT NOT NULL DEFAULT '',
        title TEXT NOT NULL DEFAULT '',
        author TEXT NOT NULL DEFAULT '',
        language TEXT NOT NULL DEFAULT 'en',
        page_count INTEGER NOT NULL DEFAULT 0,
        metadata TEXT NOT NULL DEFAULT '{{}}',
        elements TEXT NOT NULL DEFAULT '[]'
    )
    """,
    # Chunking cache (CacheStore "chunk" layer). Payload is the ChunkTree dict.
    f"""
    CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
        cache_key TEXT PRIMARY KEY,
        deps TEXT NOT NULL DEFAULT '{{}}',
        payload TEXT NOT NULL
    )
    """,
    # LLM / resolve / generate / summary cache layers (CacheStore). Layer-scoped
    # because the same dependency key can appear under different layers.
    f"""
    CREATE TABLE IF NOT EXISTS {EXTRACTIONS_TABLE} (
        layer TEXT NOT NULL,
        cache_key TEXT NOT NULL,
        deps TEXT NOT NULL DEFAULT '{{}}',
        payload TEXT NOT NULL,
        PRIMARY KEY (layer, cache_key)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {ASSETS_TABLE} (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL DEFAULT '',
        source TEXT NOT NULL DEFAULT '',
        summary TEXT NOT NULL DEFAULT '',
        ref_count INTEGER NOT NULL DEFAULT 0,
        data TEXT NOT NULL DEFAULT '{{}}'
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {TIER_STATE_TABLE} (
        stem TEXT PRIMARY KEY,
        source_file TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        current_tier INTEGER NOT NULL DEFAULT 0,
        last_compile TEXT NOT NULL DEFAULT '',
        search_hits INTEGER NOT NULL DEFAULT 0,
        last_hit TEXT NOT NULL DEFAULT '',
        ingested_by TEXT NOT NULL DEFAULT ''
    )
    """,
    # last_wiki.json ("last") and compiled_wiki.json ("compiled") share this
    # table, discriminated by kind.
    f"""
    CREATE TABLE IF NOT EXISTS {WIKI_SNAPSHOT_TABLE} (
        kind TEXT PRIMARY KEY,
        nodes TEXT NOT NULL DEFAULT '[]',
        edges TEXT NOT NULL DEFAULT '[]',
        toc TEXT NOT NULL DEFAULT '[]',
        quality_score REAL NOT NULL DEFAULT 0.0,
        iteration INTEGER NOT NULL DEFAULT 0,
        metadata TEXT NOT NULL DEFAULT '{{}}',
        written_at TEXT NOT NULL DEFAULT ''
    )
    """,
    # One row per slug. The dedup store's bookkeeping ({stamp, store_version})
    # is kept under the reserved slug "_meta".
    f"""
    CREATE TABLE IF NOT EXISTS {DEDUP_FINGERPRINTS_TABLE} (
        slug TEXT PRIMARY KEY,
        verdict TEXT NOT NULL DEFAULT '',
        closest_match TEXT,
        closest_score REAL NOT NULL DEFAULT 0.0,
        closest_method TEXT NOT NULL DEFAULT '',
        embedding_score REAL,
        minhash_score REAL,
        embedding TEXT,
        minhash TEXT
    )
    """,
    # Latest-wins per path; deleted = 1 is a tombstone carrying an explicit user
    # removal. A row with deleted = 0 is read-equivalent to the legacy
    # append-only JSONL. See attachment_manifest semantics.
    f"""
    CREATE TABLE IF NOT EXISTS {ATTACHMENT_MANIFEST_TABLE} (
        path TEXT PRIMARY KEY,
        source TEXT NOT NULL DEFAULT '',
        ingestible INTEGER NOT NULL DEFAULT 0,
        embed INTEGER,
        registered_at TEXT NOT NULL DEFAULT '',
        deleted INTEGER NOT NULL DEFAULT 0
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {LINT_RUNS_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        duration_seconds REAL NOT NULL DEFAULT 0.0,
        options TEXT NOT NULL DEFAULT '{{}}',
        llm TEXT,
        config_snapshot TEXT NOT NULL DEFAULT '{{}}',
        wiki_meta TEXT NOT NULL DEFAULT '{{}}',
        report TEXT NOT NULL DEFAULT '{{}}'
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {RUNS_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT NOT NULL,
        run_id TEXT NOT NULL,
        cmd TEXT NOT NULL DEFAULT '',
        ts TEXT NOT NULL,
        status TEXT,
        duration_s REAL,
        exit INTEGER,
        error TEXT
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_{RUNS_TABLE}_run_id ON {RUNS_TABLE}(run_id)",
    f"""
    CREATE TABLE IF NOT EXISTS {MEMORY_STATES_TABLE} (
        node_id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT '',
        confidence REAL NOT NULL DEFAULT 0.5,
        tier TEXT NOT NULL DEFAULT 'working',
        birth_at TEXT NOT NULL DEFAULT '',
        review_at TEXT NOT NULL DEFAULT '',
        review_count INTEGER NOT NULL DEFAULT 0,
        superseded_by TEXT
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {MEMORY_LOG_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mem_id TEXT,
        ts TEXT NOT NULL,
        query TEXT NOT NULL DEFAULT '',
        answer TEXT NOT NULL DEFAULT '',
        feedback TEXT NOT NULL DEFAULT '',
        memory_by TEXT NOT NULL DEFAULT ''
    )
    """,
    # External-content FTS5 over the chronological query memory — the keyword
    # fallback beside zvec semantic recall.
    f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS {MEMORY_LOG_FTS_TABLE} USING fts5(
        query, answer, feedback,
        content='{MEMORY_LOG_TABLE}', content_rowid='id', tokenize='unicode61'
    )
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS {MEMORY_LOG_TABLE}_ai AFTER INSERT ON {MEMORY_LOG_TABLE} BEGIN
        INSERT INTO {MEMORY_LOG_FTS_TABLE}(rowid, query, answer, feedback)
        VALUES (new.id, new.query, new.answer, new.feedback);
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS {MEMORY_LOG_TABLE}_ad AFTER DELETE ON {MEMORY_LOG_TABLE} BEGIN
        INSERT INTO {MEMORY_LOG_FTS_TABLE}({MEMORY_LOG_FTS_TABLE}, rowid, query, answer, feedback)
        VALUES ('delete', old.id, old.query, old.answer, old.feedback);
    END
    """,
    f"""
    CREATE TRIGGER IF NOT EXISTS {MEMORY_LOG_TABLE}_au AFTER UPDATE ON {MEMORY_LOG_TABLE} BEGIN
        INSERT INTO {MEMORY_LOG_FTS_TABLE}({MEMORY_LOG_FTS_TABLE}, rowid, query, answer, feedback)
        VALUES ('delete', old.id, old.query, old.answer, old.feedback);
        INSERT INTO {MEMORY_LOG_FTS_TABLE}(rowid, query, answer, feedback)
        VALUES (new.id, new.query, new.answer, new.feedback);
    END
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {REFERENCES_TABLE} (
        source TEXT NOT NULL,
        ref_id TEXT NOT NULL,
        type TEXT NOT NULL DEFAULT '',
        title TEXT NOT NULL DEFAULT '',
        authors TEXT NOT NULL DEFAULT '[]',
        issued TEXT NOT NULL DEFAULT '',
        container_title TEXT NOT NULL DEFAULT '',
        doi TEXT NOT NULL DEFAULT '',
        url TEXT NOT NULL DEFAULT '',
        raw TEXT NOT NULL DEFAULT '',
        citing_source TEXT NOT NULL DEFAULT '',
        in_text_keys TEXT NOT NULL DEFAULT '[]',
        payload TEXT NOT NULL DEFAULT '{{}}',
        PRIMARY KEY (source, ref_id)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {ENTITY_TYPES_TABLE} (
        concept_key TEXT PRIMARY KEY,
        folder_name TEXT NOT NULL
    )
    """,
    # URL download dedup (was cache/downloaded.json): flat url -> content hash.
    f"""
    CREATE TABLE IF NOT EXISTS {DOWNLOADED_TABLE} (
        url TEXT PRIMARY KEY,
        sha256 TEXT NOT NULL
    )
    """,
    # ``knox organize`` report (was last_organize.json). Tree/move lists are
    # nested, so they stay JSON; dry_run and time are queryable scalars.
    f"""
    CREATE TABLE IF NOT EXISTS {ORGANIZE_REPORTS_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        written_at TEXT NOT NULL DEFAULT '',
        dry_run INTEGER NOT NULL DEFAULT 0,
        categories TEXT NOT NULL DEFAULT '[]',
        moves TEXT NOT NULL DEFAULT '[]',
        link_updates TEXT NOT NULL DEFAULT '{{}}'
    )
    """,
    # Claim/citation verification report (was verification.json). Summary counts
    # are scalars; the per-claim list is JSON.
    f"""
    CREATE TABLE IF NOT EXISTS {VERIFICATION_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        written_at TEXT NOT NULL DEFAULT '',
        total INTEGER NOT NULL DEFAULT 0,
        verified INTEGER NOT NULL DEFAULT 0,
        unsupported INTEGER NOT NULL DEFAULT 0,
        average_confidence REAL NOT NULL DEFAULT 0.0,
        claims TEXT NOT NULL DEFAULT '[]'
    )
    """,
    # Private-page index (was private.json): one row per withheld page.
    f"""
    CREATE TABLE IF NOT EXISTS {PRIVATE_PAGES_TABLE} (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT '',
        saved_at TEXT NOT NULL DEFAULT ''
    )
    """,
    # Per-node provenance (was the provenance/ directory of <id>.json files):
    # one row per node, citations as a JSON list.
    f"""
    CREATE TABLE IF NOT EXISTS {PROVENANCE_TABLE} (
        node_id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT '',
        citations TEXT NOT NULL DEFAULT '[]'
    )
    """,
    # Extracted images and re-encoded .utf8 copies. ``path`` is relative to
    # .knox/ so the same row is valid across a vault move.
    f"""
    CREATE TABLE IF NOT EXISTS {BLOBS_TABLE} (
        path TEXT PRIMARY KEY,
        mime TEXT NOT NULL DEFAULT '',
        size INTEGER NOT NULL DEFAULT 0,
        content BLOB NOT NULL
    )
    """,
    # Review-queue state (was review/_state.json). The human-editable queue
    # markdown stays on disk; only the approve/reject decisions move here.
    f"""
    CREATE TABLE IF NOT EXISTS {REVIEW_STATE_TABLE} (
        slug TEXT PRIMARY KEY,
        status TEXT NOT NULL DEFAULT '',
        title TEXT NOT NULL DEFAULT '',
        reason TEXT NOT NULL DEFAULT '',
        source_node_id TEXT NOT NULL DEFAULT '',
        saved_at TEXT NOT NULL DEFAULT ''
    )
    """,
)


def apply_schema(conn: sqlite3.Connection, version: int) -> None:
    """Create every table/index/trigger and stamp *version*.

    Every statement is ``IF NOT EXISTS``, so this is fully idempotent: calling
    it on an existing store adds only missing tables/triggers and never touches
    existing rows. This is why :func:`knox.core.store.ensure_schema` can apply
    the schema on every open instead of early-returning on a version match —
    additive changes (new tables/columns) need no version bump. A version bump
    is reserved for *incompatible* changes (drops, retypes), where the caller
    recreates first.
    """
    for statement in _DDL:
        conn.execute(statement)
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
        for statement in _DDL:
            scratch.execute(statement)
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
