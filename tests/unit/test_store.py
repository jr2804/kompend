"""Unit tests for the single ``.kompend/kompend.db`` store foundation."""

# ruff: noqa: S608 — test SQL is built from Final[str] table-name constants only.

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from kompend.core.paths import VaultPaths
from kompend.core.store import SCHEMA_VERSION, TABLE_NAMES, connect, obsolete_artifacts, open_store
from kompend.core.store.connection import _drop_all
from kompend.core.store.schema import (
    BLOBS_TABLE,
    MEMORY_LOG_FTS_TABLE,
    MEMORY_LOG_TABLE,
    REFERENCES_TABLE,
    SCHEMA_VERSION_TABLE,
    TIER_STATE_TABLE,
)
from kompend.core.store.tables import METADATA

_INSERT_CASES: tuple[tuple[str, str, tuple[object, ...]], ...] = (
    ("documents", "INSERT INTO documents(cache_key, source_file) VALUES (?, ?)", ("doc-1", "s.md")),
    ("chunks", "INSERT INTO chunks(cache_key, payload) VALUES (?, ?)", ("c1", "{}")),
    ("extractions", "INSERT INTO extractions(layer, cache_key, payload) VALUES (?, ?, ?)", ("extract", "e1", "{}")),
    ("assets", "INSERT INTO assets(id, kind) VALUES (?, ?)", ("a1", "table")),
    ("tier_state", "INSERT INTO tier_state(key, stem, source_file, content_hash) VALUES (?, ?, ?, ?)", ("t1-abc", "t1", "s.md", "h")),
    ("wiki_snapshot", "INSERT INTO wiki_snapshot(kind) VALUES (?)", ("last",)),
    ("dedup_fingerprints", "INSERT INTO dedup_fingerprints(slug, closest_score) VALUES (?, ?)", ("d1", 0.5)),
    ("attachment_manifest", "INSERT INTO attachment_manifest(path, source) VALUES (?, ?)", ("x.png", "s")),
    ("lint_runs", "INSERT INTO lint_runs(timestamp) VALUES (?)", ("2026-01-01T00:00:00",)),
    ("runs", "INSERT INTO runs(event, run_id, ts) VALUES (?, ?, ?)", ("start", "r1", "2026-01-01T00:00:00")),
    ("memory_states", "INSERT INTO memory_states(node_id, confidence) VALUES (?, ?)", ("n1", 0.7)),
    ("memory_log", "INSERT INTO memory_log(ts, query) VALUES (?, ?)", ("2026-01-01T00:00:00", "q")),
    (REFERENCES_TABLE, "INSERT INTO bib_references(source, ref_id) VALUES (?, ?)", ("s.md", "ref1")),
    ("entity_types", "INSERT INTO entity_types(concept_key, folder_name) VALUES (?, ?)", ("k", "person")),
    ("downloaded", "INSERT INTO downloaded(url, sha256) VALUES (?, ?)", ("https://x", "h")),
    ("organize_reports", "INSERT INTO organize_reports(written_at) VALUES (?)", ("2026-01-01T00:00:00",)),
    ("verification", "INSERT INTO verification(written_at) VALUES (?)", ("2026-01-01T00:00:00",)),
    ("private_pages", "INSERT INTO private_pages(id) VALUES (?)", ("p1",)),
    ("provenance", "INSERT INTO provenance(node_id) VALUES (?)", ("n1",)),
    ("review_state", "INSERT INTO review_state(slug, status) VALUES (?, ?)", ("draft-1", "approved")),
    ("blobs", "INSERT INTO blobs(path, content) VALUES (?, ?)", ("cache/images/a.png", b"\x89PNG")),
)


def test_create_on_missing(tmp_path: Path) -> None:
    db = _db_path(tmp_path)
    assert not db.exists()
    with open_store(db) as conn:
        names = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert db.exists()
    assert row_version(db) == SCHEMA_VERSION
    for table in TABLE_NAMES:
        assert table in names
    assert SCHEMA_VERSION_TABLE in names


def test_wal_is_enabled(tmp_path: Path) -> None:
    with open_store(_db_path(tmp_path)) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


def test_version_mismatch_recreates_clean(tmp_path: Path) -> None:
    db = _db_path(tmp_path)
    with open_store(db) as conn:
        conn.execute("INSERT INTO tier_state(key, stem, source_file, content_hash) VALUES ('s-h', 's', 's.md', 'h')")
        conn.execute(f"UPDATE {SCHEMA_VERSION_TABLE} SET version = 999")

    # Reopening must drop and recreate: the seeded row is gone, version resets.
    with open_store(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM tier_state").fetchone()[0]
        assert count == 0
        assert conn.execute(f"SELECT version FROM {SCHEMA_VERSION_TABLE} WHERE id = 1").fetchone()[0] == SCHEMA_VERSION


def test_additive_schema_adds_missing_table_without_version_bump(tmp_path: Path) -> None:
    """A table added to the DDL after a store exists is created on next open,
    with no version bump and no data loss (the additive path).
    """
    db = _db_path(tmp_path)
    with open_store(db) as conn:
        conn.execute("INSERT INTO tier_state(key, stem, source_file, content_hash) VALUES ('keep-h', 'keep', 's.md', 'h')")
        # Simulate a store created before review_state existed.
        conn.execute("DROP TABLE review_state")

    with open_store(db) as conn:
        names = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        # Missing table is recreated …
        assert "review_state" in names
        # … and existing data survives (no recreate happened).
        assert conn.execute("SELECT COUNT(*) FROM tier_state").fetchone()[0] == 1
        assert conn.execute(f"SELECT version FROM {SCHEMA_VERSION_TABLE} WHERE id = 1").fetchone()[0] == SCHEMA_VERSION


def test_tier_state_is_rekeyed_from_stem_to_identity(tmp_path: Path) -> None:
    """A stem-keyed store is migrated in place: the row keeps its counters."""
    db = _db_path(tmp_path)
    with open_store(db) as conn:
        conn.execute(f"DROP TABLE {TIER_STATE_TABLE}")
        conn.execute(
            f"CREATE TABLE {TIER_STATE_TABLE} (stem TEXT PRIMARY KEY, source_file TEXT NOT NULL, "
            f"content_hash TEXT NOT NULL, current_tier INTEGER NOT NULL DEFAULT 0, "
            f"last_compile TEXT NOT NULL DEFAULT '', search_hits INTEGER NOT NULL DEFAULT 0, "
            f"last_hit TEXT NOT NULL DEFAULT '', ingested_by TEXT NOT NULL DEFAULT '')"
        )
        conn.execute(
            f"INSERT INTO {TIER_STATE_TABLE}(stem, source_file, content_hash, current_tier, search_hits, ingested_by) "
            f"VALUES ('P863E', 'sources/p863-v2.0/P863E.pdf', 'cedd4aee99', 2, 7, 'Jan.Reimes')"
        )

    with open_store(db) as conn:
        row = conn.execute(f"SELECT key, stem, current_tier, search_hits, ingested_by FROM {TIER_STATE_TABLE}").fetchone()
    assert row is not None
    assert row["key"] == "P863E-cedd4aee"
    assert row["stem"] == "P863E"
    assert row["current_tier"] == 2
    assert row["search_hits"] == 7
    assert row["ingested_by"] == "Jan.Reimes"


@pytest.mark.parametrize(("table", "insert_sql", "params"), _INSERT_CASES)
def test_roundtrip_each_table(tmp_path: Path, table: str, insert_sql: str, params: tuple[object, ...]) -> None:
    db = _db_path(tmp_path)
    with open_store(db) as conn:
        conn.execute(insert_sql, params)
    with open_store(db) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1


def test_blob_roundtrip(tmp_path: Path) -> None:
    db = _db_path(tmp_path)
    payload = bytes(range(256))
    with open_store(db) as conn:
        conn.execute(f"INSERT INTO {BLOBS_TABLE}(path, content) VALUES (?, ?)", ("cache/enc/a.utf8", payload))
    with open_store(db) as conn:
        stored = conn.execute(f"SELECT content FROM {BLOBS_TABLE} WHERE path = ?", ("cache/enc/a.utf8",)).fetchone()[0]
    assert bytes(stored) == payload


def test_memory_log_fts_search(tmp_path: Path) -> None:
    db = _db_path(tmp_path)
    with open_store(db) as conn:
        conn.execute(f"INSERT INTO {MEMORY_LOG_TABLE}(ts, query, answer, feedback) VALUES (?, ?, ?, ?)", ("t1", "perceptual audio quality", "POLQA", "good"))
        conn.execute(f"INSERT INTO {MEMORY_LOG_TABLE}(ts, query, answer, feedback) VALUES (?, ?, ?, ?)", ("t2", "vector database", "sqlite-vec", ""))

    with open_store(db) as conn:
        hits = conn.execute(
            f"SELECT rowid FROM {MEMORY_LOG_FTS_TABLE} WHERE {MEMORY_LOG_FTS_TABLE} MATCH ?",
            ("POLQA",),
        ).fetchall()
    assert len(hits) == 1

    # Delete must keep the external-content index in sync (triggers).
    with open_store(db) as conn:
        conn.execute(f"DELETE FROM {MEMORY_LOG_TABLE} WHERE ts = ?", ("t1",))
    with open_store(db) as conn:
        hits = conn.execute(
            f"SELECT rowid FROM {MEMORY_LOG_FTS_TABLE} WHERE {MEMORY_LOG_FTS_TABLE} MATCH ?",
            ("POLQA",),
        ).fetchall()
    assert hits == []


def test_shared_connection_across_threads(tmp_path: Path) -> None:
    db = _db_path(tmp_path)
    conn = connect(db)
    try:
        with open_store(db):
            pass
        conn.execute("INSERT INTO runs(event, run_id, ts) VALUES ('start', 'r1', 't')")
        conn.commit()
        seen: list[int] = []

        def worker() -> None:
            seen.append(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert seen == [1]
    finally:
        conn.close()


def test_obsolete_artifacts_finds_legacy_and_ignores_kept(tmp_path: Path) -> None:
    vault = VaultPaths(tmp_path)
    vault.kompend_dir.mkdir(parents=True)
    vault.dedup_dir.mkdir(parents=True)
    vault.cache_dir.mkdir(parents=True)
    vault.vectors_db_file.write_text("", encoding="utf-8")

    legacy_files = (
        vault.compile_state_file,
        vault.wiki_cache_file,
        vault.compiled_wiki_file,
        vault.memory_store_file,
        vault.memory_log_file,
        vault.references_file,
        vault.assets_file,
        vault.entity_types_file,
        vault.attachments_manifest_file,
        vault.lint_history_file,
        vault.runs_file,
        vault.organize_report_file,
        vault.verification_file,
        vault.private_file,
    )
    for path in legacy_files:
        path.write_text("{}", encoding="utf-8")

    # provenance/ is a directory of per-record files (superseded by the table).
    vault.provenance_dir.mkdir(parents=True, exist_ok=True)

    # Things that must NOT be reported.
    vault.kompend_db_file.write_text("", encoding="utf-8")
    vault.config_file.write_text("", encoding="utf-8")
    (vault.kompend_dir / "rumdl.toml").write_text("", encoding="utf-8")
    vault.memory_db_file.write_text("", encoding="utf-8")

    found = obsolete_artifacts(tmp_path)
    found_set = {path.name for path in found}

    for path in legacy_files:
        assert path in found
    assert vault.dedup_dir in found
    assert vault.cache_dir in found
    assert vault.provenance_dir in found
    assert "kompend.db" not in found_set
    assert "config.toml" not in found_set
    assert "rumdl.toml" not in found_set
    assert "vectors.db" not in found_set
    assert "memory.db" not in found_set


def test_obsolete_artifacts_empty_for_fresh_vault(tmp_path: Path) -> None:
    assert obsolete_artifacts(tmp_path) == []


def test_schema_is_usable_without_fk_violations(tmp_path: Path) -> None:
    # Foreign keys PRAGMA must be on without breaking inserts (no FKs declared).
    with open_store(_db_path(tmp_path)) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert isinstance(conn, sqlite3.Connection)


def test_core_metadata_generates_the_current_ddl() -> None:
    """The Core metadata must produce the schema the store has on disk.

    Textual DDL comparison is impossible: SQLAlchemy renders ``DEFAULT``
    before ``NOT NULL`` and emits table-level ``PRIMARY KEY`` constraints.
    Both schemas are therefore created in scratch in-memory databases and
    compared through ``PRAGMA table_info`` — column name, type, nullability,
    default value and primary-key position — the same trick
    ``schema._expected_columns`` uses to keep the expected column set honest.

    The hand-written statements are frozen verbatim below (they were the
    store's only DDL for years, and were asserted fresh against ``_DDL`` at
    the moment it was retired) so the guard outlives the DDL it protects:
    this test is what proves the generated schema still means what the
    hand-written one meant.
    """
    hand_written = (
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS documents (
            cache_key TEXT PRIMARY KEY,
            source_file TEXT NOT NULL,
            parser_fingerprint TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            author TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL DEFAULT 'en',
            page_count INTEGER NOT NULL DEFAULT 0,
            metadata TEXT NOT NULL DEFAULT '{}',
            elements TEXT NOT NULL DEFAULT '[]'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chunks (
            cache_key TEXT PRIMARY KEY,
            deps TEXT NOT NULL DEFAULT '{}',
            payload TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS extractions (
            layer TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            deps TEXT NOT NULL DEFAULT '{}',
            payload TEXT NOT NULL,
            PRIMARY KEY (layer, cache_key)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            ref_count INTEGER NOT NULL DEFAULT 0,
            data TEXT NOT NULL DEFAULT '{}'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tier_state (
            key TEXT PRIMARY KEY,
            stem TEXT NOT NULL,
            source_file TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            current_tier INTEGER NOT NULL DEFAULT 0,
            last_compile TEXT NOT NULL DEFAULT '',
            search_hits INTEGER NOT NULL DEFAULT 0,
            last_hit TEXT NOT NULL DEFAULT '',
            ingested_by TEXT NOT NULL DEFAULT ''
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS wiki_snapshot (
            kind TEXT PRIMARY KEY,
            nodes TEXT NOT NULL DEFAULT '[]',
            edges TEXT NOT NULL DEFAULT '[]',
            toc TEXT NOT NULL DEFAULT '[]',
            quality_score REAL NOT NULL DEFAULT 0.0,
            iteration INTEGER NOT NULL DEFAULT 0,
            metadata TEXT NOT NULL DEFAULT '{}',
            written_at TEXT NOT NULL DEFAULT ''
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS dedup_fingerprints (
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
        """
        CREATE TABLE IF NOT EXISTS attachment_manifest (
            path TEXT PRIMARY KEY,
            source TEXT NOT NULL DEFAULT '',
            ingestible INTEGER NOT NULL DEFAULT 0,
            embed INTEGER,
            content_digest TEXT NOT NULL DEFAULT '',
            registered_at TEXT NOT NULL DEFAULT '',
            deleted INTEGER NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS lint_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            duration_seconds REAL NOT NULL DEFAULT 0.0,
            options TEXT NOT NULL DEFAULT '{}',
            llm TEXT,
            config_snapshot TEXT NOT NULL DEFAULT '{}',
            wiki_meta TEXT NOT NULL DEFAULT '{}',
            report TEXT NOT NULL DEFAULT '{}'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS runs (
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
        """
        CREATE TABLE IF NOT EXISTS stages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            ts TEXT NOT NULL,
            source TEXT NOT NULL,
            stage TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 1,
            outcome TEXT NOT NULL,
            duration_ms REAL,
            reason_code TEXT,
            detail TEXT,
            parser TEXT,
            parser_version TEXT,
            backend TEXT,
            judge_model TEXT,
            cache_key TEXT,
            sha TEXT,
            size_bytes INTEGER
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_states (
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
        """
        CREATE TABLE IF NOT EXISTS memory_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mem_id TEXT,
            ts TEXT NOT NULL,
            query TEXT NOT NULL DEFAULT '',
            answer TEXT NOT NULL DEFAULT '',
            feedback TEXT NOT NULL DEFAULT '',
            memory_by TEXT NOT NULL DEFAULT ''
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS bib_references (
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
            payload TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (source, ref_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS entity_types (
            concept_key TEXT PRIMARY KEY,
            folder_name TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS downloaded (
            url TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS organize_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            written_at TEXT NOT NULL DEFAULT '',
            dry_run INTEGER NOT NULL DEFAULT 0,
            categories TEXT NOT NULL DEFAULT '[]',
            moves TEXT NOT NULL DEFAULT '[]',
            link_updates TEXT NOT NULL DEFAULT '{}'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS verification (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            written_at TEXT NOT NULL DEFAULT '',
            total INTEGER NOT NULL DEFAULT 0,
            verified INTEGER NOT NULL DEFAULT 0,
            unsupported INTEGER NOT NULL DEFAULT 0,
            average_confidence REAL NOT NULL DEFAULT 0.0,
            claims TEXT NOT NULL DEFAULT '[]'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS private_pages (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            saved_at TEXT NOT NULL DEFAULT ''
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS provenance (
            node_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            citations TEXT NOT NULL DEFAULT '[]'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS blobs (
            path TEXT PRIMARY KEY,
            mime TEXT NOT NULL DEFAULT '',
            size INTEGER NOT NULL DEFAULT 0,
            content BLOB NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS review_state (
            slug TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            reason TEXT NOT NULL DEFAULT '',
            source_node_id TEXT NOT NULL DEFAULT '',
            saved_at TEXT NOT NULL DEFAULT ''
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS vector_manifest (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            embed_version INTEGER NOT NULL DEFAULT 0,
            embedding_model TEXT NOT NULL DEFAULT '',
            backend TEXT NOT NULL DEFAULT '',
            ids TEXT NOT NULL DEFAULT '[]',
            content_hashes TEXT NOT NULL DEFAULT '{}',
            written_at TEXT NOT NULL DEFAULT ''
        )
        """,
    )

    hand = sqlite3.connect(":memory:")
    core = create_engine("sqlite://")
    try:
        for statement in hand_written:
            hand.execute(statement)
        METADATA.create_all(core)
        with core.connect() as conn:
            for table in (*TABLE_NAMES, SCHEMA_VERSION_TABLE):
                expected = [tuple(row)[1:] for row in hand.execute(f"PRAGMA table_info({table})")]
                generated = [tuple(row)[1:] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")]
                assert generated == expected, table
            # Table-level meanings PRAGMA cannot see: AUTOINCREMENT, the
            # single-row CHECK, the FTS5 index and its triggers.
            autoincrement = {row[0] for row in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE '%AUTOINCREMENT%'")}
            assert autoincrement == {"lint_runs", "runs", "stages", "memory_log", "organize_reports", "verification"}
            checked = {row[0] for row in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE '%CHECK (id = 1)%'")}
            assert checked == {SCHEMA_VERSION_TABLE, "vector_manifest"}
            indexes = [row[1] for row in conn.exec_driver_sql("PRAGMA index_list(runs)")]
            assert "idx_runs_run_id" in indexes
            stage_indexes = [row[1] for row in conn.exec_driver_sql("PRAGMA index_list(stages)")]
            assert {"idx_stages_run_id", "idx_stages_stage_outcome"} <= set(stage_indexes)
            fts_columns = [row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({MEMORY_LOG_FTS_TABLE})")]
            assert fts_columns == ["query", "answer", "feedback"]
            triggers = {row[0] for row in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
            assert triggers == {"memory_log_ai", "memory_log_ad", "memory_log_au"}
    finally:
        core.dispose()
        hand.close()


def test_an_existing_store_reopens_without_drift(tmp_path: Path) -> None:
    """Reopening an existing store must not recreate, migrate, or lose rows."""
    db = _db_path(tmp_path)
    with open_store(db) as conn:
        conn.execute("INSERT INTO lint_runs(timestamp) VALUES (?)", ("2026-01-01T00:00:00",))
    with open_store(db) as conn:
        rows = conn.execute("SELECT id, timestamp FROM lint_runs").fetchall()
    assert len(rows) == 1
    assert rows[0]["timestamp"] == "2026-01-01T00:00:00"
    assert row_version(db) == SCHEMA_VERSION


def row_version(db: Path) -> int:
    with open_store(db) as conn:
        row = conn.execute(f"SELECT version FROM {SCHEMA_VERSION_TABLE} WHERE id = 1").fetchone()
    assert row is not None
    return int(row["version"])


def test_drop_all_removes_the_fts_shadow_tables(tmp_path: Path) -> None:
    """A clean cut must remove FTS5 shadow tables, triggers, and every user object."""
    db = _db_path(tmp_path)
    with open_store(db) as conn:
        shadow = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'memory_log_fts_%'")}
    assert shadow, "expected the FTS5 shadow tables to exist"
    conn = connect(db)
    try:
        _drop_all(conn)
        conn.commit()
        remaining = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}
    finally:
        conn.close()
    assert remaining == set()


def _db_path(tmp_path: Path) -> Path:
    return VaultPaths(tmp_path).kompend_db_file
