"""SQLAlchemy Core metadata for the ``.kompend/kompend.db`` store.

The ``Table`` objects below are the schema's declared single source of truth
(the typed-store plan, ``.agents/plans/001-typed-store-and-validated-rows.md``):
the DDL is *generated* from this metadata instead of being written twice — once
as structure here, once as hand-written ``CREATE TABLE`` text in ``schema.py``.
``tests/unit/test_store.py`` proves the generated schema means exactly what the
retired hand-written DDL meant (columns, types, nullability, defaults, primary
keys).

Nullability is copied verbatim, including SQLite's quirk that a bare
``TEXT PRIMARY KEY`` does *not* imply ``NOT NULL`` (``PRAGMA table_info``
reports ``notnull = 0`` for it). SQLAlchemy adds ``NOT NULL`` to primary-key
columns on its own, so the explicit ``nullable=True`` below switches that off
wherever the hand-written DDL declared a bare primary key — a generated schema
that is stricter or looser than the on-disk one is drift, not an improvement.

``AUTOINCREMENT`` tables carry the ``sqlite_autoincrement=True`` table kwarg;
without it SQLAlchemy omits the keyword, which changes SQLite's rowid-reuse
behaviour. The kwarg also moves the primary key inline, so those ``id``
columns keep ``nullable=True`` for the same notnull-0 reason as above.

Defaults are ``server_default`` (SQL rendered into the DDL), never the
Python-side ``default=``: a Python-side default generates no ``DEFAULT``
clause, which is a different schema. ``REAL`` columns use the ``REAL`` type —
``Float`` renders as ``FLOAT``, which reads back differently from
``PRAGMA table_info``.

The FTS5 virtual table, its triggers, and the ``runs`` index are raw DDL
strings attached to the metadata as ``after_create`` events: FTS5/trigger
syntax has no Core object, and ``create_all`` skips indexes of tables it did
not create, so events — which fire on every ``create_all`` — keep the
repair-on-reopen behaviour the hand-written ``IF NOT EXISTS`` DDL had.
"""

from __future__ import annotations

from typing import Final

from sqlalchemy import REAL, CheckConstraint, Column, Integer, LargeBinary, MetaData, PrimaryKeyConstraint, Table, Text, event
from sqlalchemy.schema import DDL
from sqlalchemy.sql import text

#: The store's schema, expressed once. The DDL is generated from this via
#: ``METADATA.create_all(...)`` — never hand-written next to it.
METADATA: Final[MetaData] = MetaData()

#: DDL that has no Core object (FTS5, triggers) or that ``create_all`` cannot
#: repair on an existing store (the runs index). Executed on *every*
#: ``create_all`` via the metadata's ``after_create`` event, in this order —
#: ``IF NOT EXISTS`` keeps it idempotent, exactly like the retired hand-written
#: DDL. The names are frozen by ``tests/unit/test_store.py``.
AFTER_CREATE_DDL: Final[tuple[str, ...]] = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS memory_log_fts USING fts5(
        query, answer, feedback,
        content='memory_log', content_rowid='id', tokenize='unicode61'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memory_log_ai AFTER INSERT ON memory_log BEGIN
        INSERT INTO memory_log_fts(rowid, query, answer, feedback)
        VALUES (new.id, new.query, new.answer, new.feedback);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memory_log_ad AFTER DELETE ON memory_log BEGIN
        INSERT INTO memory_log_fts(memory_log_fts, rowid, query, answer, feedback)
        VALUES ('delete', old.id, old.query, old.answer, old.feedback);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memory_log_au AFTER UPDATE ON memory_log BEGIN
        INSERT INTO memory_log_fts(memory_log_fts, rowid, query, answer, feedback)
        VALUES ('delete', old.id, old.query, old.answer, old.feedback);
        INSERT INTO memory_log_fts(rowid, query, answer, feedback)
        VALUES (new.id, new.query, new.answer, new.feedback);
    END
    """,
    "CREATE INDEX IF NOT EXISTS idx_runs_run_id ON runs(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_stages_run_id ON stages(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_stages_stage_outcome ON stages(stage, outcome)",
)

#: Bookkeeping row. Single row, ``id = 1``.
schema_version: Final[Table] = Table(
    "schema_version",
    METADATA,
    Column("id", Integer, primary_key=True, nullable=True),
    Column("version", Integer, nullable=False),
    Column("created_at", Text, nullable=False),
    CheckConstraint("id = 1"),
)

#: Parsed-document cache. Keyed by the content discriminator so same-stem files
#: with different content never collide.
documents: Final[Table] = Table(
    "documents",
    METADATA,
    Column("cache_key", Text, primary_key=True, nullable=True),
    Column("source_file", Text, nullable=False),
    Column("parser_fingerprint", Text, nullable=False, server_default=text("''")),
    Column("title", Text, nullable=False, server_default=text("''")),
    Column("author", Text, nullable=False, server_default=text("''")),
    Column("language", Text, nullable=False, server_default=text("'en'")),
    Column("page_count", Integer, nullable=False, server_default=text("0")),
    Column("metadata", Text, nullable=False, server_default=text("'{}'")),
    Column("elements", Text, nullable=False, server_default=text("'[]'")),
)

#: Chunking cache (CacheStore "chunk" layer). Payload is the ChunkTree dict.
chunks: Final[Table] = Table(
    "chunks",
    METADATA,
    Column("cache_key", Text, primary_key=True, nullable=True),
    Column("deps", Text, nullable=False, server_default=text("'{}'")),
    Column("payload", Text, nullable=False),
)

#: LLM / resolve / generate / summary cache layers (CacheStore). Layer-scoped
#: because the same dependency key can appear under different layers.
extractions: Final[Table] = Table(
    "extractions",
    METADATA,
    Column("layer", Text, nullable=False),
    Column("cache_key", Text, nullable=False),
    Column("deps", Text, nullable=False, server_default=text("'{}'")),
    Column("payload", Text, nullable=False),
    PrimaryKeyConstraint("layer", "cache_key"),
)

#: Knowledge-graph assets referenced by wiki nodes.
assets: Final[Table] = Table(
    "assets",
    METADATA,
    Column("id", Text, primary_key=True, nullable=True),
    Column("kind", Text, nullable=False, server_default=text("''")),
    Column("source", Text, nullable=False, server_default=text("''")),
    Column("summary", Text, nullable=False, server_default=text("''")),
    Column("ref_count", Integer, nullable=False, server_default=text("0")),
    Column("data", Text, nullable=False, server_default=text("'{}'")),
)

#: Compile-tier state, keyed by ``<stem>-<hash8>``.
tier_state: Final[Table] = Table(
    "tier_state",
    METADATA,
    Column("key", Text, primary_key=True, nullable=True),
    Column("stem", Text, nullable=False),
    Column("source_file", Text, nullable=False),
    Column("content_hash", Text, nullable=False),
    Column("current_tier", Integer, nullable=False, server_default=text("0")),
    Column("last_compile", Text, nullable=False, server_default=text("''")),
    Column("search_hits", Integer, nullable=False, server_default=text("0")),
    Column("last_hit", Text, nullable=False, server_default=text("''")),
    Column("ingested_by", Text, nullable=False, server_default=text("''")),
)

#: last_wiki.json ("last") and compiled_wiki.json ("compiled") share this
#: table, discriminated by kind.
wiki_snapshot: Final[Table] = Table(
    "wiki_snapshot",
    METADATA,
    Column("kind", Text, primary_key=True, nullable=True),
    Column("nodes", Text, nullable=False, server_default=text("'[]'")),
    Column("edges", Text, nullable=False, server_default=text("'[]'")),
    Column("toc", Text, nullable=False, server_default=text("'[]'")),
    Column("quality_score", REAL, nullable=False, server_default=text("0.0")),
    Column("iteration", Integer, nullable=False, server_default=text("0")),
    Column("metadata", Text, nullable=False, server_default=text("'{}'")),
    Column("written_at", Text, nullable=False, server_default=text("''")),
)

#: One row per slug. The dedup store's bookkeeping ({stamp, store_version})
#: is kept under the reserved slug "_meta".
dedup_fingerprints: Final[Table] = Table(
    "dedup_fingerprints",
    METADATA,
    Column("slug", Text, primary_key=True, nullable=True),
    Column("verdict", Text, nullable=False, server_default=text("''")),
    Column("closest_match", Text),
    Column("closest_score", REAL, nullable=False, server_default=text("0.0")),
    Column("closest_method", Text, nullable=False, server_default=text("''")),
    Column("embedding_score", REAL),
    Column("minhash_score", REAL),
    Column("embedding", Text),
    Column("minhash", Text),
)

#: Latest-wins per path; deleted = 1 is a tombstone carrying an explicit user
#: removal. A row with deleted = 0 is read-equivalent to the legacy
#: append-only JSONL. See attachment_manifest semantics.
attachment_manifest: Final[Table] = Table(
    "attachment_manifest",
    METADATA,
    Column("path", Text, primary_key=True, nullable=True),
    Column("source", Text, nullable=False, server_default=text("''")),
    Column("ingestible", Integer, nullable=False, server_default=text("0")),
    Column("embed", Integer),
    Column("content_digest", Text, nullable=False, server_default=text("''")),
    Column("registered_at", Text, nullable=False, server_default=text("''")),
    Column("deleted", Integer, nullable=False, server_default=text("0")),
)

#: Lint history, one row per run.
lint_runs: Final[Table] = Table(
    "lint_runs",
    METADATA,
    Column("id", Integer, primary_key=True, nullable=True),
    Column("timestamp", Text, nullable=False),
    Column("duration_seconds", REAL, nullable=False, server_default=text("0.0")),
    Column("options", Text, nullable=False, server_default=text("'{}'")),
    Column("llm", Text),
    Column("config_snapshot", Text, nullable=False, server_default=text("'{}'")),
    Column("wiki_meta", Text, nullable=False, server_default=text("'{}'")),
    Column("report", Text, nullable=False, server_default=text("'{}'")),
    sqlite_autoincrement=True,
)

#: CLI/command event log.
runs: Final[Table] = Table(
    "runs",
    METADATA,
    Column("id", Integer, primary_key=True, nullable=True),
    Column("event", Text, nullable=False),
    Column("run_id", Text, nullable=False),
    Column("cmd", Text, nullable=False, server_default=text("''")),
    Column("ts", Text, nullable=False),
    Column("status", Text),
    Column("duration_s", REAL),
    Column("exit", Integer),
    Column("error", Text),
    sqlite_autoincrement=True,
)

#: Per-item stage ledger: one row per (run, source, stage, attempt) — the
#: observability spine (``knox-ty8f.2``). Facts columns are NULL when the
#: stage did not have that fact in hand; ``outcome``/``reason_code`` carry
#: the :class:`~kompend.core.types.StageOutcome` / :class:`~kompend.core.types.ReasonCode`
#: values verbatim. Rows are appended by
#: :class:`kompend.core.reporting.ledger.StageLedger` and pruned to a soft cap.
stages: Final[Table] = Table(
    "stages",
    METADATA,
    Column("id", Integer, primary_key=True, nullable=True),
    Column("run_id", Text, nullable=True),
    Column("ts", Text, nullable=False),
    Column("source", Text, nullable=False),
    Column("stage", Text, nullable=False),
    Column("attempt", Integer, nullable=False, server_default=text("1")),
    Column("outcome", Text, nullable=False),
    Column("duration_ms", REAL),
    Column("reason_code", Text),
    Column("detail", Text),
    Column("parser", Text),
    Column("parser_version", Text),
    Column("backend", Text),
    Column("judge_model", Text),
    Column("cache_key", Text),
    Column("sha", Text),
    Column("size_bytes", Integer),
    sqlite_autoincrement=True,
)

#: Memory node states.
memory_states: Final[Table] = Table(
    "memory_states",
    METADATA,
    Column("node_id", Text, primary_key=True, nullable=True),
    Column("title", Text, nullable=False, server_default=text("''")),
    Column("confidence", REAL, nullable=False, server_default=text("0.5")),
    Column("tier", Text, nullable=False, server_default=text("'working'")),
    Column("birth_at", Text, nullable=False, server_default=text("''")),
    Column("review_at", Text, nullable=False, server_default=text("''")),
    Column("review_count", Integer, nullable=False, server_default=text("0")),
    Column("superseded_by", Text),
)

#: Chronological query memory.
memory_log: Final[Table] = Table(
    "memory_log",
    METADATA,
    Column("id", Integer, primary_key=True, nullable=True),
    Column("mem_id", Text),
    Column("ts", Text, nullable=False),
    Column("query", Text, nullable=False, server_default=text("''")),
    Column("answer", Text, nullable=False, server_default=text("''")),
    Column("feedback", Text, nullable=False, server_default=text("''")),
    Column("memory_by", Text, nullable=False, server_default=text("''")),
    sqlite_autoincrement=True,
)

#: Bibliographic references. ``references`` is a SQLite reserved keyword, so
#: the table is named ``bib_references``.
bib_references: Final[Table] = Table(
    "bib_references",
    METADATA,
    Column("source", Text, nullable=False),
    Column("ref_id", Text, nullable=False),
    Column("type", Text, nullable=False, server_default=text("''")),
    Column("title", Text, nullable=False, server_default=text("''")),
    Column("authors", Text, nullable=False, server_default=text("'[]'")),
    Column("issued", Text, nullable=False, server_default=text("''")),
    Column("container_title", Text, nullable=False, server_default=text("''")),
    Column("doi", Text, nullable=False, server_default=text("''")),
    Column("url", Text, nullable=False, server_default=text("''")),
    Column("raw", Text, nullable=False, server_default=text("''")),
    Column("citing_source", Text, nullable=False, server_default=text("''")),
    Column("in_text_keys", Text, nullable=False, server_default=text("'[]'")),
    Column("payload", Text, nullable=False, server_default=text("'{}'")),
    PrimaryKeyConstraint("source", "ref_id"),
)

#: Concept-key → folder mapping for entity pages.
entity_types: Final[Table] = Table(
    "entity_types",
    METADATA,
    Column("concept_key", Text, primary_key=True, nullable=True),
    Column("folder_name", Text, nullable=False),
)

#: URL download dedup: flat url -> content hash.
downloaded: Final[Table] = Table(
    "downloaded",
    METADATA,
    Column("url", Text, primary_key=True, nullable=True),
    Column("sha256", Text, nullable=False),
)

#: ``kompend organize`` report. Tree/move lists are nested, so they stay JSON;
#: dry_run and time are queryable scalars.
organize_reports: Final[Table] = Table(
    "organize_reports",
    METADATA,
    Column("id", Integer, primary_key=True, nullable=True),
    Column("written_at", Text, nullable=False, server_default=text("''")),
    Column("dry_run", Integer, nullable=False, server_default=text("0")),
    Column("categories", Text, nullable=False, server_default=text("'[]'")),
    Column("moves", Text, nullable=False, server_default=text("'[]'")),
    Column("link_updates", Text, nullable=False, server_default=text("'{}'")),
    sqlite_autoincrement=True,
)

#: Claim/citation verification report. Summary counts are scalars; the
#: per-claim list is JSON.
verification: Final[Table] = Table(
    "verification",
    METADATA,
    Column("id", Integer, primary_key=True, nullable=True),
    Column("written_at", Text, nullable=False, server_default=text("''")),
    Column("total", Integer, nullable=False, server_default=text("0")),
    Column("verified", Integer, nullable=False, server_default=text("0")),
    Column("unsupported", Integer, nullable=False, server_default=text("0")),
    Column("average_confidence", REAL, nullable=False, server_default=text("0.0")),
    Column("claims", Text, nullable=False, server_default=text("'[]'")),
    sqlite_autoincrement=True,
)

#: Private-page index: one row per withheld page.
private_pages: Final[Table] = Table(
    "private_pages",
    METADATA,
    Column("id", Text, primary_key=True, nullable=True),
    Column("title", Text, nullable=False, server_default=text("''")),
    Column("saved_at", Text, nullable=False, server_default=text("''")),
)

#: Per-node provenance: one row per node, citations as a JSON list.
provenance: Final[Table] = Table(
    "provenance",
    METADATA,
    Column("node_id", Text, primary_key=True, nullable=True),
    Column("title", Text, nullable=False, server_default=text("''")),
    Column("citations", Text, nullable=False, server_default=text("'[]'")),
)

#: Extracted images and re-encoded .utf8 copies. ``path`` is relative to
#: .kompend/ so the same row is valid across a vault move.
blobs: Final[Table] = Table(
    "blobs",
    METADATA,
    Column("path", Text, primary_key=True, nullable=True),
    Column("mime", Text, nullable=False, server_default=text("''")),
    Column("size", Integer, nullable=False, server_default=text("0")),
    Column("content", LargeBinary, nullable=False),
)

#: Review-queue state. The human-editable queue markdown stays on disk; only
#: the approve/reject decisions live here.
review_state: Final[Table] = Table(
    "review_state",
    METADATA,
    Column("slug", Text, primary_key=True, nullable=True),
    Column("status", Text, nullable=False, server_default=text("''")),
    Column("title", Text, nullable=False, server_default=text("''")),
    Column("reason", Text, nullable=False, server_default=text("''")),
    Column("source_node_id", Text, nullable=False, server_default=text("''")),
    Column("saved_at", Text, nullable=False, server_default=text("''")),
)

#: Vector-branch manifest: the indexed id set, their content hashes and the
#: embed-text version of the last build, used to detect a stale vector store.
#: Single row, id = 1, like schema_version. ``backend`` names the store that
#: wrote the vectors ("sqlite-vec"): vectors of one backend are not readable
#: by another, so a manifest written by a different backend must not validate
#: the current store.
vector_manifest: Final[Table] = Table(
    "vector_manifest",
    METADATA,
    Column("id", Integer, primary_key=True, nullable=True),
    Column("embed_version", Integer, nullable=False, server_default=text("0")),
    Column("embedding_model", Text, nullable=False, server_default=text("''")),
    Column("backend", Text, nullable=False, server_default=text("''")),
    Column("ids", Text, nullable=False, server_default=text("'[]'")),
    Column("content_hashes", Text, nullable=False, server_default=text("'{}'")),
    Column("written_at", Text, nullable=False, server_default=text("''")),
    CheckConstraint("id = 1"),
)

for _ddl in AFTER_CREATE_DDL:
    event.listen(METADATA, "after_create", DDL(_ddl))
