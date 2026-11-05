"""Query memory store: sqlite-vec for semantic retrieval, ``kompend.db`` for
the chronological log.

The chronological/feedback log lives in the ``memory_log`` table of
``.kompend/kompend.db``. The sqlite-vec index (``memory.db``) remains the
regenerate-from-log embedding branch.
"""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants;
# no runtime value ever enters an identifier or statement.

from __future__ import annotations

import contextlib
import shutil
import sqlite3
from enum import StrEnum
from pathlib import Path
from typing import Final, NotRequired, TypedDict

import sqlite_vec
from sqlite_vec_client import SQLiteVecClient

from kompend.core.config import DisplayConfig
from kompend.core.paths import VaultPaths
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import MEMORY_LOG_TABLE
from kompend.core.utils.time import iso_local
from kompend.logging import get_logger
from kompend.rag.embedding_base import EmbeddingProvider
from kompend.rag.embedding_pydantic import make_provider

#: The vec0 table holding the memory entries and their query embeddings.
_MEMORY_TABLE: Final[str] = "memory_entries"

#: The store's own identity — a dimension or backend change forces a rebuild.
_MEMORY_META: Final[str] = "memory_meta"

#: Identifies the backend in the store's meta table.
_MEMORY_BACKEND: Final[str] = "sqlite-vec"

logger = get_logger(__name__)


class MemoryField(StrEnum):
    """Single source of truth for memory store field and vector names.

    Covers the store's columns, the vector key, the JSONL log keys, and the
    search-result dict keys — all are the same contract.
    """

    QUERY = "query"
    ANSWER = "answer"
    FEEDBACK = "feedback"
    TIMESTAMP = "timestamp"
    MEMORY_BY = "memory_by"
    SCORE = "score"
    QUERY_EMBEDDING = "query_embedding"


class MemoryEntry(TypedDict):
    """Schema for one memory entry (search result / JSONL log line)."""

    query: str
    answer: str
    feedback: str
    timestamp: str
    memory_by: NotRequired[str]
    score: NotRequired[float]


class MemoryStore:
    """Stores and retrieves query/answer/feedback triples for agent memory.

    Uses a dual-storage approach:
    - sqlite-vec for semantic search of past queries.
    - The vault store's ``memory_log`` table for strict chronological retrieval
      (used by synthesis).
    """

    def __init__(
        self,
        base_dir: str | Path,
        embedding_model: str | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        display: DisplayConfig | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        _paths = VaultPaths(self.base_dir)
        self.db_file = _paths.memory_db_file
        self._log_db_path = vault_db_path(self.base_dir)
        self._embedding_model = embedding_model
        self._provider: EmbeddingProvider | None = embedding_provider
        self._conn: sqlite3.Connection | None = None
        self._display = display if display is not None else DisplayConfig()

    def add_entry(
        self,
        query: str,
        answer: str,
        feedback: str | None = None,
        memory_by: str = "",
    ) -> None:
        """Add a new query/answer/feedback triple to the memory store.

        *memory_by* records the acting OS user (auditing); empty string
        records no identity.
        """
        provider = self._ensure_store()
        timestamp = iso_local(timespec="microseconds")

        # Embed before writing so the vector dimension is resolved up front
        embedding = provider.embed(query)
        mem_id = f"mem_{timestamp.replace(':', '').replace('.', '')}"
        conn = self._require_conn()
        conn.execute(
            f"INSERT INTO {_MEMORY_TABLE} (id, embedding, query, answer, feedback, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (
                mem_id,
                sqlite_vec.serialize_float32(embedding),
                query,
                answer,
                feedback or "",
                timestamp,
            ),
        )
        conn.commit()

        # Chronological/feedback log for synthesis lives in the vault store.
        with open_store(self._log_db_path) as conn:
            conn.execute(
                f"INSERT INTO {MEMORY_LOG_TABLE}(mem_id, ts, query, answer, feedback, memory_by) VALUES (?, ?, ?, ?, ?, ?)",
                (mem_id, timestamp, query, answer, feedback or "", memory_by),
            )

        query_log_limit = self._display.memory_query_chars
        logger.info("Added memory entry for query: %s", query[:query_log_limit])

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """Search memory entries semantically by query.

        The ``score`` is the cosine similarity (higher is better), the same
        convention the wiki store's hits use.
        """
        provider = self._ensure_store()
        conn = self._require_conn()

        embedding = provider.embed(query)
        rows = conn.execute(
            f"""
            SELECT query, answer, feedback, timestamp, distance
            FROM {_MEMORY_TABLE}
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (sqlite_vec.serialize_float32(embedding), top_k),
        ).fetchall()

        entries: list[MemoryEntry] = []
        for row in rows:
            entries.append(
                {
                    "query": str(row["query"] or ""),
                    "answer": str(row["answer"] or ""),
                    "feedback": str(row["feedback"] or ""),
                    "timestamp": str(row["timestamp"] or ""),
                    "score": 1.0 - float(row["distance"] or 0.0),
                }
            )  # MemoryEntry
        return entries

    def get_recent_entries(self, limit: int = 20) -> list[MemoryEntry]:
        """Get the most recent memory entries chronologically (for synthesis)."""
        if not self._log_db_path.exists():
            return []
        with open_store(self._log_db_path) as conn:
            rows = conn.execute(
                f"SELECT ts, query, answer, feedback, memory_by FROM {MEMORY_LOG_TABLE} ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "timestamp": row["ts"],
                "query": row["query"],
                "answer": row["answer"],
                "feedback": row["feedback"],
                "memory_by": row["memory_by"],
            }
            for row in rows
        ]

    def close(self) -> None:
        """Close the index file and release its handle.

        SQLite keeps the file open until the connection is closed, so a later
        rebuild of the same path races the open handle.
        """
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None

    def _ensure_store(self) -> EmbeddingProvider:
        """Open (or create) the sqlite-vec index for memory entries."""
        if self._conn is not None:
            if self._provider is None:
                raise RuntimeError("Memory store exists but embedding provider is unset — internal invariant violated")
            return self._provider

        provider = self._provider
        if provider is None:
            if not self._embedding_model:
                raise RuntimeError("No embedding model configured for memory store")
            provider = make_provider(self._embedding_model)
            self._provider = provider

        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self._drop_legacy_directory()
        conn = SQLiteVecClient.create_connection(str(self.db_file))
        try:
            if not _meta_matches(conn, provider.dim):
                _create_index(conn, provider.dim)
        except sqlite3.Error as exc:
            logger.warning("Memory store at %s is unreadable (%s); rebuilding", self.db_file, exc)
            conn.close()
            self._drop_store_file()
            conn = SQLiteVecClient.create_connection(str(self.db_file))
            _create_index(conn, provider.dim)
        self._conn = conn
        return provider

    def _drop_legacy_directory(self) -> None:
        """Remove the legacy zvec collection directory at this path.

        The path used to be a directory; it is a file now. The index is derived
        from the ``memory_log`` table, so the old directory is dead weight.
        """
        if self.db_file.is_dir():
            logger.info("Legacy memory store directory at %s — replacing it with the index file", self.db_file)
            shutil.rmtree(self.db_file, ignore_errors=True)

    def _drop_store_file(self) -> None:
        """Remove the index file and its write-ahead-log siblings."""
        self.db_file.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(f"{self.db_file}{suffix}").unlink(missing_ok=True)

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Memory store is not initialized")
        return self._conn


def _create_index(conn: sqlite3.Connection, dim: int) -> None:
    """Create the index table for *dim* and record the store's identity."""
    conn.execute(f"DROP TABLE IF EXISTS {_MEMORY_TABLE}")
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE {_MEMORY_TABLE} USING vec0(
            id        TEXT PRIMARY KEY,
            embedding FLOAT[{dim}] distance_metric=cosine,
            query     TEXT,
            answer    TEXT,
            feedback  TEXT,
            timestamp TEXT
        )
        """
    )
    conn.execute(f"CREATE TABLE IF NOT EXISTS {_MEMORY_META} (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(f"DELETE FROM {_MEMORY_META}")
    conn.executemany(
        f"INSERT INTO {_MEMORY_META} (key, value) VALUES (?, ?)",
        [("backend", _MEMORY_BACKEND), ("dim", str(dim))],
    )
    conn.commit()


def _meta_matches(conn: sqlite3.Connection, dim: int) -> bool:
    """True when the index file holds this backend and dimension."""
    try:
        rows = dict(conn.execute(f"SELECT key, value FROM {_MEMORY_META}").fetchall())
    except sqlite3.Error:
        return False
    return rows.get("backend") == _MEMORY_BACKEND and rows.get("dim") == str(dim)
