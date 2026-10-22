"""Query memory store: zvec for semantic retrieval, ``kompend.db`` for the
chronological log.

The chronological/feedback log lives in the ``memory_log`` table of
``.kompend/kompend.db``; the legacy ``memory.jsonl`` is no longer written or read.
The zvec collection (``memory.db``) remains the regenerate-from-log embedding
branch.
"""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants;
# no runtime value ever enters an identifier or statement.

from __future__ import annotations

import contextlib
import shutil
from enum import StrEnum
from pathlib import Path
from typing import NotRequired, TypedDict

import zvec as _zvec

from kompend.core.config import DisplayConfig
from kompend.core.paths import VaultPaths
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import MEMORY_LOG_TABLE
from kompend.core.utils.time import iso_local
from kompend.logging import get_logger
from kompend.rag.zvec_store import EmbeddingProvider, make_provider

_MEMORY_COLLECTION_NAME = "kompend_memory"

logger = get_logger(__name__)


class MemoryField(StrEnum):
    """Single source of truth for memory store field and vector names.

    Covers zvec schema fields, the vector key, the JSONL log keys, and the
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
    - zvec for semantic search of past queries.
    - A JSONL sidecar file for strict chronological retrieval (used by synthesis).
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
        self._collection: _zvec.Collection | None = None
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
        provider = self._ensure_collection()
        if self._collection is None:
            raise RuntimeError("Memory collection is not initialized")
        timestamp = iso_local(timespec="microseconds")

        # Trigger embedding to ensure dim is resolved if using LiteLLM
        embedding = provider.embed(query)
        mem_id = f"mem_{timestamp.replace(':', '').replace('.', '')}"
        doc = _zvec.Doc(
            id=mem_id,
            vectors={str(MemoryField.QUERY_EMBEDDING): embedding},
            fields={
                str(MemoryField.QUERY): query,
                str(MemoryField.ANSWER): answer,
                str(MemoryField.FEEDBACK): feedback or "",
                str(MemoryField.TIMESTAMP): timestamp,
            },
        )
        self._collection.insert([doc])

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
        """Search memory entries semantically by query."""
        provider = self._ensure_collection()
        if self._collection is None:
            raise RuntimeError("Memory collection is not initialized")

        embedding = provider.embed(query)
        results = self._collection.query(
            _zvec.Query(MemoryField.QUERY_EMBEDDING, vector=embedding),
            topk=top_k,
        )

        entries: list[MemoryEntry] = []
        for res in results:
            fields = res.fields if isinstance(res.fields, dict) else {}
            entries.append(
                {
                    "query": str(fields.get(MemoryField.QUERY, "")),
                    "answer": str(fields.get(MemoryField.ANSWER, "")),
                    "feedback": str(fields.get(MemoryField.FEEDBACK, "")),
                    "timestamp": str(fields.get(MemoryField.TIMESTAMP, "")),
                    "score": res.score if res.score is not None else 0.0,
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
        """Close the underlying collection and release its file handles.

        Dropping the reference alone is not enough: zvec holds a RocksDB ``LOCK``
        (and other native handles) until the collection is closed, so a later
        rmtree or reopen of the same path races the GC and can fail on Windows.
        """
        if self._collection is not None:
            with contextlib.suppress(Exception):
                self._collection.close()
            self._collection = None

    def _ensure_collection(self) -> EmbeddingProvider:
        """Initialize the zvec collection for memory entries."""
        if self._collection is not None:
            if self._provider is None:
                raise RuntimeError("Memory collection exists but embedding provider is unset — internal invariant violated")
            return self._provider

        provider = self._provider
        if provider is None:
            if not self._embedding_model:
                raise RuntimeError("No embedding model configured for memory store")
            provider = make_provider(self._embedding_model)
            self._provider = provider

        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        schema = _zvec.CollectionSchema(
            name=_MEMORY_COLLECTION_NAME,
            fields=[
                _zvec.FieldSchema(MemoryField.QUERY, _zvec.DataType.STRING),
                _zvec.FieldSchema(MemoryField.ANSWER, _zvec.DataType.STRING),
                _zvec.FieldSchema(MemoryField.FEEDBACK, _zvec.DataType.STRING),
                _zvec.FieldSchema(MemoryField.TIMESTAMP, _zvec.DataType.STRING),
            ],
            vectors=_zvec.VectorSchema(
                MemoryField.QUERY_EMBEDDING,
                _zvec.DataType.VECTOR_FP32,
                provider.dim,
            ),
        )
        # Existing store: reopen instead of recreating. ``create_and_open``
        # raises when the path already exists, which previously broke every
        # ``kompend memory add`` after the first. Rebuild when the stored vector
        # dimension no longer matches the current embedding model.
        if self.db_file.exists():
            try:
                collection = _zvec.open(path=str(self.db_file))
                stored_dim = collection.schema.vectors[0].dimension
                if stored_dim != provider.dim:
                    logger.warning(
                        "Memory store at %s has dimension %d but the embedding model produces %d; rebuilding",
                        self.db_file,
                        stored_dim,
                        provider.dim,
                    )
                    collection.destroy()
                    collection = _zvec.create_and_open(path=str(self.db_file), schema=schema)
                self._collection = collection
            except Exception as exc:
                logger.warning("Memory store at %s is unreadable (%s); rebuilding", self.db_file, exc)
                shutil.rmtree(self.db_file, ignore_errors=True)
                self._collection = _zvec.create_and_open(path=str(self.db_file), schema=schema)
            return provider
        self._collection = _zvec.create_and_open(
            path=str(self.db_file),
            schema=schema,
        )
        return provider
