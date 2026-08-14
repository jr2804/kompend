"""Lightweight query memory store using zvec for semantic retrieval and JSONL
for chronological logs.
"""

from __future__ import annotations

import json
import shutil
from enum import StrEnum
from pathlib import Path
from typing import NotRequired, TypedDict

import zvec as _zvec

from knox.core.config import DEFAULTS
from knox.core.config.keys import DISPLAY_MEMORY_QUERY_CHARS
from knox.core.paths import VaultPaths
from knox.core.utils.time import iso_local
from knox.logging import get_logger
from knox.rag.zvec_store import EmbeddingProvider, make_provider

_MEMORY_COLLECTION_NAME = "knox_memory"

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
    ) -> None:
        self.base_dir = Path(base_dir)
        _paths = VaultPaths(self.base_dir)
        self.db_file = _paths.memory_db_file
        self.log_file = _paths.memory_log_file
        self._embedding_model = embedding_model
        self._provider: EmbeddingProvider | None = embedding_provider
        self._collection: _zvec.Collection | None = None

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
        self._ensure_collection()
        if self._collection is None:
            raise RuntimeError("Memory collection is not initialized")
        timestamp = iso_local(timespec="microseconds")

        # Trigger embedding to ensure dim is resolved if using LiteLLM
        embedding = self._provider.embed(query)  # ty: ignore

        doc = _zvec.Doc(
            id=f"mem_{timestamp.replace(':', '').replace('.', '')}",
            vectors={str(MemoryField.QUERY_EMBEDDING): embedding},
            fields={
                str(MemoryField.QUERY): query,
                str(MemoryField.ANSWER): answer,
                str(MemoryField.FEEDBACK): feedback or "",
                str(MemoryField.TIMESTAMP): timestamp,
            },
        )
        self._collection.insert([doc])

        # Append to chronological JSONL log for synthesis
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        str(MemoryField.TIMESTAMP): timestamp,
                        str(MemoryField.QUERY): query,
                        str(MemoryField.ANSWER): answer,
                        str(MemoryField.FEEDBACK): feedback or "",
                        str(MemoryField.MEMORY_BY): memory_by,
                    }
                )
                + "\n"
            )

        query_log_limit = DEFAULTS.get(DISPLAY_MEMORY_QUERY_CHARS, 50)
        logger.info("Added memory entry for query: %s", query[:query_log_limit])

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """Search memory entries semantically by query."""
        self._ensure_collection()
        if self._collection is None:
            raise RuntimeError("Memory collection is not initialized")

        embedding = self._provider.embed(query)  # ty: ignore
        results = self._collection.query(
            _zvec.VectorQuery(MemoryField.QUERY_EMBEDDING, vector=embedding),
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
        if not self.log_file.exists():
            return []

        entries = []
        with open(self.log_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        return list(reversed(entries[-limit:]))

    def close(self) -> None:
        """Close the underlying collection."""
        if self._collection is not None:
            self._collection = None

    def _ensure_collection(self) -> None:
        """Initialize the zvec collection and JSONL file for memory entries."""
        if self._collection is not None:
            return

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
        # ``knox memory add`` after the first. Rebuild when the stored vector
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
            return
        self._collection = _zvec.create_and_open(
            path=str(self.db_file),
            schema=schema,
        )
