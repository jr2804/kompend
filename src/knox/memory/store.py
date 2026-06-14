"""Lightweight query memory store using zvec for semantic retrieval and JSONL for chronological logs."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import zvec as _zvec

from knox.rag.zvec_store import EmbeddingProvider, HashEmbeddingProvider, _make_provider

logger = logging.getLogger(__name__)


class MemoryStore:
    """Stores and retrieves query/answer/feedback triples for agent memory.

    Uses a dual-storage approach:
    - zvec for semantic search of past queries.
    - A JSONL sidecar file for strict chronological retrieval (used by synthesis).
    """

    def __init__(
        self,
        base_dir: str | Path,
        embedding_model: str = "hash",
        embedding_dim: int = 384,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / ".knox" / "memory.db"
        self.log_path = self.base_dir / ".knox" / "memory.jsonl"
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._provider: EmbeddingProvider = HashEmbeddingProvider(dim=embedding_dim)
        self._collection: _zvec.Collection | None = None

    def add_entry(
        self,
        query: str,
        answer: str,
        feedback: str | None = None,
    ) -> None:
        """Add a new query/answer/feedback triple to the memory store."""
        self._ensure_collection()
        if self._collection is None:
            raise RuntimeError("Memory collection is not initialized")
        timestamp = datetime.now(UTC).isoformat()

        # Trigger embedding to ensure dim is resolved if using LiteLLM
        embedding = self._provider.embed(query)

        doc = _zvec.Doc(
            id=f"mem_{timestamp.replace(':', '').replace('.', '')}",
            vectors={"query_embedding": embedding},
            fields={
                "query": query,
                "answer": answer,
                "feedback": feedback or "",
                "timestamp": timestamp,
            },
        )
        self._collection.insert([doc])

        # Append to chronological JSONL log for synthesis
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "query": query,
                        "answer": answer,
                        "feedback": feedback or "",
                    }
                )
                + "\n"
            )

        logger.info("Added memory entry for query: %s", query[:50])

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search memory entries semantically by query."""
        self._ensure_collection()
        if self._collection is None:
            raise RuntimeError("Memory collection is not initialized")

        embedding = self._provider.embed(query)
        results = self._collection.query(
            _zvec.VectorQuery("query_embedding", vector=embedding),
            topk=top_k,
        )

        entries = []
        for res in results:
            fields = res.fields if isinstance(res.fields, dict) else {}
            entries.append(
                {
                    "query": fields.get("query", ""),
                    "answer": fields.get("answer", ""),
                    "feedback": fields.get("feedback", ""),
                    "timestamp": fields.get("timestamp", ""),
                    "score": res.score if res.score is not None else 0.0,
                }
            )
        return entries

    def get_recent_entries(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get the most recent memory entries chronologically (for synthesis)."""
        if not self.log_path.exists():
            return []

        entries = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        # Return the last 'limit' entries, newest first
        return list(reversed(entries[-limit:]))

    def close(self) -> None:
        """Close the underlying collection."""
        if self._collection is not None:
            self._collection = None

    def _ensure_collection(self) -> None:
        """Initialize the zvec collection and JSONL file for memory entries."""
        if self._collection is not None:
            return

        if self._embedding_model != "hash" and isinstance(self._provider, HashEmbeddingProvider):
            self._provider = _make_provider(self._embedding_model, dim=self._embedding_dim)
            self._embedding_dim = self._provider.dim

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        schema = _zvec.CollectionSchema(
            name="knox_memory",
            fields=[
                _zvec.FieldSchema("query", _zvec.DataType.STRING),
                _zvec.FieldSchema("answer", _zvec.DataType.STRING),
                _zvec.FieldSchema("feedback", _zvec.DataType.STRING),
                _zvec.FieldSchema("timestamp", _zvec.DataType.STRING),
            ],
            vectors=_zvec.VectorSchema(
                "query_embedding",
                _zvec.DataType.VECTOR_FP32,
                self._embedding_dim,
            ),
        )
        self._collection = _zvec.create_and_open(
            path=str(self.db_path),
            schema=schema,
        )
