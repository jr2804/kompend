"""Semantic cache for query results.

Stores query embeddings and their corresponding answers. On repeat queries
with similar meaning (cosine similarity above threshold), returns the cached
answer directly — skipping retrieval and LLM synthesis entirely.

Config keys (under ``[search.cache]``):

- ``enabled`` — enable/disable (default: ``true``)
- ``threshold`` — cosine similarity threshold (default: ``0.92``)
- ``max_entries`` — max cached entries (default: ``500``)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knox.logging import get_logger
from knox.rag.zvec_store import EmbeddingProvider, _make_provider

logger = get_logger(__name__)

# ── Data model ────────────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    """A single cached query/answer pair."""

    query: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    embedding: list[float] | None = None
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 1


# ── Semantic cache ─────────────────────────────────────────────────────────


class SemanticCache:
    """Cosine-similarity cache for query results.

    Stores query embeddings and their answers. On lookup, embeds the
    incoming query and returns the cached answer if the cosine similarity
    to any stored query exceeds the configured threshold.

    Persisted to a JSON file on disk for durability across sessions.
    """

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        threshold: float = 0.92,
        max_entries: int = 500,
        embedding_model: str | None = None,
    ) -> None:
        self._threshold = threshold
        self._max_entries = max_entries
        self._entries: list[CacheEntry] = []
        self._embedding_provider: EmbeddingProvider | None = None
        self._cache_file: Path | None = None

        if embedding_model:
            try:
                self._embedding_provider = _make_provider(embedding_model)
            except Exception as exc:
                logger.warning(
                    "Semantic cache: could not init embedding provider %s: %s",
                    embedding_model,
                    exc,
                )

        if cache_dir is not None:
            self._cache_file = cache_dir / "semantic_cache.json"
            self._load()

    # ── Public API ────────────────────────────────────────────────────────

    def lookup(self, query: str) -> CacheEntry | None:
        """Look up a query in the cache.

        Returns the cached entry if a semantically similar query exists,
        or ``None`` on cache miss.
        """
        if not self._embedding_provider:
            return None

        query_emb = self._embedding_provider.embed(query)
        best_entry: CacheEntry | None = None
        best_sim: float = -1.0

        for entry in self._entries:
            if entry.embedding is None:
                continue
            sim = _cosine_similarity(query_emb, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_sim >= self._threshold and best_entry is not None:
            best_entry.hit_count += 1
            logger.debug("Semantic cache HIT (sim=%.4f, threshold=%.4f)", best_sim, self._threshold)
            return best_entry

        logger.debug("Semantic cache MISS (best=%.4f, threshold=%.4f)", best_sim, self._threshold)
        return None

    def store(
        self,
        query: str,
        answer: str,
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        """Store a query/answer pair in the cache."""
        if not self._embedding_provider:
            return

        embedding = self._embedding_provider.embed(query)

        # Evict oldest if at capacity
        if len(self._entries) >= self._max_entries:
            self._entries.sort(key=lambda e: e.timestamp)
            self._entries.pop(0)

        self._entries.append(
            CacheEntry(
                query=query,
                answer=answer,
                citations=citations or [],
                embedding=embedding,
            )
        )
        self._save()

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()
        self._save()

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._entries)

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._cache_file is None or not self._cache_file.exists():
            return
        try:
            raw = json.loads(self._cache_file.read_text(encoding="utf-8"))
            self._entries = [_deserialize_entry(e) for e in raw]
            logger.debug("Loaded %d entries from semantic cache", len(self._entries))
        except Exception as exc:
            logger.warning("Failed to load semantic cache: %s", exc)

    def _save(self) -> None:
        if self._cache_file is None:
            return
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            raw = [_serialize_entry(e) for e in self._entries]
            self._cache_file.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save semantic cache: %s", exc)


# ── Helpers ───────────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(ai * bi for ai, bi in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _serialize_entry(entry: CacheEntry) -> dict[str, Any]:
    return {
        "query": entry.query,
        "answer": entry.answer,
        "citations": entry.citations,
        "embedding": entry.embedding,
        "timestamp": entry.timestamp,
        "hit_count": entry.hit_count,
    }


def _deserialize_entry(raw: dict[str, Any]) -> CacheEntry:
    return CacheEntry(
        query=raw["query"],
        answer=raw.get("answer", ""),
        citations=raw.get("citations", []),
        embedding=raw.get("embedding"),
        timestamp=raw.get("timestamp", 0.0),
        hit_count=raw.get("hit_count", 1),
    )
