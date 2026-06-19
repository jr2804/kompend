"""Embedding-based content similarity matcher.

Uses sentence-transformers (or a litellm-compatible embedding provider)
to compute document vectors, then compares via cosine similarity.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from knox.preprocessing.dedup.base import ContentMatcher

logger = logging.getLogger(__name__)

# Default embedding model when nothing is configured.
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingMatcher(ContentMatcher):
    """Matcher that uses sentence-transformer embeddings + cosine similarity.

    Long documents are chunked and mean-pooled into a single vector.
    """

    name = "embedding"

    def __init__(
        self,
        model_name: str = _DEFAULT_EMBEDDING_MODEL,
        dim: int = 384,
        *,
        chunk_size: int = 512,
    ) -> None:
        self._model_name = model_name
        self._dim = dim
        self._chunk_size = chunk_size
        self._model: Any = None  # lazy-loaded

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name

    def fingerprint(self, text: str) -> list[float]:
        """Compute a mean-pooled embedding vector for *text*.

        Long texts are split into chunks of ``chunk_size`` tokens, each
        chunk is embedded, and the resulting vectors are averaged.
        """
        if not text or not text.strip():
            return [0.0] * self._dim

        model = self._get_model()
        chunks = self._chunk_text(text)

        if len(chunks) == 1:
            vec = model.encode(chunks[0], normalize_embeddings=True)
            return vec.tolist()

        # Mean-pool across chunks
        vectors = model.encode(chunks, normalize_embeddings=True)
        mean = vectors.mean(axis=0)
        # Re-normalize after averaging
        norm = math.sqrt((mean * mean).sum())
        if norm > 0:
            mean = mean / norm
        return mean.tolist()

    def score(self, fp_a: list[float], fp_b: list[float]) -> float:  # noqa: PLR6301 — overrides ABC
        """Cosine similarity between two embedding vectors."""
        dot = sum(a * b for a, b in zip(fp_a, fp_b, strict=True))
        norm_a = math.sqrt(sum(a * a for a in fp_a))
        norm_b = math.sqrt(sum(b * b for b in fp_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def serialize(self, fingerprint: list[float]) -> list:  # noqa: PLR6301 — overrides ABC
        return fingerprint

    def deserialize(self, data: list) -> list[float]:  # noqa: PLR6301 — overrides ABC
        return [float(v) for v in data]

    # ── Internals ────────────────────────────────────────────────────────

    def _get_model(self) -> Any:
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            # Deferred import: sentence-transformers is heavy and only
            # needed when dedup is enabled with the embedding matcher.
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 512) -> list[str]:
        """Split *text* into roughly equal-sized word chunks.

        Simple whitespace-based splitting — no dependency on chonkie
        needed here since we only need approximate chunk boundaries for
        mean-pooling.
        """
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        chunks: list[str] = []
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i : i + chunk_size]))
        return chunks

    def __repr__(self) -> str:
        return f"EmbeddingMatcher(model={self._model_name!r}, dim={self._dim})"
