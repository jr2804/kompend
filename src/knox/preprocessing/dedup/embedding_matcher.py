"""Embedding-based content similarity matcher.

Uses the configurable KNOX embedding provider (pydantic-ai ``Embedder``)
to compute document vectors, then compares via cosine similarity.
"""

from __future__ import annotations

import math

from knox.logging import get_logger
from knox.preprocessing.dedup.base import ContentMatcher, require_json_number
from knox.rag.zvec_store import EmbeddingProvider
from knox.types import JSONValue

logger = get_logger(__name__)


class EmbeddingMatcher(ContentMatcher[list[float]]):
    """Matcher that uses embedding vectors + cosine similarity.

    Long documents are chunked and mean-pooled into a single vector.

    Uses the existing KNOX ``EmbeddingProvider`` abstraction so that
    any pydantic-ai embedding provider (local or remote) works transparently.
    """

    name = "embedding"

    fp_type = list  # fingerprints are list[float]; list is the closest runtime check

    def __init__(
        self,
        model_name: str,
        dim: int,
        provider: EmbeddingProvider,
        *,
        chunk_size: int = 512,
    ) -> None:
        self._model_name = model_name
        self._dim = dim
        self._provider = provider
        self._chunk_size = chunk_size

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

        chunks = self._chunk_text(text)

        if len(chunks) == 1:
            vec = self._provider.embed(chunks[0])
            return self._normalize(vec)

        # Mean-pool across chunks
        vectors = [self._provider.embed(c) for c in chunks]
        mean = [sum(x) / len(vectors) for x in zip(*vectors, strict=False)]
        return self._normalize(mean)

    def score(self, fp_a: list[float], fp_b: list[float]) -> float:  # noqa: PLR6301 — overrides ABC
        """Cosine similarity between two embedding vectors."""
        dot = sum(a * b for a, b in zip(fp_a, fp_b, strict=True))
        norm_a = math.sqrt(sum(a * a for a in fp_a))
        norm_b = math.sqrt(sum(b * b for b in fp_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def serialize(self, fingerprint: list[float]) -> list[float]:  # noqa: PLR6301 — overrides ABC
        return fingerprint

    def deserialize(self, data: list[JSONValue]) -> list[float]:  # noqa: PLR6301 — overrides ABC
        return [require_json_number(v) for v in data]

    def __repr__(self) -> str:
        return f"EmbeddingMatcher(model={self._model_name!r}, dim={self._dim})"

    # ── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        """Return a unit-normalised copy of *vec*."""
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            return [v / norm for v in vec]
        return vec

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
