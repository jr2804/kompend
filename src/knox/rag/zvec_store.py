"""zvec-based vector store for semantic search over compiled wikis.

zvec is an embedded vector database (Alibaba, Apache 2.0) — in-process,
no server required. This module wraps it as a sidecar for KNOX's compiled
knowledge base.

Embedding providers are pluggable via ``EmbeddingProvider``. The default
uses sentence-transformers (all-MiniLM-L6-v2, 384-dim). A hash-based
fallback exists for environments without ML libraries (tests, CI).

NOTE: zvec uses CPU-specific instructions (AVX) and may crash with SIGILL
on older x86 CPUs. A subprocess-based availability check at import time
prevents the crash from taking down the interpreter.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from knox.core.models import CompiledWiki

logger = logging.getLogger(__name__)


# ── Embedding provider interface ────────────────────────────────────────


class EmbeddingProvider(ABC):
    """Interface for embedding text into fixed-dimension vectors."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a normalized embedding vector for ``text``."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimensionality."""
        ...


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based embedding for testing/dev only.

    NOT semantically meaningful — same text always produces the same vector,
    but unrelated texts get unrelated vectors. Use only for unit tests or
    environments without sentence-transformers.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(digest[i % 32] - 128) / 128.0 for i in range(self._dim)]
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm > 0 else vec


class SentenceTransformerProvider(EmbeddingProvider):
    """Real embedding via sentence-transformers.

    Defaults to all-MiniLM-L6-v2 (384-dim, fast, good quality).
    Falls back to :class:`HashEmbeddingProvider` if sentence-transformers
    is not installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(model_name)
            self._dim: int = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]
            self._available = True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — falling back to hash embeddings. "
                "Install with: uv pip install sentence-transformers"
            )
            self._model = None  # type: ignore[assignment]
            self._dim = 384
            self._available = False

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def available(self) -> bool:
        return self._available

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            return HashEmbeddingProvider(dim=self._dim).embed(text)
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """Cloud embedding via litellm (OpenAI, Azure, etc.)."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536) -> None:
        self._model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        import litellm  # noqa: PLC0415

        response = litellm.embedding(model=self._model, input=[text])
        return response.data[0]["embedding"]


def _make_provider(
    embedding_model: str | None = None,
    dim: int = 384,
) -> EmbeddingProvider:
    """Select an embedding provider based on model name.

    - None or "hash" → HashEmbeddingProvider (tests/dev)
    - "all-MiniLM-L6-v2" or other ST model names → SentenceTransformerProvider
    - "text-embedding-*" or explicit litellm prefix → LiteLLMEmbeddingProvider
    """
    if embedding_model is None or embedding_model == "hash":
        return HashEmbeddingProvider(dim=dim)
    if embedding_model.startswith("text-embedding") or embedding_model.startswith("litellm:"):
        model = embedding_model.removeprefix("litellm:")
        return LiteLLMEmbeddingProvider(model=model)
    return SentenceTransformerProvider(model_name=embedding_model)


# ── zvec availability check ─────────────────────────────────────────────


def _zvec_available() -> bool:
    """Check if zvec can be imported without crashing (SIGILL-safe)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import zvec; print('ok')"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "ok"
    except Exception:
        return False


if _zvec_available():
    import zvec as _zvec  # noqa: PLC0415
    ZVEC_AVAILABLE = True
else:
    ZVEC_AVAILABLE = False
    _zvec = None  # type: ignore[assignment]


class ZvecWikiStore:
    """Sidecar vector store for semantic search over compiled wikis.

    The store is dormant until build_index() is called.
    """

    def __init__(
        self,
        db_path: str | Path = "./.knox/vectors",
        embedding_provider: EmbeddingProvider | None = None,
        embedding_dim: int = 384,
    ) -> None:
        self.db_path = Path(db_path)
        self._provider = embedding_provider or HashEmbeddingProvider(dim=embedding_dim)
        self._embedding_dim = self._provider.dim
        self._collection: Any = None

    @property
    def available(self) -> bool:
        return ZVEC_AVAILABLE and self._collection is not None

    def build_index(
        self,
        wiki: CompiledWiki,
        embedding_model: str | None = None,
    ) -> int:
        """Index compiled wiki nodes into the zvec store.

        If ``embedding_model`` is provided and no provider was set at
        construction, a new provider is created via :func:`_make_provider`.
        """
        if not ZVEC_AVAILABLE or not _zvec:
            logger.warning("zvec not available — skipping index build")
            return 0
        if not wiki.nodes:
            return 0

        # Select provider if caller specified a model
        if embedding_model is not None and isinstance(self._provider, HashEmbeddingProvider):
            self._provider = _make_provider(embedding_model, dim=self._embedding_dim)
            self._embedding_dim = self._provider.dim

        if self._collection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            schema = _zvec.CollectionSchema(
                name="knox_wiki",
                vectors=_zvec.VectorSchema(
                    "embedding", _zvec.DataType.VECTOR_FP32, self._embedding_dim,
                ),
            )
            self._collection = _zvec.create_and_open(
                path=str(self.db_path), schema=schema,
            )

        docs = [
            _zvec.Doc(
                id=node.id,
                vectors={"embedding": self._provider.embed(f"{node.title}\n{node.content}")},
                metadata={
                    "title": node.title,
                    "node_type": node.node_type,
                    "content_preview": node.content[:200],
                },
            )
            for node in wiki.nodes
        ]
        self._collection.insert(docs)
        logger.info("Indexed %d nodes in zvec store at %s", len(docs), self.db_path)
        return len(docs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search indexed wiki content by semantic similarity."""
        if not self.available or not _zvec:
            return []
        results = self._collection.query(
            _zvec.VectorQuery("embedding", vector=self._provider.embed(query)),
            topk=top_k,
        )
        return [
            {
                "id": r.get("id", ""),
                "score": r.get("score", 0.0),
                **(r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}),
            }
            for r in results
        ]

    def close(self) -> None:
        if self._collection is not None:
            self._collection.close()
            self._collection = None