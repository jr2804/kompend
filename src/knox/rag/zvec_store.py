"""zvec-based vector store for semantic search over compiled wikis.

zvec is an embedded vector database (Alibaba, Apache 2.0) — in-process,
no server required. This module wraps it as a sidecar for KNOX's compiled
knowledge base.

Embedding providers are pluggable via ``EmbeddingProvider``. The default
uses sentence-transformers (all-MiniLM-L6-v2, 384-dim). A hash-based
fallback exists for environments without ML libraries (tests, CI).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import litellm
import zvec as _zvec
from sentence_transformers import SentenceTransformer

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
    """Real embedding via sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim: int = self._model.get_embedding_dimension() or 384

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """Embedding via litellm (supports local Ollama, OpenAI, Azure, etc.)."""

    def __init__(self, model: str, dim: int | None = None) -> None:
        self._model = model
        self._dim: int | None = dim

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("Embedding dimension not yet determined. Call embed() first.")
        return self._dim

    def embed(self, text: str) -> list[float]:
        kwargs: dict[str, Any] = {"model": self._model, "input": [text]}
        # If user explicitly requested a smaller dimension, pass it to LiteLLM
        # (Supported natively by models like OpenAI's text-embedding-3 series)
        if self._dim is not None:
            kwargs["dimensions"] = self._dim
        
        response = litellm.embedding(**kwargs)
        vector = response.data[0]["embedding"]
        actual_dim = len(vector)
        
        # Auto-detect dimension on first call if not explicitly provided
        if self._dim is None:
            self._dim = actual_dim
        # Validate that the model actually respected the requested dimension
        elif actual_dim != self._dim:
            raise ValueError(
                f"Embedding dimension mismatch: Requested {self._dim}, but model "
                f"'{self._model}' returned a vector of size {actual_dim}. The model "
                f"may not support the requested dimension, or it may ignore the parameter."
            )
            
        return vector


def _make_provider(
    embedding_model: str,
    dim: int | None = None,
) -> EmbeddingProvider:
    """Select an embedding provider based on model name.

    - "hash" → HashEmbeddingProvider (tests/dev)
    - "all-MiniLM-L6-v2" or other ST model names → SentenceTransformerProvider
    - "text-embedding-*" or explicit litellm prefix → LiteLLMEmbeddingProvider
    """
    if embedding_model == "hash":
        return HashEmbeddingProvider(dim=dim or 384)
    if embedding_model.startswith("text-embedding") or embedding_model.startswith("litellm:"):
        model = embedding_model.removeprefix("litellm:")
        return LiteLLMEmbeddingProvider(model=model, dim=dim)
    return SentenceTransformerProvider(model_name=embedding_model)


class ZvecWikiStore:
    """Sidecar vector store for semantic search over compiled wikis.

    The store is dormant until build_index() is called.
    """

    _PREVIEW_LEN: int = 1000

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
        return self._collection is not None

    def build_index(
        self,
        wiki: CompiledWiki,
        embedding_model: str,
        *,
        batch_size: int = 10,
    ) -> int:
        """Index compiled wiki nodes into the zvec store.

        If ``embedding_model`` is provided and no provider was set at
        construction, a new provider is created via :func:`_make_provider`.
        """
        if not wiki.nodes:
            return 0

        if isinstance(self._provider, HashEmbeddingProvider):
            self._provider = _make_provider(embedding_model, dim=self._embedding_dim)
            self._embedding_dim = self._provider.dim

        if self._collection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            schema = _zvec.CollectionSchema(
                name="knox_wiki",
                fields=[
                    _zvec.FieldSchema("title", _zvec.DataType.STRING),
                    _zvec.FieldSchema("node_type", _zvec.DataType.STRING),
                    _zvec.FieldSchema("source_file", _zvec.DataType.STRING),
                    _zvec.FieldSchema("content_preview", _zvec.DataType.STRING),
                ],
                vectors=_zvec.VectorSchema(
                    "embedding",
                    _zvec.DataType.VECTOR_FP32,
                    self._embedding_dim,
                ),
            )
            self._collection = _zvec.create_and_open(
                path=str(self.db_path),
                schema=schema,
            )

        # Process in batches to avoid holding all embeddings in memory
        # and to provide incremental progress logging.
        docs_batch: list[Any] = []
        indexed_count = 0
        total_nodes = len(wiki.nodes)

        for node in wiki.nodes:
            doc = _zvec.Doc(
                id=node.id,
                vectors={"embedding": self._provider.embed(f"{node.title}\n{node.content}")},
                fields={
                    "title": node.title,
                    "node_type": node.node_type,
                    "source_file": str(node.metadata.get("source", "")) if node.metadata else "",
                    "content_preview": node.content[: self._PREVIEW_LEN],
                },
            )
            docs_batch.append(doc)
            
            if len(docs_batch) >= batch_size:
                self._collection.insert(docs_batch)
                indexed_count += len(docs_batch)
                logger.info("Indexed %d/%d nodes in zvec store", indexed_count, total_nodes)
                docs_batch = []

        # Insert any remaining nodes
        if docs_batch:
            self._collection.insert(docs_batch)
            indexed_count += len(docs_batch)
            
        logger.info("Finished indexing %d nodes in zvec store at %s", indexed_count, self.db_path)
        return indexed_count

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search indexed wiki content by semantic similarity."""
        if not self.available:
            return []
        results = self._collection.query(
            _zvec.VectorQuery("embedding", vector=self._provider.embed(query)),
            topk=top_k,
        )
        return [
            {
                "id": r.id,
                "score": r.score if r.score is not None else 0.0,
                **(r.fields if isinstance(r.fields, dict) else {}),
            }
            for r in results
        ]

    def close(self) -> None:
        if self._collection is not None:
            with contextlib.suppress(Exception):
                self._collection.flush()
            self._collection = None

    @classmethod
    def open(
        cls,
        db_path: str | Path,
        embedding_model: str,
        embedding_dim: int | None = None,
    ) -> ZvecWikiStore:
        """Open an existing zvec collection for querying.

        The collection must have been created previously with
        :meth:`build_index`. The embedding model/dimension are used to
        reconstruct the correct provider.
        """
        provider = _make_provider(embedding_model, dim=embedding_dim or 384)
        store = cls(db_path=db_path, embedding_provider=provider, embedding_dim=provider.dim)
        store._collection = _zvec.open(path=str(store.db_path))
        return store
