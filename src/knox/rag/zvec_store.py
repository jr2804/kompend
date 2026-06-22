"""zvec-based vector store for semantic search over compiled wikis.

zvec is an embedded vector database (Alibaba, Apache 2.0) — in-process,
no server required. This module wraps it as a sidecar for KNOX's compiled
knowledge base.

Embedding providers are pluggable via ``EmbeddingProvider``. Two backends
are supported via ``_make_provider``:

- ``litellm:<model>`` — routes through LiteLLM (OpenAI, Azure, Ollama, etc.)
- ``<model>`` (no prefix) — loads a sentence-transformers model from HuggingFace
"""

from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import litellm
import zvec as _zvec
from sentence_transformers import SentenceTransformer

from knox.core.models import CompiledWiki, WikiNode

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

    - ``litellm:<model>`` prefix → LiteLLMEmbeddingProvider (OpenAI, Ollama, etc.)
    - No prefix → SentenceTransformerProvider (HuggingFace model name)
    """
    if embedding_model.startswith("litellm:"):
        model = embedding_model.removeprefix("litellm:")
        return LiteLLMEmbeddingProvider(model=model, dim=dim)
    return SentenceTransformerProvider(model_name=embedding_model)


class ZvecWikiStore:
    """Sidecar vector store for semantic search over compiled wikis.

    The store is dormant until build_index() is called.
    Uses a single zvec collection at ``db_path``.
    """

    _PREVIEW_LEN: int = 1000

    def __init__(
        self,
        db_path: str | Path = "./.knox/vectors",
        embedding_provider: EmbeddingProvider | None = None,
        embedding_dim: int = 384,
    ) -> None:
        self.db_path = Path(db_path)
        self._provider = embedding_provider
        self._embedding_dim = embedding_dim
        self._collection: Any | None = None

    @property
    def available(self) -> bool:
        return self._collection is not None

    def build_index(
        self,
        wiki: CompiledWiki,
        embedding_model: str | None = None,
        *,
        batch_size: int = 10,
    ) -> int:
        """Index compiled wiki nodes into the zvec store.

        If no provider was set at construction, ``embedding_model`` must be
        given to create one via :func:`_make_provider`.
        """
        if not wiki.nodes:
            return 0

        if self._provider is None:
            if not embedding_model:
                raise ValueError("embedding_model is required when no provider was injected")
            self._provider = _make_provider(embedding_model, dim=self._embedding_dim)
            self._embedding_dim = self._provider.dim

        self._collection = self._create_collection()
        return self._index_nodes(self._collection, wiki.nodes, batch_size)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search indexed wiki content by semantic similarity."""
        if not self.available or self._provider is None:
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
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> ZvecWikiStore:
        """Open an existing zvec collection for querying.

        The collection must have been created previously with
        :meth:`build_index`. Either pass ``embedding_provider`` directly,
        or pass ``embedding_model`` to create one via :func:`_make_provider`.
        """
        provider = embedding_provider or _make_provider(embedding_model or "", dim=embedding_dim or 384)
        store = cls(db_path=db_path, embedding_provider=provider, embedding_dim=provider.dim)
        coll_path = Path(store.db_path)
        if coll_path.exists():
            try:
                store._collection = _zvec.open(path=str(coll_path))
            except Exception as exc:
                logger.warning("Could not open zvec store at %s: %s", coll_path, exc, exc_info=True)
        else:
            logger.warning("No zvec store found at %s. Rebuild with build_index().", coll_path)
        return store

    def _create_collection(self) -> Any:
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
        collection = _zvec.create_and_open(
            path=str(self.db_path),
            schema=schema,
        )
        self._collection = collection
        return collection

    def _index_nodes(
        self,
        collection: Any,
        nodes: list[WikiNode],
        batch_size: int,
    ) -> int:
        provider = self._provider
        if provider is None:
            raise RuntimeError("Embedding provider not configured. Call build_index() first.")
        docs_batch: list[Any] = []
        indexed_count = 0
        total_nodes = len(nodes)

        for node in nodes:
            doc = _zvec.Doc(
                id=node.id,
                vectors={"embedding": provider.embed(f"{node.title}\n{node.content}")},
                fields={
                    "title": node.title,
                    "node_type": node.node_type,
                    "source_file": str(node.metadata.get("source", "")) if node.metadata else "",
                    "content_preview": node.content[: self._PREVIEW_LEN],
                },
            )
            docs_batch.append(doc)

            if len(docs_batch) >= batch_size:
                collection.insert(docs_batch)
                indexed_count += len(docs_batch)
                logger.info("Indexed %d/%d nodes in zvec store", indexed_count, total_nodes)
                docs_batch = []

        if docs_batch:
            collection.insert(docs_batch)
            indexed_count += len(docs_batch)

        logger.info("Finished indexing %d nodes in zvec store at %s", indexed_count, self.db_path)
        return indexed_count
