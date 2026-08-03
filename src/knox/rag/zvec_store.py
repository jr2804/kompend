"""zvec-based vector store for semantic search over compiled wikis.

zvec is an embedded vector database (Alibaba, Apache 2.0) — in-process,
no server required. This module wraps it as a sidecar for KNOX's compiled
knowledge base.

Embedding providers are pluggable via ``EmbeddingProvider``. The default
``PydanticAIEmbeddingProvider`` wraps pydantic-ai's :class:`~pydantic_ai.Embedder`,
which supports every provider pydantic-ai ships (openai, ollama, google,
cohere, bedrock, and HuggingFace sentence-transformers).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import Any

import zvec as _zvec
from pydantic_ai import Embedder
from pydantic_ai.embeddings import infer_embedding_model
from pydantic_ai.providers.ollama import OllamaProvider

from knox.core.models import CompiledWiki, MetadataField, WikiNode
from knox.core.paths import rel_to_vault
from knox.llm.models import OLLAMA_DEFAULT_BASE_URL
from knox.logging import get_logger
from knox.types import ZvecCollection

_ZVEC_MAX_ID_LEN = 64


_ZVEC_COLLECTION_NAME = "knox_wiki"

logger = get_logger(__name__)


class ZvecField(StrEnum):
    """zvec collection field and vector names — the single source of truth.

    Every schema field declared in :meth:`ZvecWikiStore._create_collection`,
    every key written in :meth:`ZvecWikiStore._index_nodes`, and every key
    read from ``hit`` dicts in ``search()`` callers must use these members.
    Hard-coding the string literals defeats the schema/reader contract and is
    forbidden (see ``src/knox/rag/AGENTS.md``).
    """

    NODE_ID = "node_id"
    TITLE = "title"
    NODE_TYPE = "node_type"
    SOURCE_FILE = "source_file"
    CONTENT_PREVIEW = "content_preview"
    TAGS = "tags"
    EMBEDDING = "embedding"


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


class PydanticAIEmbeddingProvider(EmbeddingProvider):
    """Embedding via pydantic-ai's :class:`~pydantic_ai.Embedder`.

    Supports every embedding provider pydantic-ai ships (openai, ollama,
    google, cohere, bedrock, and HuggingFace sentence-transformers).
    The *model_string* uses the same ``provider:model`` convention as
    :func:`knox.llm.models.build_model`.
    """

    def __init__(self, model_string: str) -> None:
        self._model_string = model_string
        self._embedder = build_embedder(model_string)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Determine dimensionality by embedding a short probe text.
            self._dim = len(self.embed("dimension probe"))
        return self._dim

    def embed(self, text: str) -> list[float]:
        result = self._embedder.embed_documents_sync(text)
        vector = list(result.embeddings[0])
        if self._dim is None:
            self._dim = len(vector)
        return vector


class ZvecWikiStore:
    """Sidecar vector store for semantic search over compiled wikis.

    The store is dormant until build_index() is called.
    Uses a single zvec collection at ``db_file``.
    """

    _PREVIEW_LEN: int = 1000

    def __init__(
        self,
        db_file: str | Path = "./.knox/vectors",
        embedding_provider: EmbeddingProvider | None = None,
        embedding_dim: int = 384,
    ) -> None:
        self.db_file = Path(db_file)
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
        base_dir: Path | None = None,
    ) -> int:
        """Index compiled wiki nodes into the zvec store.

        If no provider was set at construction, ``embedding_model`` must be
        given to create one via :func:`make_provider`.
        """
        if not wiki.nodes:
            return 0

        if self._provider is None:
            if not embedding_model:
                raise ValueError("embedding_model is required when no provider was injected")
            self._provider = make_provider(embedding_model, dim=self._embedding_dim)
            self._embedding_dim = self._provider.dim

        self._collection = self._create_collection()
        return self._index_nodes(self._collection, wiki.nodes, batch_size, base_dir=base_dir)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search indexed wiki content by semantic similarity."""
        if not self.available or self._provider is None:
            return []
        if self._collection is None:
            raise RuntimeError("Vector collection not initialized")
        results = self._collection.query(
            _zvec.VectorQuery(ZvecField.EMBEDDING, vector=self._provider.embed(query)),
            topk=top_k,
        )
        return [
            {
                "id": (r.fields.get(ZvecField.NODE_ID, r.id) if isinstance(r.fields, dict) else r.id),
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
        db_file: str | Path,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> ZvecWikiStore:
        """Open an existing zvec collection for querying.

        The collection must have been created previously with
        :meth:`build_index`. Either pass ``embedding_provider`` directly,
        or pass ``embedding_model`` to create one via :func:`make_provider`.
        """
        provider = embedding_provider or make_provider(embedding_model or "", dim=embedding_dim or 384)
        store = cls(db_file=db_file, embedding_provider=provider, embedding_dim=provider.dim)
        coll_dir = Path(store.db_file)
        if coll_dir.exists():
            try:
                store._collection = _zvec.open(path=str(coll_dir))
            except Exception as exc:
                logger.warning("Could not open zvec store at %s: %s", coll_dir, exc, exc_info=True)
        else:
            logger.warning("No zvec store found at %s. Rebuild with build_index().", coll_dir)
        return store

    def _create_collection(self) -> ZvecCollection:
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        schema = _zvec.CollectionSchema(
            name=_ZVEC_COLLECTION_NAME,
            fields=[
                _zvec.FieldSchema(ZvecField.NODE_ID, _zvec.DataType.STRING),
                _zvec.FieldSchema(ZvecField.TITLE, _zvec.DataType.STRING),
                _zvec.FieldSchema(ZvecField.NODE_TYPE, _zvec.DataType.STRING),
                _zvec.FieldSchema(ZvecField.SOURCE_FILE, _zvec.DataType.STRING),
                _zvec.FieldSchema(ZvecField.CONTENT_PREVIEW, _zvec.DataType.STRING),
                _zvec.FieldSchema(ZvecField.TAGS, _zvec.DataType.STRING),
            ],
            vectors=_zvec.VectorSchema(
                ZvecField.EMBEDDING,
                _zvec.DataType.VECTOR_FP32,
                self._embedding_dim,
            ),
        )
        collection = _zvec.open(path=str(self.db_file)) if self.db_file.exists() else _zvec.create_and_open(path=str(self.db_file), schema=schema)
        self._collection = collection
        return collection

    def _index_nodes(
        self,
        collection: Any,
        nodes: list[WikiNode],
        batch_size: int,
        *,
        base_dir: Path | None = None,
    ) -> int:
        provider = self._provider
        if provider is None:
            raise RuntimeError("Embedding provider not configured. Call build_index() first.")
        docs_batch: list[Any] = []
        indexed_count = 0
        total_nodes = len(nodes)

        for node in nodes:
            raw_source = str(node.metadata.get(MetadataField.SOURCE, "")) if node.metadata else ""
            source_file = rel_to_vault(raw_source, base_dir) if base_dir else raw_source
            doc = _zvec.Doc(
                id=_safe_id(node.id),
                vectors={ZvecField.EMBEDDING: provider.embed(f"{node.title}\n{node.content}")},
                fields={
                    ZvecField.NODE_ID: node.id,
                    ZvecField.TITLE: node.title,
                    ZvecField.NODE_TYPE: node.node_type,
                    ZvecField.SOURCE_FILE: source_file,
                    ZvecField.CONTENT_PREVIEW: node.content[: self._PREVIEW_LEN],
                    ZvecField.TAGS: ",".join(node.metadata.get(MetadataField.TAGS, [])) if node.metadata.get(MetadataField.TAGS) else "",
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

        logger.info("Finished indexing %d nodes in zvec store at %s", indexed_count, self.db_file)
        return indexed_count


def _safe_id(node_id: str) -> str:
    """Truncate *node_id* to zvec's 64-char PK limit with a hash suffix."""
    if len(node_id) <= _ZVEC_MAX_ID_LEN:
        return node_id
    suffix = hashlib.md5(node_id.encode()).hexdigest()[:8]  # noqa: S324
    return f"{node_id[: _ZVEC_MAX_ID_LEN - 9]}-{suffix}"


# ── Embedding provider interface ────────────────────────────────────────


def build_embedder(model_string: str) -> Embedder:
    """Build a pydantic-ai ``Embedder`` with Ollama base_url support.

    Mirrors the pattern in :mod:`knox.llm.models`: when the provider is
    Ollama, explicitly construct ``OllamaProvider`` with the local default
    ``base_url`` so it works out-of-the-box without ``OLLAMA_BASE_URL``.
    """
    if model_string.startswith("ollama:"):
        base_url = os.getenv("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_BASE_URL
        provider = OllamaProvider(base_url=base_url)
        model = infer_embedding_model(model_string, provider_factory=lambda _: provider)
        return Embedder(model)
    return Embedder(model_string)


def make_provider(
    embedding_model: str,
    dim: int | None = None,
) -> EmbeddingProvider:
    """Select an embedding provider based on model name.

    Uses the same ``provider/model`` or ``provider:model`` convention as
    :func:`knox.llm.models.build_model`. A bare HuggingFace model name
    (no prefix) defaults to the ``sentence-transformers`` provider.
    """
    pydantic_ai_string = to_pydantic_ai_embedding_string(embedding_model)
    return PydanticAIEmbeddingProvider(model_string=pydantic_ai_string)


def to_pydantic_ai_embedding_string(embedding_model: str) -> str:
    """Translate a KNOX embedding-model config value to pydantic-ai format.

    Uses the same ``provider/model`` or ``provider:model`` convention as
    :func:`knox.llm.models.build_model`. A bare HuggingFace model name
    (e.g. ``BAAI/bge-small-en`` or ``all-MiniLM-L6-v2``) defaults to the
    ``sentence-transformers`` provider.
    """
    # Known LLM/embedding providers — anything else is a HF model name.
    _KNOWN_PROVIDERS = frozenset(
        {
            "ollama",
            "openai",
            "google",
            "cohere",
            "bedrock",
            "groq",
            "mistral",
            "deepseek",
            "sentence-transformers",
        }
    )

    if ":" in embedding_model:
        provider = embedding_model.split(":", 1)[0]
        if provider in _KNOWN_PROVIDERS:
            return embedding_model
    if "/" in embedding_model:
        provider = embedding_model.split("/", 1)[0]
        if provider in _KNOWN_PROVIDERS:
            return provider + ":" + embedding_model.split("/", 1)[1]
    return f"sentence-transformers:{embedding_model}"
