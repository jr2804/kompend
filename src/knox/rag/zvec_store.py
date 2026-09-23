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
import json
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import zvec as _zvec
from pydantic_ai import Embedder
from pydantic_ai.embeddings import infer_embedding_model
from pydantic_ai.providers.ollama import OllamaProvider

from knox.core.models import CompiledWiki, MetadataField, WikiNode
from knox.core.paths import rel_to_vault
from knox.llm.models import OLLAMA_DEFAULT_BASE_URL
from knox.logging import get_logger
from knox.types import ZvecCollection, ZvecDoc

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
    ATTRIBUTES = "attributes"
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


class VectorStoreLike(Protocol):
    """Vector-store surface used by the retrievers.

    :class:`ZvecWikiStore` satisfies this structurally; tests inject a
    lightweight in-memory fake.
    """

    @property
    def available(self) -> bool: ...

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...


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
    ) -> None:
        self.db_file = Path(db_file)
        self._provider = embedding_provider
        self._collection: ZvecCollection | None = None

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
            self._provider = make_provider(embedding_model)

        self._collection = self._create_collection()
        count = self._index_nodes(self._collection, wiki.nodes, batch_size, base_dir=base_dir)
        self._prune_orphans(self._collection, wiki.nodes)
        # HNSW is built lazily: a configured vector index sits at 0% completeness
        # until optimize() runs. Build it exactly once after the bulk insert —
        # optimize() costs ~40 ms per call, so never call it per document.
        self._collection.optimize()
        return count

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        tags_any: Sequence[str] | None = None,
        tags_all: Sequence[str] | None = None,
        attributes_any: Sequence[str] | None = None,
        attributes_all: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search indexed wiki content by semantic similarity.

        Optional keyword filters are pushed down into zvec as
        ``CONTAIN_ANY``/``CONTAIN_ALL`` clauses over the array fields
        ``tags``/``attributes``; multiple clauses combine with ``and``.
        Omitting them reproduces the unfiltered result set.
        """
        if not self.available or self._provider is None:
            return []
        if self._collection is None:
            raise RuntimeError("Vector collection not initialized")
        results = self._collection.query(
            _zvec.Query(ZvecField.EMBEDDING, vector=self._provider.embed(query)),
            topk=top_k,
            filter=_build_filter(
                tags_any=tags_any,
                tags_all=tags_all,
                attributes_any=attributes_any,
                attributes_all=attributes_all,
            ),
        )
        return [
            {
                "id": (r.fields.get(ZvecField.NODE_ID, r.id) if isinstance(r.fields, dict) else r.id),
                "score": _score_from_distance(r.score),
                **(r.fields if isinstance(r.fields, dict) else {}),
            }
            for r in results
        ]

    def close(self) -> None:
        if self._collection is not None:
            with contextlib.suppress(Exception):
                self._collection.flush()
            self._collection = None

    def is_stale(self, wiki: CompiledWiki) -> bool:
        """Return True when the store's last build doesn't match ``wiki``.

        Compares the set of indexed node ids (from the build-time manifest)
        against the current wiki's node set. A missing manifest, a corrupt
        manifest, or any id set mismatch means the sidecar index is stale and
        must be rebuilt before it can serve the loaded wiki.
        """
        expected = {_safe_id(n.id) for n in wiki.nodes}
        if not expected:
            return True
        manifest = self._manifest_file()
        if not manifest.is_file():
            return True
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            stored = {str(i) for i in raw}
        except (json.JSONDecodeError, OSError):
            logger.warning("Vector manifest corrupt — treating store as stale: %s", manifest)
            return True
        return stored != expected

    @classmethod
    def open(
        cls,
        db_file: str | Path,
        embedding_model: str | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> ZvecWikiStore:
        """Open an existing zvec collection for querying.

        The collection must have been created previously with
        :meth:`build_index`. Either pass ``embedding_provider`` directly,
        or pass ``embedding_model`` to create one via :func:`make_provider`.
        """
        provider = embedding_provider or make_provider(embedding_model or "")
        store = cls(db_file=db_file, embedding_provider=provider)
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
        if self._provider is None:
            raise RuntimeError("Embedding provider not configured. Call build_index() first.")
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        schema = _zvec.CollectionSchema(
            name=_ZVEC_COLLECTION_NAME,
            fields=self._expected_field_schemas(),
            vectors=_zvec.VectorSchema(
                ZvecField.EMBEDDING,
                _zvec.DataType.VECTOR_FP32,
                self._provider.dim,
                index_param=_zvec.HnswIndexParam(metric_type=_zvec.MetricType.COSINE),
            ),
        )
        if self.db_file.exists():
            try:
                collection = _zvec.open(path=str(self.db_file))
                stored_dim = collection.schema.vectors[0].dimension
                stored_fields = {f.name: f.data_type for f in collection.schema.fields}
                expected_fields = {f.name: f.data_type for f in schema.fields}
                if stored_dim != self._provider.dim:
                    logger.warning(
                        "Vector store at %s has dimension %d but the embedding model produces %d; rebuilding",
                        self.db_file,
                        stored_dim,
                        self._provider.dim,
                    )
                    collection.destroy()
                    collection = _zvec.create_and_open(path=str(self.db_file), schema=schema)
                elif stored_fields != expected_fields:
                    # Compare (name, data_type), not names alone: a field that
                    # keeps its name but changes type (e.g. tags STRING ->
                    # ARRAY_STRING) must rebuild rather than silently accept
                    # list values into a STRING column.
                    missing = sorted(set(expected_fields) - set(stored_fields))
                    obsolete = sorted(set(stored_fields) - set(expected_fields))
                    retyped = sorted(
                        f"{name}: {stored_fields[name].name}->{expected_fields[name].name}"
                        for name in set(stored_fields) & set(expected_fields)
                        if stored_fields[name] != expected_fields[name]
                    )
                    logger.warning(
                        "Vector store at %s schema mismatch: missing=%s, obsolete=%s, retyped=%s; destroying and rebuilding (re-embedding required)",
                        self.db_file,
                        missing,
                        obsolete,
                        retyped,
                    )
                    collection.destroy()
                    collection = _zvec.create_and_open(path=str(self.db_file), schema=schema)
            except Exception as exc:
                logger.warning("Vector store at %s is unreadable (%s); rebuilding", self.db_file, exc)
                shutil.rmtree(self.db_file, ignore_errors=True)
                collection = _zvec.create_and_open(path=str(self.db_file), schema=schema)
        else:
            collection = _zvec.create_and_open(path=str(self.db_file), schema=schema)
        self._collection = collection
        return collection

    def _index_nodes(
        self,
        collection: ZvecCollection,
        nodes: list[WikiNode],
        batch_size: int,
        *,
        base_dir: Path | None = None,
    ) -> int:
        provider = self._provider
        if provider is None:
            raise RuntimeError("Embedding provider not configured. Call build_index() first.")
        docs_batch: list[ZvecDoc] = []
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
                    ZvecField.TAGS: list(node.metadata.get(MetadataField.TAGS, [])),
                    ZvecField.ATTRIBUTES: _attributes_to_list(node.metadata.get(MetadataField.ATTRIBUTES) or {}),
                },
            )
            docs_batch.append(doc)

            if len(docs_batch) >= batch_size:
                collection.upsert(docs_batch)
                indexed_count += len(docs_batch)
                logger.info("Indexed %d/%d nodes in zvec store", indexed_count, total_nodes)
                docs_batch = []

        if docs_batch:
            collection.upsert(docs_batch)
            indexed_count += len(docs_batch)

        logger.info("Finished indexing %d nodes in zvec store at %s", indexed_count, self.db_file)
        return indexed_count

    def _manifest_file(self) -> Path:
        """Id manifest from the last successful build, next to the store."""
        return self.db_file.with_name(self.db_file.name + ".manifest.json")

    def _prune_orphans(self, collection: ZvecCollection, nodes: list[WikiNode]) -> int:
        """Delete store docs whose id is absent from *nodes* (renamed/removed pages).

        The candidate set is read from the collection's actual contents
        (``iter_docs``) rather than diffing the previous manifest, so docs
        written before the manifest mechanism existed are pruned too. Upsert
        keeps changed pages fresh; this removes ids the current wiki no longer
        has.
        """
        keep = {_safe_id(n.id) for n in nodes}
        present = {doc.id for doc in collection.iter_docs()}
        orphans = sorted(present - keep)
        if orphans:
            collection.delete(orphans)
            logger.info("Pruned %d stale vector(s) from %s", len(orphans), self.db_file)
        manifest = self._manifest_file()
        try:
            manifest.write_text(json.dumps(sorted(keep)), encoding="utf-8")
        except OSError as exc:  # pragma: no cover — filesystem-dependent
            logger.warning("Could not write vector manifest (%s): %s", manifest, exc)
        return len(orphans)

    @staticmethod
    def _expected_field_schemas() -> list[_zvec.FieldSchema]:
        """Canonical field list declared by the current knox version (SoT)."""
        return [
            _zvec.FieldSchema(ZvecField.NODE_ID, _zvec.DataType.STRING),
            _zvec.FieldSchema(ZvecField.TITLE, _zvec.DataType.STRING),
            _zvec.FieldSchema(ZvecField.NODE_TYPE, _zvec.DataType.STRING),
            _zvec.FieldSchema(ZvecField.SOURCE_FILE, _zvec.DataType.STRING),
            _zvec.FieldSchema(ZvecField.CONTENT_PREVIEW, _zvec.DataType.STRING),
            _zvec.FieldSchema(ZvecField.TAGS, _zvec.DataType.ARRAY_STRING, index_param=_zvec.InvertIndexParam()),
            _zvec.FieldSchema(ZvecField.ATTRIBUTES, _zvec.DataType.ARRAY_STRING, index_param=_zvec.InvertIndexParam()),
        ]


def _safe_id(node_id: str) -> str:
    """Truncate *node_id* to zvec's 64-char PK limit with a hash suffix."""
    if len(node_id) <= _ZVEC_MAX_ID_LEN:
        return node_id
    suffix = hashlib.md5(node_id.encode()).hexdigest()[:8]  # noqa: S324
    return f"{node_id[: _ZVEC_MAX_ID_LEN - 9]}-{suffix}"


def _attributes_to_list(attrs: dict[str, list[str]]) -> list[str]:
    """Flatten an attribute dict to filterable ``key=value`` tokens.

    Stored as an ``ARRAY_STRING`` so zvec evaluates ``CONTAIN_ANY``/
    ``CONTAIN_ALL`` membership server-side instead of a wildcard scan.
    """
    return [f"{key}={value}" for key, values in attrs.items() for value in values]


def _build_filter(
    *,
    tags_any: Sequence[str] | None = None,
    tags_all: Sequence[str] | None = None,
    attributes_any: Sequence[str] | None = None,
    attributes_all: Sequence[str] | None = None,
) -> str | None:
    """Compose a zvec filter string from optional array-membership filters.

    Returns ``None`` when nothing is requested, reproducing the unfiltered
    query. Only ``CONTAIN_ANY``/``CONTAIN_ALL`` are used: ``LIKE`` on an
    ``ARRAY_STRING`` field segfaults zvec 0.7.0.
    """
    clauses: list[str] = []
    if tags_all:
        clauses.append(_contain_clause(ZvecField.TAGS, tags_all, require_all=True))
    if tags_any:
        clauses.append(_contain_clause(ZvecField.TAGS, tags_any, require_all=False))
    if attributes_all:
        clauses.append(_contain_clause(ZvecField.ATTRIBUTES, attributes_all, require_all=True))
    if attributes_any:
        clauses.append(_contain_clause(ZvecField.ATTRIBUTES, attributes_any, require_all=False))
    return " and ".join(clauses) if clauses else None


def _contain_clause(field: str, values: Sequence[str], *, require_all: bool) -> str:
    """Build a zvec ``CONTAIN_ALL``/``CONTAIN_ANY`` clause over an array field."""
    operator = "CONTAIN_ALL" if require_all else "CONTAIN_ANY"
    quoted = ", ".join(f'"{_escape_filter_value(v)}"' for v in values)
    return f"{field} {operator} ({quoted})"


def _escape_filter_value(value: str) -> str:
    """Escape a value for a double-quoted zvec filter literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _score_from_distance(score: float | None) -> float:
    """Convert a zvec COSINE distance to a similarity score (higher = better).

    The embedding field is indexed with ``MetricType.COSINE``, whose zvec score
    is a distance in ``[0, 2]``. Retrievers rank descending by ``score``, so
    expose ``1 - distance``: for L2-normalised embeddings this equals the
    cosine similarity and reproduces the previous inner-product scores.
    """
    if score is None:
        return 0.0
    return 1.0 - score


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
    :func:`knox.llm.models.build_model`. An explicit provider prefix is
    required — there is no fallback to a local embedding provider.
    """
    pydantic_ai_string = to_pydantic_ai_embedding_string(embedding_model)
    return PydanticAIEmbeddingProvider(model_string=pydantic_ai_string)


def to_pydantic_ai_embedding_string(embedding_model: str) -> str:
    """Translate a KNOX embedding-model config value to pydantic-ai format.

    Uses the same ``provider/model`` or ``provider:model`` convention as
    :func:`knox.llm.models.build_model`. An explicit provider prefix is
    required — there is no silent fallback to a local embedding provider.
    """
    # Known LLM/embedding providers — anything else is rejected.
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
    raise ValueError(
        f"Embedding model {embedding_model!r} has no known provider prefix. Use provider/model or provider:model syntax, e.g. ollama:granite-embedding:278m."
    )
