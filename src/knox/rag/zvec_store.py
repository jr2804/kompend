"""zvec-based vector store for semantic search over compiled wikis.

zvec is an embedded vector database (Alibaba, Apache 2.0) — in-process,
no server required. This module wraps it as a sidecar for KNOX's compiled
knowledge base.

Embedding providers are pluggable via ``EmbeddingProvider``. The default
``PydanticAIEmbeddingProvider`` wraps pydantic-ai's :class:`~pydantic_ai.Embedder`,
which supports every provider pydantic-ai ships (openai, ollama, google,
cohere, bedrock, and HuggingFace sentence-transformers).
"""

# ruff: noqa: S608 — SQL is built from Final[str] table-name constants only.

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sqlite3
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
from knox.core.paths import KNOX_DB_FILE, rel_to_vault
from knox.core.store import open_store
from knox.core.store.schema import VECTOR_MANIFEST_TABLE
from knox.core.utils.time import iso_local
from knox.llm.models import OLLAMA_DEFAULT_BASE_URL
from knox.logging import get_logger
from knox.types import ZvecCollection, ZvecDoc

_ZVEC_MAX_ID_LEN = 64


_ZVEC_COLLECTION_NAME = "knox_wiki"


_EMBED_TEXT_VERSION = 2

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
    ATTR_KEYS = "attr_keys"
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

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]: ...


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
        self._fts_unavailable = False

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
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_any: Sequence[str] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
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
                tags_dnf=tags_dnf,
                attributes_any=attributes_any,
                attributes_all=attributes_all,
                attribute_keys=attribute_keys,
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

    def fts_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        tags_any: Sequence[str] | None = None,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_any: Sequence[str] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        """Lexical (BM25) search over the FTS-indexed ``title`` + ``content_preview``.

        Returns the same dict shape as :meth:`search`; ``score`` is the fused
        BM25 relevance (higher = better, no distance conversion). Returns ``[]``
        when the store is closed or the collection has no FTS index (logged
        once) — never raises into the query path.
        """
        if not self.available or self._collection is None:
            return []
        try:
            results = self._collection.query(
                self._fts_queries(query),
                topk=top_k,
                filter=_build_filter(
                    tags_any=tags_any,
                    tags_all=tags_all,
                    tags_dnf=tags_dnf,
                    attributes_any=attributes_any,
                    attributes_all=attributes_all,
                    attribute_keys=attribute_keys,
                ),
                reranker=_zvec.RrfReRanker(),
            )
        except Exception as exc:
            self._note_fts_unavailable(exc)
            return []
        return [_hit_to_dict(r) for r in results]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        tags_any: Sequence[str] | None = None,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_any: Sequence[str] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
        vector_weight: float = 1.0,
        fts_weight: float = 0.3,
    ) -> list[dict[str, object]]:
        """Fuse vector similarity and lexical BM25 in ONE zvec call.

        Sub-queries are ``[vector, title-fts, content-preview-fts]`` combined by
        ``WeightedReRanker([vector_weight, fts_weight, fts_weight])``. zvec's
        weighted-reranker score is already higher-is-better (verified). When the
        collection has no FTS index, degrades to the vector-only path (logged
        once).
        """
        if not self.available or self._provider is None or self._collection is None:
            return []
        filter_expr = _build_filter(
            tags_any=tags_any,
            tags_all=tags_all,
            tags_dnf=tags_dnf,
            attributes_any=attributes_any,
            attributes_all=attributes_all,
            attribute_keys=attribute_keys,
        )
        try:
            results = self._collection.query(
                [_zvec.Query(ZvecField.EMBEDDING, vector=self._provider.embed(query)), *self._fts_queries(query)],
                topk=top_k,
                filter=filter_expr,
                reranker=_zvec.WeightedReRanker([vector_weight, fts_weight, fts_weight]),
            )
        except Exception as exc:
            self._note_fts_unavailable(exc)
            fallback = self._collection.query(
                _zvec.Query(ZvecField.EMBEDDING, vector=self._provider.embed(query)),
                topk=top_k,
                filter=filter_expr,
            )
            return [_vector_hit_to_dict(r) for r in fallback]
        return [_hit_to_dict(r) for r in results]

    def close(self) -> None:
        if self._collection is not None:
            with contextlib.suppress(Exception):
                self._collection.flush()
            self._collection = None

    def is_stale(self, wiki: CompiledWiki, *, embed_attachments: bool = False) -> bool:
        """Return True when the store's last build doesn't match ``wiki``.

        Compares the ids the index covers (:func:`indexable_node_ids`) against
        the ids recorded by the last build.  Missing bookkeeping, an unreadable
        store, or any id set mismatch means the sidecar index is stale and must
        be rebuilt before it can serve the loaded wiki.

        *embed_attachments* must match the value the index was built with — the
        eligibility rule is part of what "up to date" means, so a config flip
        makes the index stale.

        The manifest row also records the embed-text version.  A row written by
        an older composition is stale even when the ids match, because its
        vectors were built from different text.
        """
        expected = indexable_node_ids(wiki, embed_attachments=embed_attachments)
        if not expected:
            return True
        manifest_db = self._manifest_db_path()
        if not manifest_db.is_file():
            return True
        try:
            with open_store(manifest_db) as conn:
                row = conn.execute(
                    f"SELECT embed_version, ids FROM {VECTOR_MANIFEST_TABLE} WHERE id = 1",
                ).fetchone()
        except sqlite3.Error as exc:
            logger.warning("Vector manifest unreadable — treating store as stale: %s", exc)
            return True
        if row is None:
            # No build recorded yet (fresh store, or a manifest left behind by an
            # older KNOX as a loose JSON sidecar we deliberately do not read).
            return True
        if row["embed_version"] != _EMBED_TEXT_VERSION:
            logger.info(
                "Vector manifest embed-text version %r != %d — rebuilding: %s",
                row["embed_version"],
                _EMBED_TEXT_VERSION,
                self.db_file,
            )
            return True
        stored = {str(i) for i in json.loads(row["ids"])}
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

    def _note_fts_unavailable(self, exc: Exception) -> None:
        """Log FTS unavailability once; keeps the query path exception-free."""
        if not self._fts_unavailable:
            self._fts_unavailable = True
            logger.warning(
                "FTS unavailable for %s (%s); lexical/hybrid search disabled until the next rebuild",
                self.db_file,
                exc,
            )

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
                stored_fields = {f.name: _field_fingerprint(f) for f in collection.schema.fields}
                expected_fields = {f.name: _field_fingerprint(f) for f in schema.fields}
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
                        f"{name}: {stored_fields[name][1]}->{expected_fields[name][1]}"
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
                vectors={ZvecField.EMBEDDING: provider.embed(_embed_text(node))},
                fields={
                    ZvecField.NODE_ID: node.id,
                    ZvecField.TITLE: node.title,
                    ZvecField.NODE_TYPE: node.node_type,
                    ZvecField.SOURCE_FILE: source_file,
                    ZvecField.CONTENT_PREVIEW: node.content[: self._PREVIEW_LEN],
                    ZvecField.TAGS: list(node.metadata.get(MetadataField.TAGS, [])),
                    ZvecField.ATTRIBUTES: _attributes_to_list(node.metadata.get(MetadataField.ATTRIBUTES) or {}),
                    ZvecField.ATTR_KEYS: list(node.metadata.get(MetadataField.ATTRIBUTES, {}).keys()),
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

    def _manifest_db_path(self) -> Path:
        """``knox.db`` beside the vector store — home of the index manifest row.

        The vectors directory always sits in the vault's ``.knox/`` next to the
        single non-vector store, so the manifest is a row there instead of a
        loose JSON sidecar: ``.knox/`` holds one state file, not a scattering of
        state files.  The vector branch itself (this directory) stays separate
        and regenerable.
        """
        return self.db_file.parent / KNOX_DB_FILE

    def _write_manifest(self, ids: set[str]) -> None:
        """Record the indexed id set + embed-text version of the last build."""
        try:
            with open_store(self._manifest_db_path()) as conn:
                conn.execute(
                    f"INSERT INTO {VECTOR_MANIFEST_TABLE}(id, embed_version, ids, written_at) "
                    "VALUES (1, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                    "embed_version = excluded.embed_version, ids = excluded.ids, written_at = excluded.written_at",
                    (_EMBED_TEXT_VERSION, json.dumps(sorted(ids)), iso_local()),
                )
        except (sqlite3.Error, OSError) as exc:
            logger.warning(
                "Could not record the vector manifest in %s: %s",
                self._manifest_db_path(),
                exc,
            )

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
        self._write_manifest(keep)
        return len(orphans)

    @staticmethod
    def _fts_queries(query: str) -> list[_zvec.Query]:
        """FTS sub-queries over the two FTS-indexed text fields.

        Lexical recall is preview-bounded: ``content_preview`` is capped at
        ``_PREVIEW_LEN`` chars and the page body lives in the wiki markdown the
        LLM reads for synthesis (memory #844). Lexical search only has to
        *find* the right pages.
        """
        return [
            _zvec.Query(ZvecField.TITLE, fts=_zvec.Fts(match_string=query)),
            _zvec.Query(ZvecField.CONTENT_PREVIEW, fts=_zvec.Fts(match_string=query)),
        ]

    @staticmethod
    def _expected_field_schemas() -> list[_zvec.FieldSchema]:
        """Canonical field list declared by the current knox version (SoT)."""
        return [
            _zvec.FieldSchema(ZvecField.NODE_ID, _zvec.DataType.STRING),
            _zvec.FieldSchema(ZvecField.TITLE, _zvec.DataType.STRING, index_param=_zvec.FtsIndexParam()),
            _zvec.FieldSchema(ZvecField.NODE_TYPE, _zvec.DataType.STRING),
            _zvec.FieldSchema(ZvecField.SOURCE_FILE, _zvec.DataType.STRING),
            _zvec.FieldSchema(ZvecField.CONTENT_PREVIEW, _zvec.DataType.STRING, index_param=_zvec.FtsIndexParam()),
            _zvec.FieldSchema(ZvecField.TAGS, _zvec.DataType.ARRAY_STRING, index_param=_zvec.InvertIndexParam()),
            _zvec.FieldSchema(ZvecField.ATTRIBUTES, _zvec.DataType.ARRAY_STRING, index_param=_zvec.InvertIndexParam()),
            _zvec.FieldSchema(ZvecField.ATTR_KEYS, _zvec.DataType.ARRAY_STRING, index_param=_zvec.InvertIndexParam()),
        ]


"""Version of the embed-text composition used at index time.

Bump this whenever :func:`_embed_text` changes the text it feeds the embedder.
``is_stale`` compares it against the build-time manifest, so an embed-text
change forces a rebuild instead of silently serving vectors built from the old
text — a failure mode with no other signal. ``test_zvec_embed_text.py`` pins the
composition, so changing it fails a test and reminds you to bump this.
"""


def _embed_text(node: WikiNode) -> str:
    """Compose the text embedded for *node*.

    Includes :attr:`WikiNode.description` — the page's first prose sentence,
    derived from its own content and already used for the written index
    frontmatter. Leaving it out made the vector path weigh different evidence
    than the written summary. A node with no prose (empty content, or headings
    only) embeds as ``title`` plus a newline plus ``content``, as before.
    """
    description = node.description
    if description:
        return f"{node.title}\n{description}\n{node.content}"
    return f"{node.title}\n{node.content}"


def indexable_nodes(wiki: CompiledWiki, *, embed_attachments: bool = False) -> list[WikiNode]:
    """The nodes the vector index covers — the single eligibility rule.

    Every builder and :meth:`ZvecWikiStore.is_stale` must agree on this set:
    a staleness check that compares against *all* node ids reports the index
    stale forever as soon as anything is filtered out.

    Args:
        wiki: The compiled wiki (or the markdown bundle loaded from disk).
        embed_attachments: When ``False`` (default), pages derived from
            attachments (``source_asset=True``) stay out of the index.
    """
    if embed_attachments:
        return list(wiki.nodes)
    return [n for n in wiki.nodes if not (n.metadata or {}).get("source_asset", False)]


def indexable_node_ids(wiki: CompiledWiki, *, embed_attachments: bool = False) -> set[str]:
    """Index ids (zvec PK form) for :func:`indexable_nodes`."""
    return {_safe_id(n.id) for n in indexable_nodes(wiki, embed_attachments=embed_attachments)}


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


def _field_fingerprint(field: _zvec.FieldSchema) -> tuple[str, str]:
    """Schema identity used by the rebuild guard: ``(name, "TYPE/index")``.

    Includes the index kind so adding an FTS index to an existing field
    (which leaves name and data_type unchanged) still trips a rebuild.
    """
    return (field.name, f"{field.data_type.name}/{_index_kind(field.index_param)}")


def _index_kind(index_param: _zvec.InvertIndexParam | _zvec.FtsIndexParam | None) -> str:
    """Name of a scalar field's index type, or ``"none"`` when unindexed."""
    if index_param is None:
        return "none"
    return index_param.type.name


def _hit_to_dict(hit: _zvec.Doc) -> dict[str, object]:
    """Search-result dict for FTS/reranked hits — ``score`` kept as returned.

    FTS and reranker scores are already higher-is-better, so no distance
    conversion is applied here (unlike :func:`_vector_hit_to_dict`).
    """
    fields = hit.fields if isinstance(hit.fields, dict) else {}
    return {
        "id": fields.get(ZvecField.NODE_ID, hit.id),
        "score": hit.score if hit.score is not None else 0.0,
        **fields,
    }


def _vector_hit_to_dict(hit: _zvec.Doc) -> dict[str, object]:
    """Search-result dict for a raw vector hit (COSINE distance -> similarity)."""
    fields = hit.fields if isinstance(hit.fields, dict) else {}
    return {
        "id": fields.get(ZvecField.NODE_ID, hit.id),
        "score": _score_from_distance(hit.score),
        **fields,
    }


def _build_filter(
    *,
    tags_any: Sequence[str] | None = None,
    tags_all: Sequence[str] | None = None,
    tags_dnf: Sequence[Sequence[str]] | None = None,
    attributes_any: Sequence[str] | None = None,
    attributes_all: Sequence[str] | None = None,
    attribute_keys: Sequence[str] | None = None,
) -> str | None:
    """Compose a zvec filter string from optional array-membership filters.

    Returns ``None`` when nothing is requested, reproducing the unfiltered
    query. Only ``CONTAIN_ANY``/``CONTAIN_ALL`` are used: ``LIKE`` on an
    ``ARRAY_STRING`` field segfaults zvec 0.7.0.

    ``tags_dnf`` is an OR-of-ANDs (DNF): each inner group becomes
    ``tags CONTAIN_ALL (...)`` and the groups are ``or``-joined inside one
    parenthesised clause. All clause *types* are combined with ``and``, so the
    full query is ``(tags) and (attributes) and (attr-keys) and similarity``.
    ``attribute_keys`` matches bare attribute keys (presence), while
    ``attributes_all``/``attributes_any`` match ``key=value`` tokens.
    """
    clauses: list[str] = []
    if tags_all:
        clauses.append(_contain_clause(ZvecField.TAGS, tags_all, require_all=True))
    if tags_any:
        clauses.append(_contain_clause(ZvecField.TAGS, tags_any, require_all=False))
    if tags_dnf:
        groups = [_contain_clause(ZvecField.TAGS, group, require_all=True) for group in tags_dnf if group]
        if groups:
            clauses.append(f"({' or '.join(groups)})")
    if attributes_all:
        clauses.append(_contain_clause(ZvecField.ATTRIBUTES, attributes_all, require_all=True))
    if attributes_any:
        clauses.append(_contain_clause(ZvecField.ATTRIBUTES, attributes_any, require_all=False))
    if attribute_keys:
        clauses.append(_contain_clause(ZvecField.ATTR_KEYS, attribute_keys, require_all=True))
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
