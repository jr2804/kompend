"""sqlite-vec vector store for semantic search over compiled wikis.

sqlite-vec is an embedded vector extension for SQLite: the vectors, the
lexical index and the filterable metadata all live in one file, so a query
that needs a semantic score, a BM25 score and a tag filter is one SQL
statement against one database.

The store keeps the shape of the store it replaces — ``build_index``,
``search``, ``fts_search``, ``hybrid_search``, ``is_stale``, ``open``,
``close`` — so the retrievers and the CLI keep working unchanged. The rules
that decide *what* is indexed live in :mod:`kompend.rag.vector_store` and are
shared with every backend. The DDL and its fingerprint live in
:mod:`kompend.rag.sqlite_vec.schema`; the manifest row and the staleness
rules in :mod:`kompend.rag.sqlite_vec.manifest`.

Scores follow the store's convention: **higher is better**. The vector path
returns ``1 - distance`` (the cosine similarity); the lexical path
returns ``-bm25()`` (SQLite's bm25 is lower-is-better).
"""

# ruff: noqa: S608 — SQL is built from Final[str] table-name constants only.

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlite_vec_client import SQLiteVecClient

import kompend.rag
from kompend.core.models import CompiledWiki, WikiNode
from kompend.logging import get_logger
from kompend.rag.embedding_base import EmbeddingProvider
from kompend.rag.sqlite_vec.index import index_nodes
from kompend.rag.sqlite_vec.manifest import (
    content_changed,
    manifest_db_path,
    manifest_mismatch,
    read_manifest,
    write_manifest,
)
from kompend.rag.sqlite_vec.query import (
    Hit,
    fts_ranked,
    fuse,
    hit_to_dict,
    vector_hits,
)
from kompend.rag.sqlite_vec.schema import (
    VEC_ITEMS,
    create_schema,
    read_meta,
    write_meta,
)
from kompend.rag.vector_store import (
    indexable_nodes,
)

logger = get_logger(__name__)


class SqliteVecWikiStore:
    """Sidecar vector store for semantic search over compiled wikis.

    The store is dormant until :meth:`build_index` is called or an existing
    database is opened with :meth:`open`.
    """

    _PREVIEW_LEN: int = 1000

    def __init__(
        self,
        db_file: str | Path = "./.kompend/vectors.db",
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.db_file = Path(db_file)
        self._provider = embedding_provider
        self._conn: sqlite3.Connection | None = None
        self._fts_unavailable = False

    @property
    def available(self) -> bool:
        """True when the store is open and its vector table exists."""
        return self._conn is not None and self._has_items_table()

    @property
    def embedding_provider(self) -> EmbeddingProvider | None:
        """The provider behind the vectors — public so callers need no internals."""
        return self._provider

    # ── Building ─────────────────────────────────────────────────────────

    def build_index(
        self,
        wiki: CompiledWiki,
        embedding_model: str | None = None,
        *,
        batch_size: int = 10,
        base_dir: Path | None = None,
        embed_attachments: bool = False,
    ) -> int:
        """Index compiled wiki nodes into the vector store.

        If no provider was set at construction, ``embedding_model`` must be
        given to create one via :func:`make_provider`.

        Coverage comes from :func:`indexable_nodes` — the same rule
        :meth:`is_stale` compares against — so what gets embedded and what the
        manifest records can never disagree. Callers pass the whole wiki, not
        a pre-filtered copy: filtering in two places is how the manifest ended
        up recording ids the staleness check did not expect, rebuilding the
        index on the next query.
        """
        nodes = indexable_nodes(wiki, embed_attachments=embed_attachments)
        if not nodes:
            return 0

        if self._provider is None:
            if not embedding_model:
                raise ValueError("embedding_model is required when no provider was injected")
            # Resolved through the package's lazy export: the pydantic-ai
            # provider stack must not load when this module is imported.
            self._provider = kompend.rag.make_provider(embedding_model)

        self._open_for_write()
        count = self._index_nodes(nodes, batch_size, base_dir=base_dir)
        self._write_manifest(nodes)
        return count

    # ── Querying ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search indexed wiki content by semantic similarity.

        Optional keyword filters are pushed down into SQL: the tag and
        attribute side tables are consulted with ``EXISTS`` subqueries, so the
        filter is evaluated per candidate instead of scanning the corpus.
        Omitting them reproduces the unfiltered result set.

        The returned ``score`` is the cosine similarity (``1 - distance``),
        higher is better.
        """
        if not self.available or self._provider is None:
            return []
        hits = self._vector_hits(
            query,
            top_k,
            tags_all=tags_all,
            tags_dnf=tags_dnf,
            attributes_all=attributes_all,
            attribute_keys=attribute_keys,
        )
        return [hit_to_dict(hit) for hit in hits]

    def fts_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        """Lexical (BM25) search over the FTS-indexed ``title`` + ``content_preview``.

        Both lexical fields are searched and fused with reciprocal rank
        fusion, so a page matches on its title or on its preview.

        Returns the same dict shape as :meth:`search`; ``score`` is the fused
        BM25 relevance (higher = better, no distance conversion). Returns ``[]``
        when the store is closed or the lexical index cannot answer the query
        (logged once) — never raises into the query path.
        """
        if not self.available:
            return []
        filters = {
            "tags_all": tags_all,
            "tags_dnf": tags_dnf,
            "attributes_all": attributes_all,
            "attribute_keys": attribute_keys,
        }
        try:
            title_hits = self._fts_ranked(query, top_k, column="title", **filters)
            preview_hits = self._fts_ranked(query, top_k, column="content_preview", **filters)
        except sqlite3.Error as exc:
            self._note_fts_unavailable(exc)
            return []
        fused = fuse(((title_hits, 1.0), (preview_hits, 1.0)), top_k)
        return [hit_to_dict(hit) for hit in fused]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
        vector_weight: float = 1.0,
        fts_weight: float = 0.3,
    ) -> list[dict[str, object]]:
        """Fuse vector similarity and lexical BM25.

        The two signals are retrieved separately — the vector candidate pool
        and the two lexical indexes (``title``, ``content_preview``) — and
        fused with weighted reciprocal rank fusion: a document's score is the
        sum of ``weight / (60 + rank)`` over the lists it appears in. The
        weights are ``[vector_weight, fts_weight, fts_weight]``.

        When the lexical index cannot answer, the vector-only result is
        returned (logged once).
        """
        if not self.available or self._provider is None:
            return []
        filters = {
            "tags_all": tags_all,
            "tags_dnf": tags_dnf,
            "attributes_all": attributes_all,
            "attribute_keys": attribute_keys,
        }
        vector_hits = self._vector_hits(query, top_k, **filters)
        try:
            title_hits = self._fts_ranked(query, top_k, column="title", **filters)
            preview_hits = self._fts_ranked(query, top_k, column="content_preview", **filters)
        except sqlite3.Error as exc:
            self._note_fts_unavailable(exc)
            return [hit_to_dict(hit) for hit in vector_hits]

        fused = fuse(
            (
                (vector_hits, vector_weight),
                (title_hits, fts_weight),
                (preview_hits, fts_weight),
            ),
            top_k,
        )
        return [hit_to_dict(hit) for hit in fused]

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the store file.

        SQLite holds the file open until the connection is closed, so a later
        rebuild of the same path races the open handle.
        """
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None

    # ── Staleness ────────────────────────────────────────────────────────

    def is_stale(self, wiki: CompiledWiki, *, embed_attachments: bool = False) -> bool:
        """Return True when the store's last build doesn't match ``wiki``.

        Compares the ids the index covers (:func:`indexable_nodes`) against the
        ids recorded by the last build, then their content hashes
        (:func:`content_hash`). Missing bookkeeping, an unreadable store, or
        any mismatch means the sidecar index is stale and must be rebuilt
        before it can serve the loaded wiki. ``kompend query`` calls this on
        every open, so a false positive re-embeds the whole corpus: a
        build/check disagreement must not cost a full rebuild per query.

        The content comparison catches the other direction: an edited page keeps
        its id, so an ids-only check would keep serving vectors that describe
        text the wiki no longer contains.

        *embed_attachments* must match the value the index was built with — the
        eligibility rule is part of what "up to date" means, so a config flip
        makes the index stale.

        The manifest row also records the embed-text version and the backend.
        A row written by an older composition is stale even when the ids and
        hashes match, because its vectors were built from different text; a
        row written by another backend is stale because those vectors are not
        readable here at all.
        """
        nodes = indexable_nodes(wiki, embed_attachments=embed_attachments)
        expected = {node.id for node in nodes}
        if not expected:
            return True
        if not self.db_file.exists():
            return True
        manifest_db = self._manifest_db_path()
        if not manifest_db.is_file():
            return True
        row = self._read_manifest(manifest_db)
        if row is None:
            return True
        reason = self._manifest_mismatch(row)
        if reason is not None:
            logger.info("Vector manifest %s — rebuilding: %s", reason, self.db_file)
            return True
        stored = {str(i) for i in json.loads(row["ids"])}
        if stored != expected:
            return True
        return self._content_changed(row["content_hashes"], nodes)

    @classmethod
    def open(
        cls,
        db_file: str | Path,
        embedding_model: str | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> SqliteVecWikiStore:
        """Open an existing store for querying.

        The store must have been created previously with :meth:`build_index`.
        Either pass ``embedding_provider`` directly, or pass
        ``embedding_model`` to create one via :func:`make_provider`.
        """
        provider = embedding_provider or kompend.rag.make_provider(embedding_model or "")
        store = cls(db_file=db_file, embedding_provider=provider)
        if not store.db_file.exists():
            logger.warning("No vector store found at %s. Rebuild with build_index().", store.db_file)
            return store
        try:
            store._conn = store._connect()
        except Exception as exc:
            logger.warning("Could not open the vector store at %s: %s", store.db_file, exc, exc_info=True)
            return store
        if not store._meta_matches():
            logger.warning(
                "Vector store at %s was written by another schema or backend — rebuild it",
                store.db_file,
            )
            store.close()
            return store
        return store

    def _open_for_write(self) -> None:
        """Open (or recreate) the store file and lay out the schema."""
        self.close()
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        if self.db_file.exists() and not self._meta_matches():
            # The file holds another schema, another dimension or another
            # backend's vectors. It is derived data: drop it and rebuild.
            logger.info("Vector store at %s has a stale schema — rebuilding", self.db_file)
            self._drop_store_file()
        self._conn = self._connect()
        self._create_schema()

    def _drop_store_file(self) -> None:
        """Remove the store file and its write-ahead-log siblings."""
        self.db_file.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(f"{self.db_file}{suffix}").unlink(missing_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Open the store file with the sqlite-vec extension loaded.

        The extension must be loaded on **every** connection that touches the
        vec0 tables — an unloaded connection does not even see them.
        """
        return SQLiteVecClient.create_connection(str(self.db_file))

    def _create_schema(self) -> None:
        """(Re)create the store's tables."""
        conn = self._require_conn()
        dim = self._require_provider().dim
        model_id = self._provider.model_id if self._provider is not None else ""
        create_schema(conn, dim, model_id)

    def _index_nodes(
        self,
        nodes: list[WikiNode],
        batch_size: int,
        *,
        base_dir: Path | None = None,
    ) -> int:
        """Embed and write *nodes* into the store."""
        return index_nodes(
            self._require_conn(),
            self._require_provider(),
            nodes,
            batch_size,
            preview_len=self._PREVIEW_LEN,
            db_file=self.db_file,
            base_dir=base_dir,
        )

    def _vector_hits(
        self,
        query: str,
        top_k: int,
        *,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> list[Hit]:
        """Run the vector query and return the hits in rank order."""
        return vector_hits(
            self._require_conn(),
            self._require_provider().embed(query),
            top_k,
            tags_all=tags_all,
            tags_dnf=tags_dnf,
            attributes_all=attributes_all,
            attribute_keys=attribute_keys,
        )

    def _fts_ranked(
        self,
        query: str,
        top_k: int,
        *,
        column: str = "title",
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> list[Hit]:
        """Run one lexical query and return the hits in rank order."""
        return fts_ranked(
            self._require_conn(),
            query,
            top_k,
            column=column,
            tags_all=tags_all,
            tags_dnf=tags_dnf,
            attributes_all=attributes_all,
            attribute_keys=attribute_keys,
        )

    def _note_fts_unavailable(self, exc: Exception) -> None:
        """Log a failing lexical query once; keeps the query path exception-free.

        A query FTS5 cannot parse answers with no lexical results, not with an
        error: the caller sees an empty lexical list and the vector path still
        works.
        """
        if not self._fts_unavailable:
            self._fts_unavailable = True
            logger.warning(
                "Lexical search failed for %s (%s); that query answers with no lexical results",
                self.db_file,
                exc,
            )

    def _manifest_mismatch(self, row: sqlite3.Row) -> str | None:
        """Why the manifest does not describe the current vectors, or ``None``."""
        current_model = self._provider.model_id if self._provider is not None else ""
        return manifest_mismatch(row, current_model)

    def _content_changed(self, recorded: str, nodes: list[WikiNode]) -> bool:
        """Whether the recorded hashes differ from *nodes*."""
        return content_changed(recorded, nodes, self.db_file)

    # ── Manifest ─────────────────────────────────────────────────────────

    def _manifest_db_path(self) -> Path:
        """``kompend.db`` beside the vector store — home of the index manifest row."""
        return manifest_db_path(self.db_file)

    def _write_manifest(self, nodes: list[WikiNode]) -> None:
        """Record the indexed id set, hashes, embed-text version, model and backend."""
        model_id = self._provider.model_id if self._provider is not None else ""
        write_manifest(self.db_file, nodes, model_id)

    # ── Store identity ───────────────────────────────────────────────────

    def _write_meta(self, dim: int) -> None:
        """Record the store's schema fingerprint."""
        conn = self._require_conn()
        model_id = self._provider.model_id if self._provider is not None else ""
        write_meta(conn, dim, model_id)

    def _meta_matches(self) -> bool:
        """True when the store file holds this schema, dimension and backend."""
        if self._conn is not None:
            return self._read_meta(self._conn)
        return self._read_meta_without_connection()

    def _read_meta_without_connection(self) -> bool:
        """Read the fingerprint from a file that is not open yet."""
        try:
            conn = self._connect()
        except Exception:
            return False
        try:
            return self._read_meta(conn)
        finally:
            conn.close()

    def _read_meta(self, conn: sqlite3.Connection) -> bool:
        dim = self._provider.dim if self._provider is not None else 0
        return read_meta(conn, dim)

    # ── Internals ────────────────────────────────────────────────────────

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Vector store is not open")
        return self._conn

    def _require_provider(self) -> EmbeddingProvider:
        if self._provider is None:
            raise RuntimeError("Embedding provider not configured. Call build_index() first.")
        return self._provider

    def _has_items_table(self) -> bool:
        if self._conn is None:
            return False
        try:
            row = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE name = ?",
                (VEC_ITEMS,),
            ).fetchone()
        except sqlite3.Error:
            return False
        return row is not None

    @staticmethod
    def _read_manifest(manifest_db: Path) -> sqlite3.Row | None:
        """The manifest row of the last build, or ``None`` when there is none."""
        return read_manifest(manifest_db)
