"""sqlite-vec vector store for semantic search over compiled wikis.

sqlite-vec is an embedded vector extension for SQLite: the vectors, the
lexical index and the filterable metadata all live in one file, so a query
that needs a semantic score, a BM25 score and a tag filter is one SQL
statement against one database.

The store keeps the shape of the store it replaces — ``build_index``,
``search``, ``fts_search``, ``hybrid_search``, ``is_stale``, ``open``,
``close`` — so the retrievers and the CLI keep working unchanged. The rules
that decide *what* is indexed live in :mod:`kompend.rag.vector_store` and are
shared with every backend.

Scores follow the store's convention: **higher is better**. The vector path
returns ``1 - cosine_distance`` (the cosine similarity); the lexical path
returns ``-bm25()`` (SQLite's bm25 is lower-is-better).
"""

# ruff: noqa: S608 — SQL is built from Final[str] table-name constants only.

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import sqlite_vec
from sqlite_vec_client import SQLiteVecClient

import kompend.rag
from kompend.core.models import CompiledWiki, MetadataField, WikiNode
from kompend.core.models.metadata import meta_str_list_dict
from kompend.core.paths import KOMP_DB_FILE, rel_to_vault
from kompend.core.store import open_store
from kompend.core.store.schema import VECTOR_MANIFEST_TABLE
from kompend.core.utils.time import iso_local
from kompend.logging import LOG_SAMPLE_LIMIT, get_logger
from kompend.rag.embedding_base import EmbeddingProvider
from kompend.rag.vector_store import (
    EMBED_TEXT_VERSION,
    VectorField,
    attributes_to_list,
    content_hash,
    embed_node,
    embed_text,
    figure_paths,
    indexable_nodes,
)

#: Identifies this backend in the manifest row and in the store's own meta
#: table. A manifest written by another backend must not validate this store.
_BACKEND: Final[str] = "sqlite-vec"

#: The vec0 table holding the vectors and the scalar fields of a node.
_VEC_ITEMS: Final[str] = "vec_items"

#: The FTS5 table holding the lexical index (title + content preview). The
#: node type and source file ride along as unindexed columns, so a lexical
#: hit carries the same fields as a vector hit without a join.
_VEC_FTS: Final[str] = "vec_fts"

#: Side tables: one row per (node, tag/attribute/key) pair.
_VEC_TAGS: Final[str] = "vec_item_tags"
_VEC_ATTRIBUTES: Final[str] = "vec_item_attributes"
_VEC_ATTR_KEYS: Final[str] = "vec_item_attr_keys"

#: The store's own schema fingerprint — what makes a rebuild necessary.
_VEC_META: Final[str] = "vec_meta"

#: The manifest column naming the embedding model behind the vectors.
#: It is a store column, not a node-metadata key — the manifest is not a
#: :class:`~kompend.core.models.MetadataField` carrier — so it carries its own
#: constant instead of borrowing the metadata member.
_EMBEDDING_MODEL_COLUMN: Final[str] = "embedding_model"

#: The manifest column naming the backend that wrote the vectors.
_BACKEND_COLUMN: Final[str] = "backend"

#: Reciprocal-rank-fusion constant: a document's contribution is
#: ``weight / (RRF_K + rank)``. 60 is the value the previous store's
#: reranker used.
_RRF_K: Final[int] = 60

#: Candidate pool multiplier when filters are applied: the filters run over
#: the vector candidates, so a filtered query needs more than ``top_k``
#: candidates to still return ``top_k`` rows.
_FILTER_OVERSAMPLE: Final[int] = 4

#: Lower bound for the candidate pool of a filtered query.
_FILTER_POOL_MIN: Final[int] = 50

#: The word characters of a query — everything else is an FTS5 operator.
_TERM_RE: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)

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
        """(Re)create the store's tables.

        The index is rebuilt from scratch on every build: the corpus is small,
        and a fresh table cannot carry rows the current wiki no longer has.
        """
        conn = self._require_conn()
        dim = self._require_provider().dim
        conn.execute(f"DROP TABLE IF EXISTS {_VEC_ITEMS}")
        conn.execute(f"DROP TABLE IF EXISTS {_VEC_FTS}")
        conn.execute(f"DROP TABLE IF EXISTS {_VEC_TAGS}")
        conn.execute(f"DROP TABLE IF EXISTS {_VEC_ATTRIBUTES}")
        conn.execute(f"DROP TABLE IF EXISTS {_VEC_ATTR_KEYS}")
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE {_VEC_ITEMS} USING vec0(
                id              TEXT PRIMARY KEY,
                embedding       FLOAT[{dim}] distance_metric=cosine,
                title           TEXT,
                node_type       TEXT,
                source_file     TEXT,
                content_preview TEXT
            )
            """
        )
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE {_VEC_FTS} USING fts5(
                id UNINDEXED,
                title,
                content_preview,
                node_type UNINDEXED,
                source_file UNINDEXED,
                tokenize='unicode61'
            )
            """
        )
        conn.execute(f"CREATE TABLE {_VEC_TAGS} (item_id TEXT, tag TEXT, PRIMARY KEY (item_id, tag))")
        conn.execute(f"CREATE TABLE {_VEC_ATTRIBUTES} (item_id TEXT, token TEXT, PRIMARY KEY (item_id, token))")
        conn.execute(f"CREATE TABLE {_VEC_ATTR_KEYS} (item_id TEXT, key TEXT, PRIMARY KEY (item_id, key))")
        conn.execute(f"CREATE INDEX ix_vec_tags_token ON {_VEC_TAGS} (tag)")
        conn.execute(f"CREATE INDEX ix_vec_attributes_token ON {_VEC_ATTRIBUTES} (token)")
        conn.execute(f"CREATE INDEX ix_vec_attr_keys_key ON {_VEC_ATTR_KEYS} (key)")
        conn.execute(f"CREATE TABLE IF NOT EXISTS {_VEC_META} (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(f"DELETE FROM {_VEC_META}")
        self._write_meta(dim)
        conn.commit()

    def _index_nodes(
        self,
        nodes: list[WikiNode],
        batch_size: int,
        *,
        base_dir: Path | None = None,
    ) -> int:
        """Embed and write *nodes* into the store."""
        provider = self._require_provider()
        conn = self._require_conn()
        indexed_count = 0
        figure_nodes = 0
        figure_embedded = 0
        degraded: list[str] = []

        items: list[tuple[Any, ...]] = []
        fts_rows: list[tuple[str, str, str, str, str]] = []
        tag_rows: list[tuple[str, str]] = []
        attribute_rows: list[tuple[str, str]] = []
        attr_key_rows: list[tuple[str, str]] = []

        def flush() -> int:
            """Write the pending rows and clear the buffers."""
            if not items:
                return 0
            count = len(items)
            conn.executemany(
                f"INSERT INTO {_VEC_ITEMS} (id, embedding, title, node_type, source_file, content_preview) VALUES (?, ?, ?, ?, ?, ?)",
                items,
            )
            conn.executemany(
                f"INSERT INTO {_VEC_FTS} (id, title, content_preview, node_type, source_file) VALUES (?, ?, ?, ?, ?)",
                fts_rows,
            )
            conn.executemany(f"INSERT INTO {_VEC_TAGS} (item_id, tag) VALUES (?, ?)", tag_rows)
            conn.executemany(f"INSERT INTO {_VEC_ATTRIBUTES} (item_id, token) VALUES (?, ?)", attribute_rows)
            conn.executemany(f"INSERT INTO {_VEC_ATTR_KEYS} (item_id, key) VALUES (?, ?)", attr_key_rows)
            items.clear()
            fts_rows.clear()
            tag_rows.clear()
            attribute_rows.clear()
            attr_key_rows.clear()
            return count

        for node in nodes:
            raw_source = str(node.metadata.get(MetadataField.SOURCE, "")) if node.metadata else ""
            source_file = rel_to_vault(raw_source, base_dir) if base_dir else raw_source
            text = embed_text(node)
            vector, note = embed_node(provider, node, text, base_dir=base_dir)
            if figure_paths(node):
                figure_nodes += 1
                if note is None:
                    figure_embedded += 1
                else:
                    degraded.append(f"{node.id}: {note}")
            attributes = meta_str_list_dict(node.metadata.get(MetadataField.ATTRIBUTES))
            preview = node.content[: self._PREVIEW_LEN]
            items.append(
                (
                    node.id,
                    sqlite_vec.serialize_float32(vector),
                    node.title,
                    node.node_type.value,
                    source_file,
                    preview,
                )
            )
            fts_rows.append((node.id, node.title, preview, node.node_type.value, source_file))
            for tag in node.metadata.get(MetadataField.TAGS, []):
                tag_rows.append((node.id, str(tag)))
            for token in attributes_to_list(attributes):
                attribute_rows.append((node.id, token))
            for key in attributes:
                attr_key_rows.append((node.id, key))

            if len(items) >= batch_size:
                indexed_count += flush()
        indexed_count += flush()
        conn.commit()

        if figure_nodes:
            logger.info(
                "Embedded %d of %d node(s) with figures; %d node(s) were indexed from text alone",
                figure_embedded,
                figure_nodes,
                figure_nodes - figure_embedded,
            )
        if degraded:
            logger.warning(
                "%d node(s) with figures were indexed without their image: %s",
                len(degraded),
                "; ".join(degraded[:LOG_SAMPLE_LIMIT]),
            )
        logger.info("Finished indexing %d nodes in the vector store at %s", indexed_count, self.db_file)
        return indexed_count

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
        return [_hit_to_dict(hit) for hit in hits]

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
        fused = _fuse(((title_hits, 1.0), (preview_hits, 1.0)), top_k)
        return [_hit_to_dict(hit) for hit in fused]

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
            return [_hit_to_dict(hit) for hit in vector_hits]

        fused = _fuse(
            (
                (vector_hits, vector_weight),
                (title_hits, fts_weight),
                (preview_hits, fts_weight),
            ),
            top_k,
        )
        return [_hit_to_dict(hit) for hit in fused]

    def _vector_hits(
        self,
        query: str,
        top_k: int,
        *,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> list[_Hit]:
        """Run the vector query and return the hits in rank order."""
        provider = self._require_provider()
        conn = self._require_conn()
        clause, params = _filter_clause(
            "v",
            tags_all=tags_all,
            tags_dnf=tags_dnf,
            attributes_all=attributes_all,
            attribute_keys=attribute_keys,
        )
        pool = _candidate_pool(top_k, filtered=bool(clause))
        sql = f"""
            SELECT v.id, v.title, v.node_type, v.source_file, v.content_preview, v.distance
            FROM {_VEC_ITEMS} v
            WHERE v.embedding MATCH ? AND v.k = ?
            {clause}
            ORDER BY v.distance
        """
        rows = conn.execute(
            sql,
            (sqlite_vec.serialize_float32(provider.embed(query)), pool, *params),
        ).fetchall()
        return [
            _Hit(
                id=str(row[0]),
                score=_score_from_distance(row[5]),
                title=row[1] or "",
                node_type=row[2] or "",
                source_file=row[3] or "",
                content_preview=row[4] or "",
            )
            for row in rows
        ]

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
    ) -> list[_Hit]:
        """Run one lexical query and return the hits in rank order.

        The query matches a single indexed column — ``title`` or
        ``content_preview``. FTS5's ``MATCH`` operator needs the *real* table
        name on its left for a cross-column match, so a column-scoped match
        (``f.title MATCH ?``) is the form that works with an alias.
        """
        conn = self._require_conn()
        match = _fts_match_expression(query)
        if match is None:
            return []
        clause, params = _filter_clause(
            "f",
            tags_all=tags_all,
            tags_dnf=tags_dnf,
            attributes_all=attributes_all,
            attribute_keys=attribute_keys,
        )
        sql = f"""
            SELECT f.id, f.title, f.content_preview, f.node_type, f.source_file, bm25({_VEC_FTS}) AS rank
            FROM {_VEC_FTS} f
            WHERE f.{column} MATCH ? {clause}
            ORDER BY rank
            LIMIT ?
        """
        rows = conn.execute(sql, (match, *params, top_k)).fetchall()
        return [
            _Hit(
                id=str(row[0]),
                score=-float(row[5]),
                title=row[1] or "",
                node_type=row[3] or "",
                source_file=row[4] or "",
                content_preview=row[2] or "",
            )
            for row in rows
        ]

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

    @staticmethod
    def _read_manifest(manifest_db: Path) -> sqlite3.Row | None:
        """The manifest row of the last build, or ``None`` when there is none."""
        try:
            with open_store(manifest_db) as conn:
                return conn.execute(
                    f"SELECT embed_version, {_EMBEDDING_MODEL_COLUMN}, {_BACKEND_COLUMN}, ids, content_hashes FROM {VECTOR_MANIFEST_TABLE} WHERE id = 1",
                ).fetchone()
        except sqlite3.Error as exc:
            logger.warning("Vector manifest unreadable — treating store as stale: %s", exc)
            return None

    def _manifest_mismatch(self, row: sqlite3.Row) -> str | None:
        """Why the manifest does not describe the current vectors, or ``None``.

        A row written before the hashes were tracked is not a mismatch here —
        the content comparison catches that.
        """
        if row[_BACKEND_COLUMN] != _BACKEND:
            return f"backend {row[_BACKEND_COLUMN]!r} != {_BACKEND!r}"
        if row["embed_version"] != EMBED_TEXT_VERSION:
            return f"embed-text version {row['embed_version']!r} != {EMBED_TEXT_VERSION!r}"
        current_model = self._provider.model_id if self._provider is not None else ""
        if row[_EMBEDDING_MODEL_COLUMN] != current_model:
            return f"embedding model {row[_EMBEDDING_MODEL_COLUMN]!r} != {current_model!r}"
        return None

    def _content_changed(self, recorded: str, nodes: list[WikiNode]) -> bool:
        """Compare recorded content hashes against *nodes*.

        The id check alone cannot see an edited page: a `lint --fix`, a hand
        edit, or a recompile that produced different prose leaves the id set
        unchanged while the vectors describe text search no longer returns.
        Hashing the indexed payload closes that blind spot.

        A row written before hashes were tracked (empty mapping) is stale by
        definition — one rebuild records them.
        """
        stored: dict[str, str] = json.loads(recorded) if recorded else {}
        if not stored:
            logger.info("Vector manifest has no content hashes (pre-tracking build) — rebuilding: %s", self.db_file)
            return True
        expected = {node.id: content_hash(node) for node in nodes}
        changed = sorted(node_id for node_id, digest in expected.items() if stored.get(node_id) != digest)
        if not changed:
            return False
        logger.info(
            "Vector index content changed for %d of %d node(s), e.g. %s — rebuilding: %s",
            len(changed),
            len(expected),
            changed[:LOG_SAMPLE_LIMIT],
            self.db_file,
        )
        return True

    # ── Manifest ─────────────────────────────────────────────────────────

    def _manifest_db_path(self) -> Path:
        """``kompend.db`` beside the vector store — home of the index manifest row.

        The vectors file always sits in the vault's ``.kompend/`` next to the
        single non-vector store, so the manifest is a row there instead of a
        loose JSON sidecar: ``.kompend/`` holds one state file, not a scattering
        of state files.  The vector branch itself (this file) stays separate
        and regenerable.
        """
        return self.db_file.parent / KOMP_DB_FILE

    def _write_manifest(self, nodes: list[WikiNode]) -> None:
        """Record the indexed id set, content hashes, embed-text version, model and backend."""
        ids = sorted(node.id for node in nodes)
        hashes = {node.id: content_hash(node) for node in nodes}
        model_id = self._provider.model_id if self._provider is not None else ""
        try:
            with open_store(self._manifest_db_path()) as conn:
                conn.execute(
                    f"INSERT INTO {VECTOR_MANIFEST_TABLE}"
                    f"(id, embed_version, embedding_model, {_BACKEND_COLUMN}, ids, content_hashes, written_at) "
                    "VALUES (1, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                    "embed_version = excluded.embed_version, embedding_model = excluded.embedding_model, "
                    f"{_BACKEND_COLUMN} = excluded.{_BACKEND_COLUMN}, "
                    "ids = excluded.ids, content_hashes = excluded.content_hashes, written_at = excluded.written_at",
                    (
                        EMBED_TEXT_VERSION,
                        model_id,
                        _BACKEND,
                        json.dumps(ids),
                        json.dumps(hashes, sort_keys=True),
                        iso_local(),
                    ),
                )
        except (sqlite3.Error, OSError) as exc:
            logger.warning(
                "Could not record the vector manifest in %s: %s",
                self._manifest_db_path(),
                exc,
            )

    # ── Store identity ───────────────────────────────────────────────────

    def _write_meta(self, dim: int) -> None:
        """Record the store's schema fingerprint."""
        conn = self._require_conn()
        fingerprint = _schema_fingerprint(dim)
        conn.executemany(
            f"INSERT OR REPLACE INTO {_VEC_META} (key, value) VALUES (?, ?)",
            [
                ("backend", _BACKEND),
                ("dim", str(dim)),
                ("schema", fingerprint),
                ("embed_version", str(EMBED_TEXT_VERSION)),
                ("model_id", self._provider.model_id if self._provider is not None else ""),
            ],
        )

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
        try:
            rows = dict(conn.execute(f"SELECT key, value FROM {_VEC_META}").fetchall())
        except sqlite3.Error:
            return False
        dim = self._provider.dim if self._provider is not None else 0
        return rows.get("backend") == _BACKEND and rows.get("dim") == str(dim) and rows.get("schema") == _schema_fingerprint(dim)

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
                (_VEC_ITEMS,),
            ).fetchone()
        except sqlite3.Error:
            return False
        return row is not None


class _Hit:
    """One search result, in the shape the retrievers consume."""

    __slots__ = ("content_preview", "id", "node_type", "score", "source_file", "title")

    def __init__(
        self,
        *,
        id: str,
        score: float,
        title: str = "",
        node_type: str = "",
        source_file: str = "",
        content_preview: str = "",
    ) -> None:
        self.id = id
        self.score = score
        self.title = title
        self.node_type = node_type
        self.source_file = source_file
        self.content_preview = content_preview


def _hit_to_dict(hit: _Hit) -> dict[str, Any]:
    """Search-result dict — the contract the retrievers read."""
    return {
        "id": hit.id,
        "score": hit.score,
        VectorField.NODE_ID: hit.id,
        VectorField.TITLE: hit.title,
        VectorField.NODE_TYPE: hit.node_type,
        VectorField.SOURCE_FILE: hit.source_file,
        VectorField.CONTENT_PREVIEW: hit.content_preview,
    }


def _score_from_distance(distance: float | None) -> float:
    """Convert a cosine distance to a similarity score (higher = better).

    The embedding field is indexed with ``distance_metric=cosine``, whose
    distance is in ``[0, 2]``. Retrievers rank descending by ``score``, so
    expose ``1 - distance``: for L2-normalised embeddings this equals the
    cosine similarity.
    """
    if distance is None:
        return 0.0
    return 1.0 - float(distance)


def _candidate_pool(top_k: int, *, filtered: bool) -> int:
    """How many vector candidates to pull before the filters are applied."""
    if not filtered:
        return top_k
    return max(_FILTER_POOL_MIN, top_k * _FILTER_OVERSAMPLE)


def _fts_match_expression(query: str) -> str | None:
    """Turn a user query into an FTS5 match expression, or ``None``.

    FTS5's query language is not a plain-text search: ``-`` is the NOT
    operator, ``:`` introduces a column filter, and ``"`` quotes a phrase. A
    user query like ``context-window`` therefore fails to parse ("no such
    column: window"). Splitting the query into its word characters and
    quoting each term keeps every term literal and every operator out of the
    expression.
    """
    terms = _TERM_RE.findall(query)
    if not terms:
        return None
    return " ".join(f'"{term}"' for term in terms)


def _filter_clause(
    alias: str,
    *,
    tags_all: Sequence[str] | None = None,
    tags_dnf: Sequence[Sequence[str]] | None = None,
    attributes_all: Sequence[str] | None = None,
    attribute_keys: Sequence[str] | None = None,
) -> tuple[str, list[str]]:
    """Compose the SQL for the requested filters, with bound parameters.

    *alias* is the table alias the filter correlates against (``v`` for the
    vector query, ``f`` for the lexical one).

    ``tags_all``/``attributes_all``/``attribute_keys`` mean "the node has
    *every* one of these" — one ``EXISTS`` per value, ANDed.
    ``tags_dnf`` is an OR of ANDs (DNF): each group is a set of tags the node
    must all have, and the groups are OR-joined inside one parenthesised
    clause.

    Every value travels as a bound parameter — never as SQL text.
    """
    clauses: list[str] = []
    params: list[str] = []

    for tag in tags_all or ():
        clauses.append(f"EXISTS (SELECT 1 FROM {_VEC_TAGS} t WHERE t.item_id = {alias}.id AND t.tag = ?)")
        params.append(str(tag))

    groups: list[str] = []
    group_params: list[str] = []
    for group in tags_dnf or ():
        members = [str(tag) for tag in group]
        if not members:
            continue
        group_clauses = [f"EXISTS (SELECT 1 FROM {_VEC_TAGS} t WHERE t.item_id = {alias}.id AND t.tag = ?)" for _ in members]
        groups.append(" AND ".join(group_clauses))
        group_params.extend(members)
    if groups:
        clauses.append(f"({' OR '.join(groups)})")
        params.extend(group_params)

    for token in attributes_all or ():
        clauses.append(f"EXISTS (SELECT 1 FROM {_VEC_ATTRIBUTES} a WHERE a.item_id = {alias}.id AND a.token = ?)")
        params.append(str(token))

    for key in attribute_keys or ():
        clauses.append(f"EXISTS (SELECT 1 FROM {_VEC_ATTR_KEYS} k WHERE k.item_id = {alias}.id AND k.key = ?)")
        params.append(str(key))

    if not clauses:
        return "", []
    return "AND " + " AND ".join(clauses), params


def _fuse(
    ranked_lists: Sequence[tuple[list[_Hit], float]],
    top_k: int,
) -> list[_Hit]:
    """Weighted reciprocal rank fusion over the ranked lists.

    A document's score is the sum of ``weight / (RRF_K + rank)`` over the
    lists it appears in; rank is 1-based. Documents missing from a list
    contribute nothing from it.
    """
    scores: dict[str, float] = {}
    best: dict[str, _Hit] = {}
    for hits, weight in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + weight / (_RRF_K + rank)
            current = best.get(hit.id)
            if current is None or hit.score > current.score:
                best[hit.id] = hit
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[_Hit] = []
    for node_id, score in ordered[:top_k]:
        hit = best[node_id]
        fused.append(
            _Hit(
                id=hit.id,
                score=score,
                title=hit.title,
                node_type=hit.node_type,
                source_file=hit.source_file,
                content_preview=hit.content_preview,
            )
        )
    return fused


def _schema_fingerprint(dim: int) -> str:
    """Identity of the store's schema — a change forces a rebuild."""
    return f"vec0:1:{dim}:{_VEC_ITEMS}/{_VEC_FTS}/{_VEC_TAGS}/{_VEC_ATTRIBUTES}/{_VEC_ATTR_KEYS}"
