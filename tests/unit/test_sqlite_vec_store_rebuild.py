"""Rebuild guards for the sqlite-vec wiki and memory stores.

The vector index is derived data: whenever the store cannot describe the
vectors it holds — a different dimension, a different schema, another
backend's manifest — the file is dropped and rebuilt rather than reopened.
"""

# ruff: noqa: S608 — SQL is built from Final[str] table-name constants only.

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest
import sqlite_vec
from sqlite_vec_client import SQLiteVecClient

from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.models.enums import NodeType
from kompend.core.store.schema import VECTOR_MANIFEST_TABLE
from kompend.memory.store import MemoryStore
from kompend.rag.embedding_base import EmbeddingProvider
from kompend.rag.sqlite_vec_store import SqliteVecWikiStore


class StubProvider(EmbeddingProvider):
    def __init__(self, dim: int, model_id: str = "") -> None:
        self._dim = dim
        self._model_id = model_id

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed(self, text: str) -> list[float]:
        return [0.5] * self._dim


def test_wiki_store_is_stale_when_the_embedding_model_changed(tmp_path: Path) -> None:
    """Vectors from one model do not describe the space of another.

    Switching the embedding model (e.g. to a vision-capable one) must
    invalidate the index — otherwise the queries keep hitting vectors built
    by the old model.
    """
    db = tmp_path / "vectors.db"
    wiki = _wiki()

    store = SqliteVecWikiStore(db_file=db, embedding_provider=StubProvider(4, "ollama:embeddinggemma-2:270m"))
    store.build_index(wiki)
    store.close()

    same = SqliteVecWikiStore(db_file=db, embedding_provider=StubProvider(4, "ollama:embeddinggemma-2:270m"))
    assert same.is_stale(wiki) is False
    same.close()

    other = SqliteVecWikiStore(db_file=db, embedding_provider=StubProvider(4, "ollama:embeddinggemma-2:440m"))
    assert other.is_stale(wiki) is True
    other.close()


def test_wiki_store_rebuilds_on_dim_change(tmp_path: Path) -> None:
    """A different vector dimension is a different store — rebuild, not reopen."""
    db = tmp_path / "vectors.db"

    store = SqliteVecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    assert store.build_index(_wiki()) == 1
    assert _store_dim(db) == "4"
    store.close()

    # Same path, model now produces 8 dims — must rebuild, not reopen.
    store = SqliteVecWikiStore(db_file=db, embedding_provider=StubProvider(8))
    assert store.build_index(_wiki()) == 1
    assert _store_dim(db) == "8"
    assert store.search("q")  # insert + query at 8 dims works
    store.close()


def test_memory_store_rebuilds_on_dim_change(tmp_path: Path) -> None:
    """The memory index rebuilds when the embedding dimension changes."""
    store = MemoryStore(tmp_path, embedding_model="", embedding_provider=StubProvider(4))
    store.add_entry("q1", "a1")
    assert _memory_dim(tmp_path) == "4"

    store = MemoryStore(tmp_path, embedding_model="", embedding_provider=StubProvider(8))
    store.add_entry("q2", "a2")
    assert _memory_dim(tmp_path) == "8"


def test_memory_store_replaces_the_legacy_collection_directory(tmp_path: Path) -> None:
    """The old zvec collection was a *directory* at the same path.

    The path is a file now; the directory is dead weight and is replaced.
    """
    legacy = tmp_path / ".kompend" / "memory.db"
    legacy.mkdir(parents=True)
    (legacy / "fts.2.rocksdb").mkdir()

    store = MemoryStore(tmp_path, embedding_model="", embedding_provider=StubProvider(4))
    store.add_entry("q1", "a1")

    assert legacy.is_file()
    assert store.search("q1")


def test_wiki_store_default_dim_resolved_from_provider(tmp_path: Path) -> None:
    """The store takes its dimension from the provider, not from a config value."""
    db = tmp_path / "vectors.db"
    store = SqliteVecWikiStore(db_file=db, embedding_provider=StubProvider(7))
    store.build_index(_wiki())
    assert _store_dim(db) == "7"
    store.close()


def test_wiki_store_prunes_stale_ids(tmp_path: Path) -> None:
    """Renaming or removing a page must not leave its old id in the store."""
    db = tmp_path / "vectors.db"

    store = SqliteVecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    assert store.build_index(_wiki_ids("alpha", "beta")) == 2
    assert store.search("q", top_k=10)  # sanity: hits exist
    store.close()

    store = SqliteVecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    # "beta" renamed to "gamma" — old "beta" hits must disappear.
    assert store.build_index(_wiki_ids("alpha", "gamma")) == 2
    ids = {hit["id"] for hit in store.search("q", top_k=10)}
    assert "beta" not in ids
    assert "alpha" in ids
    assert "gamma" in ids
    store.close()


def test_wiki_store_rebuilds_on_schema_mismatch(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A store file written by another schema is rebuilt, not reopened.

    The vector index is derived data, so dropping it is the correct recovery —
    the alternative is a store whose columns do not match the code that
    writes them.
    """
    db = tmp_path / "vectors.db"
    conn = SQLiteVecClient.create_connection(str(db))
    conn.execute(
        """
        CREATE VIRTUAL TABLE vec_items USING vec0(
            id TEXT PRIMARY KEY,
            embedding FLOAT[4] distance_metric=cosine,
            title TEXT,
            node_type TEXT,
            source_file TEXT,
            content_preview TEXT
        )
        """
    )
    conn.execute("CREATE TABLE vec_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.executemany(
        "INSERT INTO vec_meta (key, value) VALUES (?, ?)",
        [("backend", "sqlite-vec"), ("dim", "4"), ("schema", "an-older-schema")],
    )
    conn.commit()
    conn.close()

    caplog.set_level(logging.INFO)
    store = SqliteVecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    assert store.build_index(_wiki()) == 1
    assert _store_dim(db) == "4"
    assert any("stale schema" in rec.message for rec in caplog.records)
    store.close()


def test_wiki_store_prunes_rows_missing_from_manifest(tmp_path: Path) -> None:
    """A row present in the store but absent from the manifest is pruned.

    The rebuild recreates the tables, so a row written by a crashed build (or
    before the manifest existed) cannot survive the next one.
    """
    db = tmp_path / "vectors.db"
    store = SqliteVecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    assert store.build_index(_wiki_ids("alpha", "beta")) == 2
    # Inject a row the manifest will never list (simulates a pre-manifest
    # build, or a crash before the manifest was written).
    store._require_conn().execute(
        "INSERT INTO vec_items (id, embedding, title, node_type, source_file, content_preview) VALUES (?, ?, ?, ?, ?, ?)",
        ("ghost", sqlite_vec.serialize_float32([0.5] * 4), "g", "concept", "", ""),
    )
    store.close()

    assert "ghost" not in _manifest_ids(tmp_path / "kompend.db")

    store = SqliteVecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    assert store.build_index(_wiki_ids("alpha", "beta")) == 2
    present = {row[0] for row in store._require_conn().execute("SELECT id FROM vec_items").fetchall()}
    assert "ghost" not in present
    assert present == {"alpha", "beta"}
    store.close()


def test_manifest_from_another_backend_is_stale(tmp_path: Path) -> None:
    """A manifest written by a different backend must not validate this store.

    The vectors of one backend are not readable by another, so the manifest
    names the backend that wrote them.
    """
    db = tmp_path / "vectors.db"
    wiki = _wiki()

    store = SqliteVecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    store.build_index(wiki)
    store.close()

    with sqlite3.connect(tmp_path / "kompend.db") as conn:
        conn.execute(f"UPDATE {VECTOR_MANIFEST_TABLE} SET backend = 'zvec' WHERE id = 1")
        conn.commit()

    other = SqliteVecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    assert other.is_stale(wiki) is True
    other.close()


def _store_dim(db_path: Path) -> str:
    """The vector dimension the store recorded in its own meta table."""
    conn = SQLiteVecClient.create_connection(str(db_path))
    try:
        row = conn.execute("SELECT value FROM vec_meta WHERE key = 'dim'").fetchone()
    finally:
        conn.close()
    assert row is not None, "build_index must record the store's dimension"
    return str(row[0])


def _memory_dim(base_dir: Path) -> str:
    """The vector dimension the memory store recorded."""
    conn = SQLiteVecClient.create_connection(str(base_dir / ".kompend" / "memory.db"))
    try:
        row = conn.execute("SELECT value FROM memory_meta WHERE key = 'dim'").fetchone()
    finally:
        conn.close()
    assert row is not None, "the memory store must record its dimension"
    return str(row[0])


def _manifest_ids(db_path: Path) -> list[str]:
    """Indexed id set recorded by the last build (kompend.db ``vector_manifest``)."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f"SELECT ids FROM {VECTOR_MANIFEST_TABLE} WHERE id = 1").fetchone()
    assert row is not None, "build_index must record a vector manifest row"
    return json.loads(row[0])


def _wiki_ids(*ids: str) -> CompiledWiki:
    return CompiledWiki(
        nodes=[WikiNode(id=i, title=i, content=f"c-{i}", node_type=NodeType.CONCEPT) for i in ids],
        edges=[],
        toc=[],
    )


def _wiki() -> CompiledWiki:
    return CompiledWiki(
        nodes=[
            WikiNode(id="n1", title="t", content="c", node_type=NodeType.CONCEPT),
        ],
        edges=[],
        toc=[],
    )
