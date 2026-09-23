"""Rebuild-on-dimension-change guard for the zvec wiki/memory stores."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pytest
import zvec as _zvec

from knox.core.models import CompiledWiki, WikiNode
from knox.memory.store import MemoryStore
from knox.rag.zvec_store import EmbeddingProvider, ZvecWikiStore


class StubProvider(EmbeddingProvider):
    def __init__(self, dim: int) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return [0.5] * self._dim


def test_wiki_store_rebuilds_on_dim_change() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "vectors"

        store = ZvecWikiStore(db_file=db, embedding_provider=StubProvider(4))
        assert store.build_index(_wiki(4)) == 1
        assert store._collection is not None
        assert store._collection.schema.vectors[0].dimension == 4
        store.close()

        # Same path, model now produces 8 dims — must rebuild, not reopen.
        store = ZvecWikiStore(db_file=db, embedding_provider=StubProvider(8))
        assert store.build_index(_wiki(8)) == 1
        assert store._collection is not None
        assert store._collection.schema.vectors[0].dimension == 8
        assert store.search("q")  # insert + query at 8 dims works
        store.close()


def test_memory_store_rebuilds_on_dim_change(tmp_path: Path) -> None:
    db = tmp_path

    store = MemoryStore(db, embedding_model="", embedding_provider=StubProvider(4))
    store.add_entry("q1", "a1")
    assert store._collection is not None
    assert store._collection.schema.vectors[0].dimension == 4

    store = MemoryStore(db, embedding_model="", embedding_provider=StubProvider(8))
    store.add_entry("q2", "a2")
    assert store._collection is not None
    assert store._collection.schema.vectors[0].dimension == 8


def test_wiki_store_default_dim_resolved_from_provider() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ZvecWikiStore(db_file=Path(tmp) / "vectors", embedding_provider=StubProvider(7))
        store.build_index(_wiki(7))
        assert store._collection is not None
        assert store._collection.schema.vectors[0].dimension == 7
        store.close()


def test_wiki_store_prunes_stale_ids(tmp_path: Path) -> None:
    """Follow-up to #24: renaming/removing a page must not leave its
    old id in the vector store after a rebuild.
    """
    db = tmp_path / "vectors"

    store = ZvecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    assert store.build_index(_wiki_ids(4, "alpha", "beta")) == 2
    assert store.search("q", top_k=10)  # sanity: hits exist
    store.close()

    store = ZvecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    # "beta" renamed to "gamma" — old "beta" hits must disappear.
    assert store.build_index(_wiki_ids(4, "alpha", "gamma")) == 2
    ids = {hit["id"] for hit in store.search("q", top_k=10)}
    assert "beta" not in ids
    assert "alpha" in ids
    assert "gamma" in ids
    store.close()


def test_wiki_store_rebuilds_on_field_mismatch(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """When a stored collection predates a newer knox field, the store
    detects the mismatch and rebuilds rather than silently failing at
    insert time with ``attributes not found in collection schema``.

    The mismatch is simulated by creating the collection directly with a
    schema missing the ``attributes`` field (the real-world scenario: a
    wiki compiled before that field was added to knox). zvec's STRING
    columns are immutable (add_column / drop_column only accept numeric
    types), so in-place migration is not possible — destroy+rebuild is the
    only correct path.
    """
    db = tmp_path / "vectors"
    # Seed an older collection without the "attributes" field.
    legacy_schema = _zvec.CollectionSchema(
        name="wiki",
        fields=[
            _zvec.FieldSchema("node_id", _zvec.DataType.STRING),
            _zvec.FieldSchema("title", _zvec.DataType.STRING),
            _zvec.FieldSchema("node_type", _zvec.DataType.STRING),
            _zvec.FieldSchema("source_file", _zvec.DataType.STRING),
            _zvec.FieldSchema("content_preview", _zvec.DataType.STRING),
            _zvec.FieldSchema("tags", _zvec.DataType.STRING),
            # Note: no "attributes" field — mirrors pre-4746a81d wiki state.
        ],
        vectors=_zvec.VectorSchema("embedding", _zvec.DataType.VECTOR_FP32, 4),
    )
    legacy = _zvec.create_and_open(path=str(db), schema=legacy_schema)
    legacy.close()

    caplog.set_level(logging.WARNING)
    store = ZvecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    assert store.build_index(_wiki(4)) == 1
    assert store._collection is not None
    # Schema was rebuilt with the missing field restored.
    assert "attributes" in {f.name for f in store._collection.schema.fields}
    assert any("schema mismatch" in rec.message for rec in caplog.records)
    store.close()


def test_wiki_store_rebuilds_on_obsolete_field(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A stored field no longer in the expected schema also triggers
    destroy+rebuild (data loss is logged at WARNING).
    """
    db = tmp_path / "vectors"
    # Seed a collection with an extra numeric field (zvec accepts add_column
    # for numeric types). Drop it later by running through knox which has a
    # different expected schema.
    seed_schema = _zvec.CollectionSchema(
        name="wiki",
        fields=[
            _zvec.FieldSchema("node_id", _zvec.DataType.STRING),
            _zvec.FieldSchema("title", _zvec.DataType.STRING),
            _zvec.FieldSchema("node_type", _zvec.DataType.STRING),
            _zvec.FieldSchema("source_file", _zvec.DataType.STRING),
            _zvec.FieldSchema("content_preview", _zvec.DataType.STRING),
            _zvec.FieldSchema("tags", _zvec.DataType.STRING),
        ],
        vectors=_zvec.VectorSchema("embedding", _zvec.DataType.VECTOR_FP32, 4),
    )
    legacy = _zvec.create_and_open(path=str(db), schema=seed_schema)
    legacy.add_column(_zvec.FieldSchema("legacy_numeric", _zvec.DataType.INT64, nullable=True))
    legacy.close()

    caplog.set_level(logging.WARNING)
    store = ZvecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    assert store.build_index(_wiki(4)) == 1
    assert store._collection is not None
    assert "legacy_numeric" not in {f.name for f in store._collection.schema.fields}
    assert any("schema mismatch" in rec.message for rec in caplog.records)
    store.close()


def test_wiki_store_rebuilds_on_field_type_change(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A field whose name is unchanged but whose type changed must rebuild.

    ``tags`` moved STRING -> ARRAY_STRING while keeping its name; a name-only
    drift check would reopen the old STRING collection and silently mis-insert
    list values. The rebuild must key on (name, data_type).
    """
    db = tmp_path / "vectors"
    legacy_schema = _zvec.CollectionSchema(
        name="wiki",
        fields=[
            _zvec.FieldSchema("node_id", _zvec.DataType.STRING),
            _zvec.FieldSchema("title", _zvec.DataType.STRING),
            _zvec.FieldSchema("node_type", _zvec.DataType.STRING),
            _zvec.FieldSchema("source_file", _zvec.DataType.STRING),
            _zvec.FieldSchema("content_preview", _zvec.DataType.STRING),
            _zvec.FieldSchema("tags", _zvec.DataType.STRING),
            _zvec.FieldSchema("attributes", _zvec.DataType.STRING),
        ],
        vectors=_zvec.VectorSchema("embedding", _zvec.DataType.VECTOR_FP32, 4),
    )
    legacy = _zvec.create_and_open(path=str(db), schema=legacy_schema)
    legacy.close()

    caplog.set_level(logging.WARNING)
    store = ZvecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    assert store.build_index(_wiki(4)) == 1
    assert store._collection is not None
    field_types = {f.name: f.data_type for f in store._collection.schema.fields}
    assert field_types["tags"] == _zvec.DataType.ARRAY_STRING
    assert field_types["attributes"] == _zvec.DataType.ARRAY_STRING
    assert any("schema mismatch" in rec.message for rec in caplog.records)
    store.close()


def test_wiki_store_prunes_docs_missing_from_manifest(tmp_path: Path) -> None:
    """A doc present in the collection but absent from the manifest is pruned.

    Regression for the manifest-diff bug: docs written before the manifest
    mechanism existed were invisible to pruning and persisted forever,
    inflating every search.
    """
    db = tmp_path / "vectors"
    store = ZvecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    assert store.build_index(_wiki_ids(4, "alpha", "beta")) == 2
    assert store._collection is not None
    # Inject a doc the manifest will never list (simulates a pre-manifest
    # build, or a crash before the manifest was written).
    store._collection.upsert(
        [
            _zvec.Doc(
                id="ghost",
                vectors={"embedding": [0.5, 0.5, 0.5, 0.5]},
                fields={
                    "node_id": "ghost",
                    "title": "g",
                    "node_type": "concept",
                    "source_file": "",
                    "content_preview": "",
                    "tags": [],
                    "attributes": [],
                },
            ),
        ],
    )
    store.close()

    manifest = json.loads((tmp_path / "vectors.manifest.json").read_text(encoding="utf-8"))
    assert "ghost" not in manifest

    store = ZvecWikiStore(db_file=db, embedding_provider=StubProvider(4))
    assert store.build_index(_wiki_ids(4, "alpha", "beta")) == 2
    assert store._collection is not None
    present = {doc.id for doc in store._collection.iter_docs()}
    assert "ghost" not in present
    assert present == {"alpha", "beta"}
    store.close()


def _wiki_ids(dim: int, *ids: str) -> CompiledWiki:
    return CompiledWiki(
        nodes=[WikiNode(id=i, title=i, content=f"c-{i}", node_type="concept") for i in ids],
        edges=[],
        toc=[],
    )


def _wiki(dim: int) -> CompiledWiki:
    return CompiledWiki(
        nodes=[
            WikiNode(id="n1", title="t", content="c", node_type="concept"),
        ],
        edges=[],
        toc=[],
    )
