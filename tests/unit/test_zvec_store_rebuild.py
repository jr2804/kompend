"""Rebuild-on-dimension-change guard for the zvec wiki/memory stores."""

from __future__ import annotations

import tempfile
from pathlib import Path

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


def _wiki(dim: int) -> CompiledWiki:
    return CompiledWiki(
        nodes=[
            WikiNode(id="n1", title="t", content="c", node_type="concept"),
        ],
        edges=[],
        toc=[],
    )


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


def _wiki_ids(dim: int, *ids: str) -> CompiledWiki:
    return CompiledWiki(
        nodes=[WikiNode(id=i, title=i, content=f"c-{i}", node_type="concept") for i in ids],
        edges=[],
        toc=[],
    )
