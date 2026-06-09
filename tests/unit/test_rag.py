"""Tests for KNOX Hybrid RAG — zvec vector store."""

import gc
import shutil
from pathlib import Path

import pytest

from knox.core.models import CompiledWiki, WikiNode
from knox.rag.zvec_store import HashEmbeddingProvider, ZvecWikiStore


@pytest.fixture
def sample_wiki() -> CompiledWiki:
    return CompiledWiki(
        nodes=[
            WikiNode(
                id="intro",
                title="Introduction",
                content="This is the introduction content.",
                node_type="section",
            ),
            WikiNode(
                id="methods",
                title="Methods",
                content="We used several analysis methods.",
                node_type="section",
            ),
        ],
        edges=[{"source": "intro", "target": "methods", "type": "next"}],
        toc=[{"title": "Introduction", "id": "intro", "level": 1}],
        metadata={"chunk_count": 2},
    )


@pytest.fixture
def zvec_db(tmp_path: Path) -> Path:
    """Provide a temp directory for zvec test databases."""
    return tmp_path / "zvec_test_store"


class TestZvecWikiStore:
    """Tests for ZvecWikiStore."""

    def test_empty_store_returns_empty_search(self) -> None:
        store = ZvecWikiStore()
        assert store.search("anything") == []
        assert store.available is False

    def test_empty_wiki_returns_zero_indexed(self) -> None:
        empty = CompiledWiki(nodes=[], edges=[], toc=[])
        store = ZvecWikiStore()
        assert store.build_index(empty) == 0

    def test_build_index_works(self, zvec_db: Path) -> None:
        wiki = CompiledWiki(
            nodes=[WikiNode(id="n1", title="Test", content="...", node_type="section")],
            edges=[],
            toc=[],
        )
        store = ZvecWikiStore(db_path=str(zvec_db))
        assert store.build_index(wiki) == 1
        store.close()

    def test_search_after_build_returns_results(self, zvec_db: Path) -> None:
        """Integration test: build index and search."""
        try:
            store = ZvecWikiStore(db_path=str(zvec_db), embedding_dim=384)
            wiki = CompiledWiki(
                nodes=[
                    WikiNode(id="n1", title="Testing", content="test content here", node_type="section"),
                    WikiNode(id="n2", title="Results", content="important findings", node_type="section"),
                ],
                edges=[],
                toc=[],
            )
            count = store.build_index(wiki)
            assert count == 2, "Should index 2 nodes"
            results = store.search("test")
            assert len(results) >= 1
            store.close()
        finally:
            gc.collect()
            shutil.rmtree(zvec_db, ignore_errors=True)

    def test_build_index_creates_parent_dir(self, zvec_db: Path) -> None:
        db = zvec_db / "nested" / "deep" / "store"
        try:
            store = ZvecWikiStore(db_path=str(db))
            wiki = CompiledWiki(
                nodes=[
                    WikiNode(id="n1", title="Test", content="...", node_type="section"),
                ],
                edges=[],
                toc=[],
            )
            count = store.build_index(wiki)
            assert count == 1
            assert db.parent.exists()
            store.close()
        finally:
            gc.collect()
            shutil.rmtree(zvec_db, ignore_errors=True)

    def test_close_then_search_returns_empty(self) -> None:
        store = ZvecWikiStore()
        store.close()
        assert store.search("anything") == []

    def test_hash_embedding_deterministic(self) -> None:
        provider = HashEmbeddingProvider(dim=10)
        v1 = provider.embed("hello world")
        v2 = provider.embed("hello world")
        v3 = provider.embed("different text")
        assert v1 == v2
        assert v1 != v3

    def test_hash_embedding_normalized(self) -> None:
        provider = HashEmbeddingProvider(dim=20)
        vec = provider.embed("test")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 0.001

    def test_available_property_before_build(self) -> None:
        store = ZvecWikiStore()
        assert store.available is False
        store.close()

    def test_close_is_idempotent(self) -> None:
        store = ZvecWikiStore()
        store.close()
        store.close()  # should not raise
