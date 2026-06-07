"""Tests for KNOX Hybrid RAG — zvec vector store."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knox.core.models import CompiledWiki, WikiNode
from knox.rag.zvec_store import ZvecWikiStore, ZVEC_AVAILABLE


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

    def test_build_index_without_zvec_returns_zero(self) -> None:
        wiki = CompiledWiki(
            nodes=[WikiNode(id="n1", title="Test", content="...", node_type="section")],
            edges=[],
            toc=[],
        )
        with patch("knox.rag.zvec_store.ZVEC_AVAILABLE", False):
            store = ZvecWikiStore(db_path="/tmp/test_zvec_not_used")
            assert store.build_index(wiki) == 0
            store.close()

    def test_search_after_build_returns_results(self) -> None:
        """Integration test: requires zvec to be importable."""
        if not ZVEC_AVAILABLE:
            pytest.skip("zvec not available on this platform")
        db = Path("/tmp/test_zvec_store")
        if db.exists():
            import shutil
            shutil.rmtree(db)
        try:
            store = ZvecWikiStore(db_path=str(db), embedding_dim=384)
            wiki = CompiledWiki(
                nodes=[
                    WikiNode(
                        id="n1", title="Testing", content="test content here", node_type="section"
                    ),
                    WikiNode(
                        id="n2", title="Results", content="important findings", node_type="section"
                    ),
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
            if db.exists():
                import shutil
                shutil.rmtree(db)

    def test_build_index_creates_parent_dir(self) -> None:
        if not ZVEC_AVAILABLE:
            pytest.skip("zvec not available")
        db = Path("/tmp/test_zvec_nested/deep/store")
        if db.exists():
            import shutil
            shutil.rmtree(db.parent)
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
            if db.parent.exists():
                import shutil
                shutil.rmtree(db.parent)

    def test_close_then_search_returns_empty(self) -> None:
        store = ZvecWikiStore()
        store.close()
        assert store.search("anything") == []

    def test_text_to_vec_deterministic(self) -> None:
        v1 = ZvecWikiStore._text_to_vec("hello world", dim=10)
        v2 = ZvecWikiStore._text_to_vec("hello world", dim=10)
        v3 = ZvecWikiStore._text_to_vec("different text", dim=10)
        assert v1 == v2
        assert v1 != v3

    def test_text_to_vec_normalized(self) -> None:
        vec = ZvecWikiStore._text_to_vec("test", dim=20)
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 0.001

    def test_available_property_without_zvec(self) -> None:
        with patch("knox.rag.zvec_store.ZVEC_AVAILABLE", False):
            store = ZvecWikiStore()
            assert store.available is False
            store.close()

    def test_close_is_idempotent(self) -> None:
        store = ZvecWikiStore()
        store.close()
        store.close()  # should not raise