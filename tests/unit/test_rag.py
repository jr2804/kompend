"""Tests for KNOX Hybrid RAG — zvec vector store."""

import gc
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knox.core.models import CompiledWiki, WikiNode
from knox.rag.zvec_store import HashEmbeddingProvider, LiteLLMEmbeddingProvider, ZvecWikiStore


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
        assert store.build_index(empty, embedding_model="hash") == 0

    def test_build_index_works(self, zvec_db: Path) -> None:
        wiki = CompiledWiki(
            nodes=[WikiNode(id="n1", title="Test", content="...", node_type="section")],
            edges=[],
            toc=[],
        )
        store = ZvecWikiStore(db_path=str(zvec_db))
        assert store.build_index(wiki, embedding_model="hash") == 1
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
            count = store.build_index(wiki, embedding_model="hash")
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
            count = store.build_index(wiki, embedding_model="hash")
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

    def test_open_existing_store_and_search(self, zvec_db: Path) -> None:
        """Opening an existing collection allows querying without rebuilding."""
        wiki = CompiledWiki(
            nodes=[WikiNode(id="n1", title="Testing", content="test content here", node_type="section")],
            edges=[],
            toc=[],
        )
        store = ZvecWikiStore(db_path=str(zvec_db))
        store.build_index(wiki, embedding_model="hash")
        store.close()

        try:
            reopened = ZvecWikiStore.open(str(zvec_db), embedding_model="hash", embedding_dim=384)
            assert reopened.available
            results = reopened.search("test")
            assert any(r["id"] == "n1" for r in results)
            reopened.close()
        finally:
            gc.collect()
            shutil.rmtree(zvec_db, ignore_errors=True)

    def test_source_file_is_preserved_in_search_results(self, zvec_db: Path) -> None:
        """Node source metadata is stored as a searchable field."""
        wiki = CompiledWiki(
            nodes=[
                WikiNode(
                    id="n1",
                    title="Methods",
                    content="Analysis methods used in the study.",
                    node_type="section",
                    metadata={"source": "paper.pdf"},
                ),
            ],
            edges=[],
            toc=[],
        )
        store = ZvecWikiStore(db_path=str(zvec_db))
        store.build_index(wiki, embedding_model="hash")
        try:
            results = store.search("methods")
            assert results
            assert results[0].get("source_file") == "paper.pdf"
        finally:
            store.close()
            gc.collect()
            shutil.rmtree(zvec_db, ignore_errors=True)


class TestLiteLLMEmbeddingProvider:
    """Tests for LiteLLMEmbeddingProvider dimension handling and validation."""

    def test_auto_detects_dimension_on_first_call(self) -> None:
        """Provider should auto-detect dimension if not explicitly provided."""
        provider = LiteLLMEmbeddingProvider(model="test-model")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1] * 768}]

        with patch("knox.rag.zvec_store.litellm.embedding", return_value=mock_response):
            vec = provider.embed("test text")

        assert len(vec) == 768
        assert provider.dim == 768

    def test_accepts_explicit_smaller_dimension(self) -> None:
        """Provider should accept and use an explicitly requested smaller dimension."""
        provider = LiteLLMEmbeddingProvider(model="test-model", dim=256)
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1] * 256}]

        with patch("knox.rag.zvec_store.litellm.embedding", return_value=mock_response) as mock_embed:
            vec = provider.embed("test text")

        # Verify the dimensions kwarg was passed to litellm
        mock_embed.assert_called_once_with(model="test-model", input=["test text"], dimensions=256)
        assert len(vec) == 256
        assert provider.dim == 256

    def test_rejects_larger_requested_dimension(self) -> None:
        """Provider should raise ValueError if model returns smaller vector than requested."""
        # Simulates a model that ignores the 'dimensions' param and returns its native 768 dim
        provider = LiteLLMEmbeddingProvider(model="test-model", dim=1024)
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1] * 768}]

        with patch("knox.rag.zvec_store.litellm.embedding", return_value=mock_response), pytest.raises(ValueError) as exc_info:
            provider.embed("test text")

        assert "Embedding dimension mismatch" in str(exc_info.value)
        assert "Requested 1024" in str(exc_info.value)
        assert "returned a vector of size 768" in str(exc_info.value)

    def test_rejects_smaller_requested_dimension(self) -> None:
        """Provider should raise ValueError if model returns larger vector than requested."""
        # Simulates a model that ignores the 'dimensions' param and returns its native 768 dim
        provider = LiteLLMEmbeddingProvider(model="test-model", dim=128)
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1] * 768}]

        with patch("knox.rag.zvec_store.litellm.embedding", return_value=mock_response), pytest.raises(ValueError) as exc_info:
            provider.embed("test text")

        assert "Embedding dimension mismatch" in str(exc_info.value)
        assert "Requested 128" in str(exc_info.value)
        assert "returned a vector of size 768" in str(exc_info.value)

    def test_raises_if_dim_accessed_before_embed(self) -> None:
        """Provider should raise RuntimeError if dim is accessed before first embed call."""
        provider = LiteLLMEmbeddingProvider(model="test-model")
        with pytest.raises(RuntimeError) as exc_info:
            _ = provider.dim
        assert "not yet determined" in str(exc_info.value)
