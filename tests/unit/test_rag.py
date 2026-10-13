"""Tests for KOMPEND Hybrid RAG — zvec vector store."""

import gc
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kompend.core.models import CompiledWiki, WikiNode
from kompend.rag.zvec_store import PydanticAIEmbeddingProvider, ZvecWikiStore, make_provider, to_pydantic_ai_embedding_string
from tests.conftest import DeterministicTestEmbeddingProvider

_TEST_PROVIDER = DeterministicTestEmbeddingProvider(dim=384)


class TestZvecWikiStore:
    """Tests for ZvecWikiStore."""

    def test_empty_store_returns_empty_search(self) -> None:
        store = ZvecWikiStore()
        assert store.search("anything") == []
        assert store.available is False

    def test_empty_wiki_returns_zero_indexed(self) -> None:
        empty = CompiledWiki(nodes=[], edges=[], toc=[])
        store = ZvecWikiStore(embedding_provider=_TEST_PROVIDER)
        assert store.build_index(empty) == 0

    def test_build_index_works(self, zvec_db: Path) -> None:
        wiki = CompiledWiki(
            nodes=[WikiNode(id="n1", title="Test", content="...", node_type="section")],
            edges=[],
            toc=[],
        )
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        assert store.build_index(wiki) == 1
        store.close()

    def test_search_after_build_returns_results(self, zvec_db: Path) -> None:
        """Integration test: build index and search."""
        try:
            store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
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
            store = ZvecWikiStore(db_file=str(db), embedding_provider=_TEST_PROVIDER)
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

    def test_deterministic_embedding_deterministic(self) -> None:
        provider = DeterministicTestEmbeddingProvider(dim=10)
        v1 = provider.embed("hello world")
        v2 = provider.embed("hello world")
        v3 = provider.embed("different text")
        assert v1 == v2
        assert v1 != v3

    def test_deterministic_embedding_normalized(self) -> None:
        provider = DeterministicTestEmbeddingProvider(dim=20)
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
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        store.build_index(wiki)
        store.close()

        try:
            reopened = ZvecWikiStore.open(str(zvec_db), embedding_provider=_TEST_PROVIDER)
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
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        store.build_index(wiki)
        try:
            results = store.search("methods")
            assert results
            assert results[0].get("source_file") == "paper.pdf"
        finally:
            store.close()
            gc.collect()
            shutil.rmtree(zvec_db, ignore_errors=True)


class TestEmbeddingStringConversion:
    """Tests for _to_pydantic_ai_embedding_string format translation."""

    def test_provider_slash_format_translated_to_colon(self) -> None:
        assert to_pydantic_ai_embedding_string("openai/text-embedding-3-small") == "openai:text-embedding-3-small"

    def test_ollama_slash_format_translated(self) -> None:
        assert to_pydantic_ai_embedding_string("ollama/nomic-embed-text") == "ollama:nomic-embed-text"

    def test_provider_colon_format_unchanged(self) -> None:
        assert to_pydantic_ai_embedding_string("openai:text-embedding-3-small") == "openai:text-embedding-3-small"

    def test_bare_hf_name_rejected_without_provider(self) -> None:
        """A model without an explicit provider prefix is rejected (no local fallback)."""
        with pytest.raises(ValueError, match="provider"):
            to_pydantic_ai_embedding_string("BAAI/bge-small-en")
        with pytest.raises(ValueError, match="provider"):
            to_pydantic_ai_embedding_string("all-MiniLM-L6-v2")


class TestPydanticAIEmbeddingProvider:
    """Tests for PydanticAIEmbeddingProvider dimension handling."""

    def test_auto_detects_dimension_on_first_embed(self) -> None:
        """Provider should auto-detect dimension after the first embed call."""
        provider = PydanticAIEmbeddingProvider(model_string="test:test-model")
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 768]
        with patch.object(provider, "_embedder") as mock_embedder:
            mock_embedder.embed_documents_sync.return_value = mock_result
            vec = provider.embed("test text")
        assert len(vec) == 768
        assert provider.dim == 768

    def test_dim_probe_triggers_embed(self) -> None:
        """Accessing dim before any embed call triggers a probe embed."""
        provider = PydanticAIEmbeddingProvider(model_string="test:test-model")
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 384]
        with patch.object(provider, "_embedder") as mock_embedder:
            mock_embedder.embed_documents_sync.return_value = mock_result
            assert provider.dim == 384

    def test_dim_shortcut_skips_probe(self) -> None:
        """A known ``dim`` makes ``.dim`` daemon-free — the probe embed must
        not be issued when the caller already supplies the dimension.
        """
        provider = PydanticAIEmbeddingProvider(model_string="test:test-model", dim=768)
        with patch.object(provider, "_embedder") as mock_embedder:
            assert provider.dim == 768
        mock_embedder.embed_documents_sync.assert_not_called()

    def test_make_provider_forwards_dim_to_provider(self) -> None:
        """``make_provider`` wires the ``dim`` shortcut through to the
        underlying provider so factory callers can avoid the probe embed.
        """
        provider = make_provider("ollama:granite-embedding:278m", dim=384)
        assert provider.dim == 384


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
