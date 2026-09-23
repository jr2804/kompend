"""Tests for the zvec embed-text composition and its staleness guard.

``_embed_text`` decides what text the vector store embeds for a page, and
``is_stale`` decides whether an existing index may still be served. Both are
invisible at query time when they are wrong — a changed composition silently
keeps serving old vectors, and a dropped ``description`` silently narrows what
retrieval can see. These tests pin both.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest

from knox.core.models import CompiledWiki, WikiNode
from knox.rag.zvec_store import _EMBED_TEXT_VERSION, ZvecWikiStore, _embed_text
from tests.conftest import DeterministicTestEmbeddingProvider

_TEST_PROVIDER = DeterministicTestEmbeddingProvider(dim=384)


class TestEmbedTextComposition:
    """Pin the exact text fed to the embedder."""

    def test_includes_title_description_and_content(self) -> None:
        node = _node("a", content="## Definition\n\nP.863 is a standard.")
        assert node.description == "P.863 is a standard."
        assert _embed_text(node) == "T\nP.863 is a standard.\n## Definition\n\nP.863 is a standard."

    def test_headings_only_falls_back_to_title_and_content(self) -> None:
        node = _node("a", content="## A\n### B")
        assert node.description == ""
        assert _embed_text(node) == "T\n## A\n### B"

    def test_empty_content_falls_back_to_title(self) -> None:
        assert _embed_text(_node("a", content="")) == "T\n"


class TestDescriptionDerivation:
    """The derivation moved to the model; pin the behaviour it must keep."""

    def test_skips_leading_headings(self) -> None:
        assert _node("a", content="# H\n## H2\nReal prose.").description == "Real prose."

    def test_reduces_wikilinks_to_display_text(self) -> None:
        node = _node("a", content="See [[Page|the standard]] for details.")
        assert node.description == "See the standard for details."

    def test_does_not_truncate_after_an_initial(self) -> None:
        node = _node("a", content="John G. Beerends is the author.")
        assert node.description == "John G. Beerends is the author."


class TestManifestStaleness:
    """A manifest must prove both its ids AND its embed-text version."""

    def test_build_records_the_current_version(self, zvec_db: Path) -> None:
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        wiki = _wiki(_node("n1"))
        assert store.build_index(wiki) == 1
        manifest = json.loads(store._manifest_file().read_text(encoding="utf-8"))
        assert manifest["embed_version"] == _EMBED_TEXT_VERSION
        assert manifest["ids"] == ["n1"]
        assert store.is_stale(wiki) is False
        store.close()
        gc.collect()

    def test_version_mismatch_is_stale(self, zvec_db: Path) -> None:
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        wiki = _wiki(_node("n1"))
        store.build_index(wiki)
        store._manifest_file().write_text(
            json.dumps({"embed_version": _EMBED_TEXT_VERSION - 1, "ids": ["n1"]}),
            encoding="utf-8",
        )
        assert store.is_stale(wiki) is True
        store.close()
        gc.collect()

    def test_legacy_bare_id_list_is_stale(self, zvec_db: Path) -> None:
        """A manifest written before versioning cannot vouch for its vectors."""
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        wiki = _wiki(_node("n1"))
        store.build_index(wiki)
        store._manifest_file().write_text(json.dumps(["n1"]), encoding="utf-8")
        assert store.is_stale(wiki) is True
        store.close()
        gc.collect()

    def test_id_mismatch_is_stale(self, zvec_db: Path) -> None:
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        store.build_index(_wiki(_node("n1")))
        assert store.is_stale(_wiki(_node("n1"), _node("n2"))) is True
        store.close()
        gc.collect()


@pytest.fixture
def zvec_db(tmp_path: Path) -> Path:
    """Provide a temp directory for zvec test databases."""
    return tmp_path / "zvec_test_store"


def _wiki(*nodes: WikiNode) -> CompiledWiki:
    return CompiledWiki(nodes=list(nodes), edges=[], toc=[])


def _node(node_id: str, content: str = "body") -> WikiNode:
    return WikiNode(id=node_id, title="T", content=content, node_type="page")
