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
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import pytest

from kompend.core.models import CompiledWiki, WikiNode, load_wiki_from_files
from kompend.rag.zvec_store import (
    _EMBED_TEXT_VERSION,
    ZvecWikiStore,
    _content_hash,
    _embed_text,
)
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

    def test_binary_content_never_reaches_the_embedder_as_a_repr(self) -> None:
        """A payload that is not text must not embed as a bytes repr.

        Node content is declared ``str``, so a non-text payload used to arrive
        already stringified by an upstream ``str()`` - invisibly, because the
        declared type made it look handled.
        """
        payload = b"\x89PNG\r\n\x1a\n" + bytes(2048)
        node = replace(_node("a", content="placeholder"), content=payload)  # type: ignore[arg-type]
        embedded = _embed_text(node)
        assert "b'" not in embedded
        assert "x89" not in embedded
        assert "binary content" in embedded
        assert str(len(payload)) in embedded

    def test_structured_table_content_embeds_as_rows(self) -> None:
        node = replace(_node("a", content="placeholder"), content=[["Rate", "Value"], ["1", "2"]])  # type: ignore[arg-type]
        embedded = _embed_text(node)
        assert "Rate | Value" in embedded
        assert "1 | 2" in embedded
        assert "[['Rate'" not in embedded


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


class TestIndexEligibilityParity:
    """Build and staleness must agree on *which* nodes the index covers.

    A mismatch here is invisible except as waste: the index looks stale, so the
    next query re-embeds the whole corpus. It regressed twice — once because the
    compile path indexed the in-memory graph while the query path compares
    against the markdown bundle (which also holds summary pages), and once
    because ``embed_attachments`` filtered the build but not the comparison.
    """

    def test_attachment_filter_is_symmetric(self, zvec_db: Path) -> None:
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        wiki = _wiki(_node("page"), _node("att", source_asset=True))

        # Built without attachments: the index covers one node, and the check
        # must judge against that same one — not both. The store owns that
        # rule, so no caller-side pre-filtering is involved.
        assert store.build_index(wiki) == 1
        assert store.is_stale(wiki) is False
        # The same store judged with the other setting *is* stale.
        assert store.is_stale(wiki, embed_attachments=True) is True
        store.close()
        gc.collect()

    def test_bundle_pages_the_graph_lacks_do_not_look_stale(self, tmp_path: Path) -> None:
        """A summary page present only in the bundle must not force a rebuild."""
        wiki_dir = tmp_path / "wiki"
        (wiki_dir / "concepts").mkdir(parents=True)
        (wiki_dir / "summaries").mkdir(parents=True)
        (wiki_dir / "concepts" / "loudness.md").write_text("---\ntitle: Loudness\n---\n\nBody.\n", encoding="utf-8")
        (wiki_dir / "summaries" / "itu-p862.md").write_text("---\ntitle: ITU P.862\n---\n\nSummary.\n", encoding="utf-8")
        bundle = load_wiki_from_files(wiki_dir)
        assert bundle is not None
        assert sorted(n.id for n in bundle.nodes) == ["itu-p862", "loudness"]

        store = ZvecWikiStore(db_file=str(tmp_path / "vectors"), embedding_provider=_TEST_PROVIDER)
        assert store.build_index(bundle) == 2
        assert store.is_stale(bundle) is False
        store.close()
        gc.collect()


class TestManifestStaleness:
    """A manifest must prove both its ids AND its embed-text version."""

    def test_build_records_the_current_version(self, zvec_db: Path) -> None:
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        wiki = _wiki(_node("n1"))
        assert store.build_index(wiki) == 1
        manifest = _manifest_row(store)
        assert manifest["embed_version"] == _EMBED_TEXT_VERSION
        assert manifest["ids"] == ["n1"]
        assert store.is_stale(wiki) is False
        store.close()
        gc.collect()

    def test_version_mismatch_is_stale(self, zvec_db: Path) -> None:
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        wiki = _wiki(_node("n1"))
        store.build_index(wiki)
        _set_manifest_version(store, _EMBED_TEXT_VERSION - 1)
        assert store.is_stale(wiki) is True
        store.close()
        gc.collect()

    def test_legacy_json_sidecar_is_ignored(self, zvec_db: Path) -> None:
        """A pre-migration vault's loose JSON manifest is never read.

        Policy: ``kompend.db`` holds all non-vector state. A ``vectors.manifest.json``
        left by an older kompend is dead weight (reported by ``kompend doctor``) and
        its store is rebuilt rather than trusted.
        """
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        wiki = _wiki(_node("n1"))
        store.build_index(wiki)
        _delete_manifest_row(store)
        (store.db_file.parent / "vectors.manifest.json").write_text(json.dumps(["n1"]), encoding="utf-8")
        assert store.is_stale(wiki) is True
        store.close()
        gc.collect()

    def test_id_mismatch_is_stale(self, zvec_db: Path) -> None:
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        store.build_index(_wiki(_node("n1")))
        assert store.is_stale(_wiki(_node("n1"), _node("n2"))) is True
        store.close()
        gc.collect()

    def test_build_records_content_hashes(self, zvec_db: Path) -> None:
        """One digest per indexed id — the fingerprint a later check compares."""
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        wiki = _wiki(_node("n1"))
        store.build_index(wiki)

        hashes = _manifest_row(store)["hashes"]
        assert list(hashes) == ["n1"]
        assert hashes["n1"] == _content_hash(_node("n1"))
        assert store.is_stale(wiki) is False
        store.close()
        gc.collect()

    def test_content_change_is_stale(self, zvec_db: Path) -> None:
        """An edited page invalidates the index even though its id is unchanged.

        The id set cannot see this: a lint fix, a hand edit or a recompile that
        reworded a page leaves the vectors describing text search no longer
        returns.
        """
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        store.build_index(_wiki(_node("n1", "original body")))
        assert store.is_stale(_wiki(_node("n1", "rewritten body"))) is True
        assert store.is_stale(_wiki(_node("n1", "original body"))) is False
        store.close()
        gc.collect()

    def test_tag_change_is_stale(self, zvec_db: Path) -> None:
        """Tags are filterable index fields, so retagging is a content change."""
        tagged = WikiNode(
            id="n1",
            title="T",
            content="body",
            node_type="page",
            metadata={"tags": ["a"]},
        )
        retagged = replace(tagged, metadata={"tags": ["a", "b"]})
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        store.build_index(_wiki(tagged))

        assert store.is_stale(_wiki(retagged)) is True
        store.close()
        gc.collect()

    def test_manifest_without_hashes_is_stale(self, zvec_db: Path) -> None:
        """A store built before hashing is stale once, then records them."""
        wiki = _wiki(_node("n1"))
        store = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        store.build_index(wiki)
        _clear_manifest_hashes(store)
        assert store.is_stale(wiki) is True
        store.close()
        gc.collect()

        # Rebuild through a fresh store — the shape `open_vector_store` uses.
        rebuilt = ZvecWikiStore(db_file=str(zvec_db), embedding_provider=_TEST_PROVIDER)
        rebuilt.build_index(wiki)
        assert rebuilt.is_stale(wiki) is False
        rebuilt.close()
        gc.collect()

    def test_hash_is_stable_for_a_fixed_node(self, zvec_db: Path) -> None:
        """The digest is a contract: the payload cannot change silently.

        A mismatch here means the hashed payload changed — update the expected
        value deliberately, and bump ``_EMBED_TEXT_VERSION`` if the embedded
        text itself moved.
        """
        assert _content_hash(_node("n1")) == "893fa586f558d6e1"


class _ManifestRow(TypedDict):
    """The `vector_manifest` row a build records: ids, hashes, embed version."""

    embed_version: int
    ids: list[str]
    hashes: dict[str, str]


@pytest.fixture
def zvec_db(tmp_path: Path) -> Path:
    """Provide a temp directory for zvec test databases."""
    return tmp_path / "zvec_test_store"


def _manifest_row(store: ZvecWikiStore) -> _ManifestRow:
    """The vector manifest recorded by the last build (kompend.db row)."""
    with sqlite3.connect(store._manifest_db_path()) as conn:
        row = conn.execute("SELECT embed_version, ids, content_hashes FROM vector_manifest WHERE id = 1").fetchone()
    assert row is not None, "a build must record the vector manifest"
    return {"embed_version": row[0], "ids": json.loads(row[1]), "hashes": json.loads(row[2])}


def _set_manifest_version(store: ZvecWikiStore, version: int) -> None:
    """Overwrite the recorded embed-text version (simulates an older build)."""
    with sqlite3.connect(store._manifest_db_path()) as conn:
        conn.execute("UPDATE vector_manifest SET embed_version = ? WHERE id = 1", (version,))


def _clear_manifest_hashes(store: ZvecWikiStore) -> None:
    """Blank the recorded hashes, as a build from before they were tracked has."""
    with sqlite3.connect(store._manifest_db_path()) as conn:
        conn.execute("UPDATE vector_manifest SET content_hashes = '{}' WHERE id = 1")


def _delete_manifest_row(store: ZvecWikiStore) -> None:
    """Drop the manifest row, as a store built before the migration has none."""
    with sqlite3.connect(store._manifest_db_path()) as conn:
        conn.execute("DELETE FROM vector_manifest")


def _wiki(*nodes: WikiNode) -> CompiledWiki:
    return CompiledWiki(nodes=list(nodes), edges=[], toc=[])


def _node(node_id: str, content: str = "body", *, source_asset: bool = False) -> WikiNode:
    metadata: dict[str, object] = {"source_asset": True} if source_asset else {}
    return WikiNode(id=node_id, title="T", content=content, node_type="page", metadata=metadata)
