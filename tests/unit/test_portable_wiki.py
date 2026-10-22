"""Portable wiki loading and keyword search (no .kompend state)."""

from __future__ import annotations

from pathlib import Path

from kompend.core.models import load_wiki_from_files
from kompend.core.models.enums import NodeType
from kompend.search.retriever import KeywordRetriever


class TestLoadWikiFromFiles:
    def test_returns_none_without_output_dir(self, tmp_path: Path) -> None:
        assert load_wiki_from_files(tmp_path / "missing") is None

    def test_loads_nodes_from_markdown(self, tmp_path: Path) -> None:
        _write_page(tmp_path, "alpha", "Alpha", "A page about alpha waves.")
        wiki = load_wiki_from_files(tmp_path)
        assert wiki is not None
        assert len(wiki.nodes) == 1
        node = wiki.nodes[0]
        assert node.id == "alpha"
        assert node.title == "Alpha"
        assert "alpha waves" in node.content
        assert node.node_type == NodeType.CONCEPT

    def test_skips_index_log_references_pages(self, tmp_path: Path) -> None:
        _write_page(tmp_path, "alpha", "Alpha", "Content.")
        (tmp_path / "index.md").write_text("# Index\n", encoding="utf-8")
        (tmp_path / "log.md").write_text("# Log\n", encoding="utf-8")
        (tmp_path / "references.md").write_text("# Refs\n", encoding="utf-8")
        wiki = load_wiki_from_files(tmp_path)
        assert wiki is not None
        assert {n.id for n in wiki.nodes} == {"alpha"}

    def test_builds_edges_from_links(self, tmp_path: Path) -> None:
        _write_page(tmp_path, "alpha", "Alpha", "See [Beta](concepts/beta.md).")
        _write_page(tmp_path, "beta", "Beta", "See [[Alpha]].")
        wiki = load_wiki_from_files(tmp_path)
        assert wiki is not None
        sources = {e["source"] for e in wiki.edges}
        targets = {e["target"] for e in wiki.edges}
        assert sources == {"alpha", "beta"}
        assert targets == {"alpha", "beta"}

    def test_metadata_from_frontmatter(self, tmp_path: Path) -> None:
        _write_page(tmp_path, "alpha", "Alpha", "Content.")
        wiki = load_wiki_from_files(tmp_path)
        assert wiki is not None
        assert "description" in wiki.nodes[0].metadata


class TestKeywordRetriever:
    def test_ranks_by_title_and_content(self, tmp_path: Path) -> None:
        _write_page(tmp_path, "pesq", "PESQ", "Speech quality measurement algorithm.")
        _write_page(tmp_path, "acr", "ACR", "Subjective opinion scale for speech.")
        wiki = load_wiki_from_files(tmp_path)
        assert wiki is not None
        retriever = KeywordRetriever(wiki)
        results = retriever.search("speech quality measurement", top_k=5)
        assert results
        assert results[0].article_id == "pesq"

    def test_no_match_returns_empty(self, tmp_path: Path) -> None:
        _write_page(tmp_path, "pesq", "PESQ", "Speech quality.")
        wiki = load_wiki_from_files(tmp_path)
        assert wiki is not None
        assert KeywordRetriever(wiki).search("quantum chromodynamics", top_k=5) == []

    def test_surface_compatible_with_hybrid(self, tmp_path: Path) -> None:
        _write_page(tmp_path, "pesq", "PESQ", "Speech quality.")
        wiki = load_wiki_from_files(tmp_path)
        assert wiki is not None
        r = KeywordRetriever(wiki)
        assert r.search_wiki("speech", top_k=1) == r.search("speech", top_k=1)
        assert r.search_hybrid("speech", top_k=1) == r.search("speech", top_k=1)


def _write_page(base: Path, slug: str, title: str, body: str, *, kind: str = "concept") -> None:
    folder = base / "concepts" if kind == "concept" else base
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{slug}.md").write_text(
        f"---\ntitle: {title}\ntype: Concept\nkind: {kind}\ndescription: {body[:60]}\n---\n\n# {title}\n\n{body}\n",
        encoding="utf-8",
    )
