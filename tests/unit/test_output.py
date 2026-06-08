"""Tests for KNOX output writers (Milestone 5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from knox.core.models import CompiledWiki, WikiNode
from knox.output.json_writer import JsonWriter
from knox.output.markdown import MarkdownWriter
from knox.output.typst import TypstWriter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_wiki() -> CompiledWiki:
    """A CompiledWiki with a few nodes, edges, and TOC entries."""
    nodes = [
        WikiNode(
            id="intro",
            title="Introduction",
            content="This is the introduction section.",
            node_type="section",
            source_chunks=["chunk-1", "chunk-2"],
            metadata={"heading_level": 1},
        ),
        WikiNode(
            id="concept-1",
            title="Core Concepts",
            content="Understanding the fundamental ideas.",
            node_type="concept",
            source_chunks=["chunk-3"],
            metadata={"heading_level": 2},
        ),
        WikiNode(
            id="procedure-1",
            title="How to Use",
            content="Step-by-step instructions.",
            node_type="procedure",
            source_chunks=None,
            metadata={},
        ),
    ]
    edges = [
        {"source": "intro", "target": "concept-1", "relation": "contains"},
        {"source": "concept-1", "target": "procedure-1", "relation": "prerequisite"},
    ]
    toc = [
        {"id": "intro", "title": "Introduction"},
        {"id": "concept-1", "title": "Core Concepts"},
        {"id": "procedure-1", "title": "How to Use"},
    ]
    metadata = {"title": "Test Wiki", "description": "A wiki for testing"}
    return CompiledWiki(nodes=nodes, edges=edges, toc=toc, metadata=metadata)


@pytest.fixture
def empty_wiki() -> CompiledWiki:
    """A CompiledWiki with no nodes, edges, or TOC."""
    return CompiledWiki(nodes=[], edges=[], toc=[], metadata={})


@pytest.fixture
def temp_output(tmp_path: Path) -> Path:
    """Temporary output directory."""
    return tmp_path / "output"


# ---------------------------------------------------------------------------
# MarkdownWriter Tests
# ---------------------------------------------------------------------------


class TestMarkdownWriter:
    def test_writes_valid_markdown(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = MarkdownWriter(vault_mode=False)
        result_path = writer.write(sample_wiki, temp_output)

        assert result_path == temp_output / "wiki.md"
        assert result_path.exists()

        content = result_path.read_text(encoding="utf-8")
        assert "# Test Wiki" in content
        assert "# Introduction" in content  # heading_level=1 -> prefix "#"
        assert "## Core Concepts" in content  # heading_level=2 -> prefix "##"
        assert "# How to Use" in content  # default heading_level=1 -> prefix "#"
        assert "This is the introduction section." in content
        assert "Understanding the fundamental ideas." in content
        assert "Step-by-step instructions." in content

    def test_handles_empty_wiki(self, empty_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = MarkdownWriter(vault_mode=False)
        result_path = writer.write(empty_wiki, temp_output)

        assert result_path.exists()
        content = result_path.read_text(encoding="utf-8")
        assert "# Compiled Wiki" in content
        assert "Table of Contents" not in content

    def test_creates_output_dir(self, sample_wiki: CompiledWiki, tmp_path: Path) -> None:
        deep_dir = tmp_path / "a" / "b" / "c"
        writer = MarkdownWriter(vault_mode=False)
        result_path = writer.write(sample_wiki, deep_dir)

        assert result_path.exists()
        assert deep_dir.exists()

    def test_vault_mode_writes_one_file_per_node(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = MarkdownWriter(vault_mode=True)
        result_path = writer.write(sample_wiki, temp_output)

        assert result_path == temp_output / "index.md"
        assert result_path.exists()
        assert (temp_output / "intro.md").exists()
        assert (temp_output / "concept-1.md").exists()
        assert (temp_output / "procedure-1.md").exists()

    def test_vault_mode_index_contains_wikilinks(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = MarkdownWriter(vault_mode=True)
        writer.write(sample_wiki, temp_output)

        index = (temp_output / "index.md").read_text(encoding="utf-8")
        assert "# Test Wiki" in index
        assert "Introduction" in index
        assert "Core Concepts" in index
        assert "How to Use" in index

    def test_vault_mode_related_section(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = MarkdownWriter(vault_mode=True)
        writer.write(sample_wiki, temp_output)

        intro = (temp_output / "intro.md").read_text(encoding="utf-8")
        assert "## Related" in intro
        assert "[[Core Concepts]]" in intro

    def test_vault_mode_empty_wiki(self, empty_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = MarkdownWriter(vault_mode=True)
        result_path = writer.write(empty_wiki, temp_output)

        assert result_path == temp_output / "index.md"
        index = (temp_output / "index.md").read_text(encoding="utf-8")
        assert "# Wiki Index" in index

    def test_vault_mode_single_node(self, tmp_path: Path) -> None:
        from knox.core.models import CompiledWiki, WikiNode

        wiki = CompiledWiki(
            nodes=[
                WikiNode(id="test", title="Test Page", content="Hello.", node_type="concept"),
            ],
            edges=[],
            toc=[],
        )
        writer = MarkdownWriter(vault_mode=True)
        result_path = writer.write(wiki, tmp_path)

        assert result_path == tmp_path / "index.md"
        assert (tmp_path / "test.md").exists()
        content = (tmp_path / "test.md").read_text(encoding="utf-8")
        assert "# Test Page" in content
        assert "Hello." in content
        assert "## Related" not in content


# ---------------------------------------------------------------------------
# JsonWriter Tests
# ---------------------------------------------------------------------------


class TestJsonWriter:
    def test_writes_valid_json(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        import json

        writer = JsonWriter()
        result_path = writer.write(sample_wiki, temp_output)

        assert result_path == temp_output / "wiki.json"
        assert result_path.exists()

        data = json.loads(result_path.read_text(encoding="utf-8"))
        assert len(data["nodes"]) == 3
        assert data["nodes"][0]["id"] == "intro"
        assert data["nodes"][0]["title"] == "Introduction"
        assert data["nodes"][0]["content"] == "This is the introduction section."
        assert data["nodes"][0]["node_type"] == "section"
        assert data["nodes"][0]["source_chunks"] == ["chunk-1", "chunk-2"]
        assert data["nodes"][0]["metadata"]["heading_level"] == 1

        assert len(data["edges"]) == 2
        assert data["edges"][0]["source"] == "intro"
        assert data["edges"][0]["target"] == "concept-1"

        assert len(data["toc"]) == 3
        assert data["toc"][1]["title"] == "Core Concepts"

        assert data["metadata"]["title"] == "Test Wiki"
        assert data["quality_score"] == 0.0
        assert data["iteration"] == 0

    def test_handles_empty_wiki(self, empty_wiki: CompiledWiki, temp_output: Path) -> None:
        import json

        writer = JsonWriter()
        result_path = writer.write(empty_wiki, temp_output)

        data = json.loads(result_path.read_text(encoding="utf-8"))
        assert data["nodes"] == []
        assert data["edges"] == []
        assert data["toc"] == []
        assert data["metadata"] == {}

    def test_creates_output_dir(self, sample_wiki: CompiledWiki, tmp_path: Path) -> None:
        writer = JsonWriter()
        result_path = writer.write(sample_wiki, tmp_path / "deep" / "dir")
        assert result_path.exists()


# ---------------------------------------------------------------------------
# TypstWriter Tests
# ---------------------------------------------------------------------------


class TestTypstWriter:
    def test_writes_valid_typst_source(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = TypstWriter()
        result_path = writer.write(sample_wiki, temp_output)

        assert result_path == temp_output / "wiki.typ"
        assert result_path.exists()

        content = result_path.read_text(encoding="utf-8")
        # Document setup
        assert '#set page("a4")' in content
        assert "#set text(size: 11pt)" in content

        # Title page
        assert "#align(center + top)" in content
        assert "Test Wiki" in content

        # TOC
        assert "= Table of Contents" in content
        assert "#outline()" in content

        # Node sections with labels
        assert "#label(<intro>)" in content
        assert "== Introduction" in content  # heading_level=1 -> prefix "=="
        assert "#label(<concept-1>)" in content
        assert "=== Core Concepts" in content  # heading_level=2 -> prefix "==="
        assert "#label(<procedure-1>)" in content

        # Content
        assert "This is the introduction section." in content
        assert "Step-by-step instructions." in content

        # Source chunk references
        assert "*Source: 2 chunk(s)*" in content

        # Back links
        assert "Back to Table of Contents" in content
        assert "#pagebreak()" in content

    def test_handles_empty_wiki(self, empty_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = TypstWriter()
        result_path = writer.write(empty_wiki, temp_output)

        assert result_path.exists()
        content = result_path.read_text(encoding="utf-8")
        assert '#set page("a4")' in content
        assert "Compiled Wiki" in content  # fallback title
        assert "#pagebreak()" in content

    def test_wiki_without_metadata_title(self, temp_output: Path) -> None:
        """Fallback title 'Compiled Wiki' when no metadata title."""
        wiki = CompiledWiki(
            nodes=[WikiNode(id="n1", title="Node", content="Content", node_type="section")],
            edges=[],
            toc=[],
            metadata={},
        )
        writer = TypstWriter()
        result_path = writer.write(wiki, temp_output)
        content = result_path.read_text(encoding="utf-8")
        assert "Compiled Wiki" in content

    def test_creates_output_dir(self, sample_wiki: CompiledWiki, tmp_path: Path) -> None:
        writer = TypstWriter()
        result_path = writer.write(sample_wiki, tmp_path / "nested" / "path")
        assert result_path.exists()
