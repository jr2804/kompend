"""Tests for output document model and renderer (N14, N13)."""

from __future__ import annotations

from pathlib import Path

from kompend.core.models import CompiledWiki, WikiNode
from kompend.output.markdown import MarkdownWriter
from kompend.output.models import Document
from kompend.output.renderer import (
    DocumentRenderer,
    detect_flavor,
    render_frontmatter,
)


class TestDocument:
    def test_minimal(self) -> None:
        doc = Document(title="Test", body="Hello")
        assert doc.title == "Test"
        assert doc.body == "Hello"
        assert doc.frontmatter == {}
        assert doc.heading_level == 1

    def test_with_frontmatter(self) -> None:
        doc = Document(title="Page", body="Content", frontmatter={"lang": "en"})
        assert doc.frontmatter["lang"] == "en"

    def test_with_source_chunks(self) -> None:
        doc = Document(title="Page", body="Content", source_chunks=["a.pdf:42", "b.pdf:10"])
        assert len(doc.source_chunks or []) == 2


class TestRenderFrontmatter:
    def test_empty(self) -> None:
        assert render_frontmatter({}) == ""

    def test_single_field(self) -> None:
        result = render_frontmatter({"lang": "en"})
        assert "---" in result
        assert "lang: en" in result

    def test_multiple_fields(self) -> None:
        result = render_frontmatter({"lang": "en", "domain": "ml"})
        assert "lang: en" in result
        assert "domain: ml" in result


class TestDetectFlavor:
    def test_default(self) -> None:
        assert detect_flavor() == "obsidian"

    def test_explicit(self) -> None:
        assert detect_flavor(configured="gfm") == "gfm"

    def test_unknown_flavor_falls_back(self) -> None:
        assert detect_flavor(configured="nonexistent") == "obsidian"

    def test_case_insensitive(self) -> None:
        assert detect_flavor(configured="GFM") == "gfm"
        assert detect_flavor(configured="Obsidian") == "obsidian"

    def test_github_alias(self) -> None:
        assert detect_flavor(configured="github") == "gfm"


class TestDocumentRenderer:
    def test_render_simple(self) -> None:
        renderer = DocumentRenderer(flavor="obsidian")
        doc = Document(title="Test Page", body="Some content here.")
        result = renderer.render(doc)
        assert "# Test Page" in result
        assert "Some content here." in result

    def test_render_with_frontmatter(self) -> None:
        renderer = DocumentRenderer(flavor="obsidian")
        doc = Document(title="Page", body="Content", frontmatter={"lang": "en"})
        result = renderer.render(doc)
        assert "---" in result
        assert "lang: en" in result
        assert "# Page" in result

    def test_render_all(self) -> None:
        renderer = DocumentRenderer(flavor="obsidian")
        docs = [
            Document(title="First", body="First content"),
            Document(title="Second", body="Second content"),
        ]
        result = renderer.render_all(docs)
        assert "# First" in result
        assert "# Second" in result
        assert "First content" in result
        assert "Second content" in result

    def test_render_empty_body(self) -> None:
        renderer = DocumentRenderer(flavor="obsidian")
        doc = Document(title="Empty Page", body="")
        result = renderer.render(doc)
        assert "# Empty Page" in result

    def test_different_flavors(self) -> None:
        """Different flavors should produce different output (if rumdl available)."""
        renderer_obsidian = DocumentRenderer(flavor="obsidian")
        renderer_gfm = DocumentRenderer(flavor="gfm")
        doc = Document(title="Test", body="Some **bold** text")
        # Both produce valid markdown (rumdl may or may not be installed)
        result_obs = renderer_obsidian.render(doc)
        result_gfm = renderer_gfm.render(doc)
        assert result_obs
        assert result_gfm

    def test_heading_level(self) -> None:
        renderer = DocumentRenderer(flavor="obsidian")
        doc = Document(title="Sub Section", body="Content", heading_level=2)
        result = renderer.render(doc)
        assert "## Sub Section" in result


class TestDetectFlavorFromDir:
    def test_obsidian_vault(self, tmp_path: Path) -> None:
        (tmp_path / ".obsidian").mkdir()
        assert detect_flavor(output_dir=tmp_path) == "obsidian"

    def test_mkdocs_project(self, tmp_path: Path) -> None:
        (tmp_path / "mkdocs.yml").write_text("site_name: Test")
        assert detect_flavor(output_dir=tmp_path) == "mkdocs"

    def test_quarto_project(self, tmp_path: Path) -> None:
        (tmp_path / "_quarto.yml").write_text("project:\n  type: book")
        assert detect_flavor(output_dir=tmp_path) == "quarto"

    def test_flavor_precedence(self, tmp_path: Path) -> None:
        """Explicit flavor overrides auto-detection."""
        (tmp_path / "mkdocs.yml").write_text("site_name: Test")
        assert detect_flavor(output_dir=tmp_path, configured="gfm") == "gfm"

    def test_quarto_qmd_files(self, tmp_path: Path) -> None:
        """Auto-detect quarto from .qmd files."""
        (tmp_path / "index.qmd").write_text("---\ntitle: Test\n---")
        assert detect_flavor(output_dir=tmp_path) == "quarto"


class TestMarkdownWriterRegression:
    """Regression: refactored MarkdownWriter produces valid output."""

    def test_writes_valid_markdown_structure(self, tmp_path: Path) -> None:
        """MarkdownWriter.write() produces a file with expected structure."""
        wiki = CompiledWiki(
            nodes=[
                WikiNode(
                    id="test-page",
                    title="Test Page",
                    content="This is the body.",
                    node_type="concept",
                ),
            ],
            edges=[],
            toc=[{"id": "test-page", "title": "Test Page", "level": 1}],
            metadata={"title": "Test Wiki"},
        )
        writer = MarkdownWriter(flavor="obsidian", vault_mode=False)
        output_file = writer.write(wiki, tmp_path)

        text = output_file.read_text(encoding="utf-8")
        assert "# Test Wiki" in text
        assert "# Test Page" in text or "## Test Page" in text
        assert "This is the body." in text
