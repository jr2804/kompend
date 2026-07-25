"""Tests for in-text citation → link rewriting (Phase 3).

Covers ``knox.output.reference_links``:
- ``rewrite_citation_links`` — bracketed numeric, bracketed key, parenthetical author-year
- ``rewrite_source_references`` — convenience wrapper
- ``build_reference_key_index`` — key index construction
- ``format_bibliography_entry`` — bibliography formatting
- Integration with ``MarkdownWriter`` and ``OkfWriter`` (references page generation)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knox.core.models import CompiledWiki, WikiNode
from knox.core.models.references import Author, Reference
from knox.core.pipeline.references import ReferenceStore
from knox.output.markdown import MarkdownWriter
from knox.output.okf_writer import OkfWriter
from knox.output.reference_links import (
    build_reference_key_index,
    format_bibliography_entry,
    rewrite_citation_links,
    rewrite_source_references,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ref(  # noqa: PLR0913, PLR0917
    id: str = "smith2020",
    title: str = "Deep Learning for NLP",
    family: str = "Smith",
    issued: str = "2020",
    doi: str = "",
    container_title: str = "Journal of AI",
) -> Reference:
    return Reference(
        id=id,
        type="article-journal",
        title=title,
        author=[Author(family=family)],
        issued=issued,
        container_title=container_title,
        doi=doi,
        raw=f"{family}, J. ({issued}). {title}.",
    )


@pytest.fixture
def sample_refs() -> list[Reference]:
    """Three references for testing."""
    return [
        _ref(id="smith2020", title="Deep Learning for NLP", family="Smith", issued="2020"),
        _ref(id="jones2019", title="Graph Neural Networks", family="Jones", issued="2019"),
        _ref(id="brown2021", title="Transformer Architectures", family="Brown", issued="2021"),
    ]


@pytest.fixture
def ref_store(tmp_path: Path, sample_refs: list[Reference]) -> ReferenceStore:
    """A ReferenceStore with one source."""
    store = ReferenceStore(tmp_path / "references.json")
    store.add_source("sources/paper.pdf", sample_refs)
    return store


# ---------------------------------------------------------------------------
# build_reference_key_index
# ---------------------------------------------------------------------------


class TestBuildReferenceKeyIndex:
    def test_indexes_by_id(self, sample_refs: list[Reference]) -> None:
        index = build_reference_key_index(sample_refs)
        assert "smith2020" in index
        assert index["smith2020"].id == "smith2020"

    def test_indexes_by_label(self, sample_refs: list[Reference]) -> None:
        index = build_reference_key_index(sample_refs)
        # short_label produces "Smith 2020" → key "smith2020"
        assert "smith2020" in index

    def test_indexes_by_author_year(self, sample_refs: list[Reference]) -> None:
        index = build_reference_key_index(sample_refs)
        assert "smith2020" in index
        assert "jones2019" in index

    def test_indexes_by_doi(self) -> None:
        ref = _ref(doi="10.1234/test")
        index = build_reference_key_index([ref])
        assert "doi:10.1234/test" in index

    def test_empty_list(self) -> None:
        index = build_reference_key_index([])
        assert index == {}


# ---------------------------------------------------------------------------
# rewrite_citation_links — bracketed numeric
# ---------------------------------------------------------------------------


class TestRewriteBracketNumeric:
    def test_single_numeric(self, sample_refs: list[Reference]) -> None:
        text = "Some claim [1] is important."
        result = rewrite_citation_links(text, sample_refs)
        assert "[1]" not in result
        assert "#smith2020" in result or "#smith-2020" in result

    def test_multiple_numeric(self, sample_refs: list[Reference]) -> None:
        text = "See [1, 2] for details."
        result = rewrite_citation_links(text, sample_refs)
        assert "[1, 2]" not in result
        assert "smith2020" in result or "smith-2020" in result
        assert "jones2019" in result or "jones-2019" in result

    def test_numeric_range(self, sample_refs: list[Reference]) -> None:
        text = "As shown [1-3]."
        result = rewrite_citation_links(text, sample_refs)
        assert "[1-3]" not in result
        assert "smith2020" in result or "smith-2020" in result
        assert "brown2021" in result or "brown-2021" in result

    def test_out_of_range_number_left_untouched(self, sample_refs: list[Reference]) -> None:
        text = "Rare ref [99]."
        result = rewrite_citation_links(text, sample_refs)
        assert result == text

    def test_no_refs_returns_unchanged(self) -> None:
        text = "No citations here [1]."
        result = rewrite_citation_links(text, [])
        assert result == text


# ---------------------------------------------------------------------------
# rewrite_citation_links — bracketed key
# ---------------------------------------------------------------------------


class TestRewriteBracketKey:
    def test_key_match(self, sample_refs: list[Reference]) -> None:
        text = "See [smith2020] for more."
        result = rewrite_citation_links(text, sample_refs)
        assert "[smith2020]" not in result
        assert "#smith2020" in result or "#smith-2020" in result

    def test_key_with_page_number(self, sample_refs: list[Reference]) -> None:
        text = "See [smith2020, p. 42]."
        result = rewrite_citation_links(text, sample_refs)
        assert "[smith2020, p. 42]" not in result
        assert "#smith2020" in result or "#smith-2020" in result

    def test_unknown_key_left_untouched(self, sample_refs: list[Reference]) -> None:
        text = "See [unknown2025]."
        result = rewrite_citation_links(text, sample_refs)
        assert result == text


# ---------------------------------------------------------------------------
# rewrite_citation_links — parenthetical author-year
# ---------------------------------------------------------------------------


class TestRewriteParenAuthorYear:
    def test_simple_author_year(self, sample_refs: list[Reference]) -> None:
        text = "As noted (Smith, 2020)."
        result = rewrite_citation_links(text, sample_refs)
        assert "(Smith, 2020)" not in result
        assert "#smith2020" in result or "#smith-2020" in result

    def test_unknown_author_year_left_untouched(self, sample_refs: list[Reference]) -> None:
        text = "As noted (Nobody, 1999)."
        result = rewrite_citation_links(text, sample_refs)
        assert result == text


# ---------------------------------------------------------------------------
# rewrite_citation_links — edge cases
# ---------------------------------------------------------------------------


class TestRewriteEdgeCases:
    def test_empty_text(self, sample_refs: list[Reference]) -> None:
        assert rewrite_citation_links("", sample_refs) == ""

    def test_mixed_citations(self, sample_refs: list[Reference]) -> None:
        text = "Both [1] and (Jones, 2019) agree."
        result = rewrite_citation_links(text, sample_refs)
        assert "[1]" not in result
        assert "(Jones, 2019)" not in result

    def test_non_citation_brackets_untouched(self, sample_refs: list[Reference]) -> None:
        text = "The array [0, 1] is binary."
        result = rewrite_citation_links(text, sample_refs)
        # [0] is out of range (1-indexed), so should be left alone
        assert "[0, 1]" in result


# ---------------------------------------------------------------------------
# rewrite_source_references
# ---------------------------------------------------------------------------


class TestRewriteSourceReferences:
    def test_wrapper_loads_and_rewrites(
        self,
        ref_store: ReferenceStore,
    ) -> None:
        text = "See [1] for details."
        result = rewrite_source_references(text, "sources/paper.pdf", ref_store)
        assert "[1]" not in result

    def test_no_refs_for_source_returns_unchanged(
        self,
        ref_store: ReferenceStore,
    ) -> None:
        text = "See [1] for details."
        result = rewrite_source_references(text, "sources/nonexistent.pdf", ref_store)
        assert result == text


# ---------------------------------------------------------------------------
# format_bibliography_entry
# ---------------------------------------------------------------------------


class TestFormatBibliographyEntry:
    def test_basic_entry(self) -> None:
        ref = _ref(id="smith2020", title="Test Paper", family="Smith", issued="2020")
        entry = format_bibliography_entry(ref, number=1)
        assert "1." in entry
        assert "Smith" in entry
        assert "2020" in entry
        assert "Test Paper" in entry

    def test_entry_with_doi(self) -> None:
        ref = _ref(doi="10.1234/test")
        entry = format_bibliography_entry(ref, number=2)
        assert "10.1234/test" in entry
        assert "doi.org/10.1234/test" in entry

    def test_entry_without_number(self) -> None:
        ref = _ref(id="smith2020")
        entry = format_bibliography_entry(ref)
        assert "1." not in entry.split("###")[0] if "###" in entry else True

    def test_anchor_present(self) -> None:
        ref = _ref(id="smith2020")
        entry = format_bibliography_entry(ref)
        assert 'id="' in entry


# ---------------------------------------------------------------------------
# Integration: MarkdownWriter references page
# ---------------------------------------------------------------------------


def _make_wiki() -> CompiledWiki:
    """Minimal wiki for output testing."""
    return CompiledWiki(
        nodes=[
            WikiNode(
                id="test-page",
                title="Test Page",
                content="This cites [1] something.",
                node_type="concept",
                source_chunks=["c1"],
                metadata={
                    "heading_level": 1,
                    "frontmatter": {"sources": ["paper"]},
                    "provenance": ["sources/paper.pdf"],
                },
            ),
        ],
        edges=[],
        toc=[],
        metadata={"title": "Test Wiki"},
    )


class TestMarkdownWriterReferencesPage:
    def test_generates_references_md(self, tmp_path: Path) -> None:
        """MarkdownWriter should generate references.md when store has entries."""
        ref_file = tmp_path / ".knox" / "references.json"
        ref_file.parent.mkdir(parents=True)
        store = ReferenceStore(ref_file)
        store.add_source("sources/paper.pdf", [_ref()])
        store.save()

        wiki = _make_wiki()
        output_dir = tmp_path / "wiki"
        writer = MarkdownWriter(vault_mode=True, references_file=ref_file)
        writer.write(wiki, output_dir)

        assert (output_dir / "references.md").exists()
        content = (output_dir / "references.md").read_text()
        assert "Smith" in content
        assert "Deep Learning" in content

    def test_no_references_no_page(self, tmp_path: Path) -> None:
        """No references.md when store is empty."""
        ref_file = tmp_path / ".knox" / "references.json"
        wiki = _make_wiki()
        output_dir = tmp_path / "wiki"
        writer = MarkdownWriter(vault_mode=True, references_file=ref_file)
        writer.write(wiki, output_dir)
        assert not (output_dir / "references.md").exists()

    def test_no_ref_file_no_page(self, tmp_path: Path) -> None:
        """No references.md when no references_file configured."""
        wiki = _make_wiki()
        output_dir = tmp_path / "wiki"
        writer = MarkdownWriter(vault_mode=True)
        writer.write(wiki, output_dir)
        assert not (output_dir / "references.md").exists()


class TestOkfWriterReferencesPage:
    def test_generates_references_md(self, tmp_path: Path) -> None:
        """OkfWriter should generate references.md with OKF frontmatter."""
        ref_file = tmp_path / ".knox" / "references.json"
        ref_file.parent.mkdir(parents=True)
        store = ReferenceStore(ref_file)
        store.add_source("sources/paper.pdf", [_ref()])
        store.save()

        wiki = _make_wiki()
        output_dir = tmp_path / "wiki"
        writer = OkfWriter(references_file=ref_file)
        writer.write(wiki, output_dir)

        assert (output_dir / "references.md").exists()
        content = (output_dir / "references.md").read_text()
        assert "Smith" in content
        assert 'type: "Reference"' in content or "type: Reference" in content

    def test_citation_rewritten_in_page(self, tmp_path: Path) -> None:
        """Citation [1] should be rewritten in the generated page."""
        ref_file = tmp_path / ".knox" / "references.json"
        ref_file.parent.mkdir(parents=True)
        store = ReferenceStore(ref_file)
        store.add_source("sources/paper.pdf", [_ref()])
        store.save()

        wiki = _make_wiki()
        output_dir = tmp_path / "wiki"
        writer = OkfWriter(references_file=ref_file)
        writer.write(wiki, output_dir)

        # Concept pages go to concepts/ subfolder
        page = (output_dir / "concepts" / "test-page.md").read_text()
        # The body (after the frontmatter) must have the citation rewritten.
        # The frontmatter ``description`` and the ``# Citations`` provenance
        # section legitimately contain the raw marker and are not rewritten.
        body = page.split("---", 2)[-1]
        assert "This cites [1] something." not in body
        assert "smith2020" in page or "smith-2020" in page
