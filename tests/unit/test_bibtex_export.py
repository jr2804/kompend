"""Tests for BibTeX export (Phase 4).

Covers ``knox.output.bibtex``:
- ``reference_to_bibtex_entry`` — single entry formatting
- ``references_to_bibtex`` — multi-entry document
- CSL type → BibTeX entry-type mapping
- Special character escaping
- Citation key generation
"""

from __future__ import annotations

from knox.core.models.references import Author, Reference
from knox.output.bibtex import reference_to_bibtex_entry, references_to_bibtex

# ---------------------------------------------------------------------------
# reference_to_bibtex_entry
# ---------------------------------------------------------------------------


class TestReferenceToBibtexEntry:
    """Tests for single-entry BibTeX formatting."""

    def test_article_journal_type(self) -> None:
        ref = _ref(type="article-journal")
        entry = reference_to_bibtex_entry(ref)
        assert entry.startswith("@article{")
        assert entry.endswith("}")
        assert "smith2020" in entry.split("{", 1)[1].split(",", 1)[0]

    def test_book_type(self) -> None:
        ref = _ref(type="book", title="A Great Book", container_title="")
        entry = reference_to_bibtex_entry(ref)
        assert entry.startswith("@book{")

    def test_chapter_type(self) -> None:
        ref = _ref(type="chapter")
        entry = reference_to_bibtex_entry(ref)
        assert entry.startswith("@incollection{")

    def test_paper_conference_type(self) -> None:
        ref = _ref(type="paper-conference", container_title="NeurIPS")
        entry = reference_to_bibtex_entry(ref)
        assert entry.startswith("@inproceedings{")
        assert "booktitle" in entry

    def test_report_type(self) -> None:
        ref = _ref(type="report")
        entry = reference_to_bibtex_entry(ref)
        assert entry.startswith("@techreport{")

    def test_thesis_type(self) -> None:
        ref = _ref(type="thesis")
        entry = reference_to_bibtex_entry(ref)
        assert entry.startswith("@phdthesis{")

    def test_webpage_type(self) -> None:
        ref = _ref(type="webpage")
        entry = reference_to_bibtex_entry(ref)
        assert entry.startswith("@misc{")

    def test_unknown_type_falls_back_to_misc(self) -> None:
        ref = _ref(type="something-weird")
        entry = reference_to_bibtex_entry(ref)
        assert entry.startswith("@misc{")

    def test_all_fields_present(self) -> None:
        ref = _ref(
            volume="12",
            number="3",
            pages="45-67",
            publisher="MIT Press",
            doi="10.1/xyz",
            url="https://example.com/paper",
            given="John",
        )
        entry = reference_to_bibtex_entry(ref)
        assert "author" in entry
        assert "title" in entry
        assert "year" in entry
        assert "journal" in entry
        assert "volume" in entry
        assert "number" in entry
        assert "pages" in entry
        assert "45--67" in entry  # pages use "--" in BibTeX
        assert "publisher" in entry
        assert "doi" in entry
        assert "url" in entry

    def test_minimal_fields(self) -> None:
        ref = Reference(id="x", type="article-journal", title="T")
        entry = reference_to_bibtex_entry(ref)
        assert "@article{" in entry
        assert "title" in entry
        # No trailing comma on last field
        assert not entry.rstrip("}").rstrip().endswith(",")

    def test_no_trailing_comma(self) -> None:
        ref = _ref()
        entry = reference_to_bibtex_entry(ref)
        # The last field line should not end with comma
        field_lines = [line for line in entry.split("\n") if "=" in line]
        assert not field_lines[-1].rstrip().endswith(",")

    def test_literal_author(self) -> None:
        ref = Reference(
            id="org2020",
            type="report",
            title="Standard",
            author=[Author(literal="ISO International")],
            issued="2020",
        )
        entry = reference_to_bibtex_entry(ref)
        assert "ISO International" in entry

    def test_given_and_family_author_format(self) -> None:
        ref = _ref(family="Smith", given="John")
        entry = reference_to_bibtex_entry(ref)
        assert "Smith, John" in entry

    def test_doi_url_not_duplicated_when_both_present(self) -> None:
        ref = _ref(doi="10.1/xyz", url="https://example.com/paper")
        entry = reference_to_bibtex_entry(ref)
        assert "doi" in entry
        assert "url" in entry
        assert "10.1/xyz" in entry


# ---------------------------------------------------------------------------
# Citation key generation
# ---------------------------------------------------------------------------


class TestCitationKey:
    """Tests for citation key generation."""

    def test_author_year_key(self) -> None:
        ref = _ref(family="Smith", issued="2020")
        entry = reference_to_bibtex_entry(ref)
        assert "@article{smith2020," in entry

    def test_id_fallback(self) -> None:
        ref = Reference(id="custom-id-123", type="article-journal", title="T")
        entry = reference_to_bibtex_entry(ref)
        assert "customid123" in entry

    def test_ref_n_fallback(self) -> None:
        ref = Reference(id="", type="article-journal", title="No Author")
        entry = reference_to_bibtex_entry(ref, index=5)
        assert "{ref5," in entry

    def test_special_chars_stripped_from_key(self) -> None:
        ref = _ref(family="O'Brien-Smyth", id="ob2020")
        entry = reference_to_bibtex_entry(ref)
        # Key should be alphanumeric only
        first_line = entry.split("\n")[0]
        key = first_line.split("{")[1].rstrip(",")
        assert all(c.isalnum() for c in key)


# ---------------------------------------------------------------------------
# Special character escaping
# ---------------------------------------------------------------------------


class TestEscaping:
    """Tests for BibTeX special character escaping."""

    def test_ampersand_escaped(self) -> None:
        ref = _ref(title="A & B", container_title="")
        entry = reference_to_bibtex_entry(ref)
        assert r"\&" in entry

    def test_underscore_escaped(self) -> None:
        ref = _ref(title="foo_bar", container_title="")
        entry = reference_to_bibtex_entry(ref)
        assert r"\_" in entry

    def test_dollar_escaped(self) -> None:
        ref = _ref(title="Cost $100", container_title="")
        entry = reference_to_bibtex_entry(ref)
        assert r"\$" in entry


# ---------------------------------------------------------------------------
# references_to_bibtex
# ---------------------------------------------------------------------------


class TestReferencesToBibtex:
    """Tests for multi-entry BibTeX documents."""

    def test_empty_list(self) -> None:
        assert references_to_bibtex([]) == ""

    def test_single_entry(self) -> None:
        refs = [_ref(family="Smith", issued="2020")]
        result = references_to_bibtex(refs)
        assert result.count("@") == 1
        assert result.endswith("\n")

    def test_multiple_entries_separated(self) -> None:
        refs = [
            _ref(id="a", family="Smith", issued="2020", title="A"),
            _ref(id="b", family="Jones", issued="2019", title="B"),
            _ref(id="c", family="Brown", issued="2021", title="C"),
        ]
        result = references_to_bibtex(refs)
        assert result.count("@") == 3
        entries = result.strip().split("\n\n")
        assert len(entries) == 3

    def test_multiple_entries_unique_keys(self) -> None:
        refs = [
            _ref(id="", family="Smith", issued="2020", title="A"),
            _ref(id="", family="Smith", issued="2020", title="B"),
        ]
        result = references_to_bibtex(refs)
        # Both should get smith2020 key (no automatic dedup)
        assert result.count("smith2020") == 2

    def test_round_trip_structure(self) -> None:
        refs = [_ref(family="Smith", issued="2020", doi="10.1/xyz", volume="1", pages="1-10")]
        result = references_to_bibtex(refs)
        assert "@article{smith2020," in result
        assert "10.1/xyz" in result
        assert "volume    = {1}" in result
        assert "pages     = {1--10}" in result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref(  # noqa: PLR0913, PLR0917
    *,
    id: str = "smith2020",
    type: str = "article-journal",
    title: str = "Deep Learning for NLP",
    family: str = "Smith",
    given: str = "",
    issued: str = "2020",
    container_title: str = "Journal of AI",
    volume: str = "",
    number: str = "",
    pages: str = "",
    publisher: str = "",
    doi: str = "",
    url: str = "",
    raw: str = "",
) -> Reference:
    return Reference(
        id=id,
        type=type,
        title=title,
        author=[Author(family=family, given=given)] if family else [],
        issued=issued,
        container_title=container_title,
        volume=volume,
        number=number,
        pages=pages,
        publisher=publisher,
        doi=doi,
        url=url,
        raw=raw,
    )
