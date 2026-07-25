"""Tests for the bibliographic Reference model (CSL JSON-aligned)."""

from __future__ import annotations

from knox.core.models.references import (
    Author,
    Reference,
    ReferenceExtractionOutput,
)

# ---------------------------------------------------------------------------
# Author
# ---------------------------------------------------------------------------


class TestAuthor:
    def test_to_csl_family_given(self) -> None:
        a = Author(family="Smith", given="John")
        assert a.to_csl() == {"family": "Smith", "given": "John"}

    def test_to_csl_literal(self) -> None:
        a = Author(literal="ACM")
        assert a.to_csl() == {"literal": "ACM"}

    def test_to_csl_empty(self) -> None:
        a = Author()
        assert a.to_csl() == {}

    def test_to_csl_literal_preferred(self) -> None:
        """When both literal and family are set, literal wins."""
        a = Author(family="Smith", given="J.", literal="Team Foo")
        assert a.to_csl() == {"literal": "Team Foo"}


# ---------------------------------------------------------------------------
# Reference.to_csl
# ---------------------------------------------------------------------------


class TestReferenceToCsl:
    def test_full_reference(self) -> None:
        ref = Reference(
            id="smith2020",
            type="article-journal",
            title="A Paper",
            author=[Author(family="Smith", given="J.")],
            issued="2020",
            container_title="Journal",
            volume="5",
            number="2",
            pages="1-10",
            publisher="Pub",
            doi="10.1000/abc",
            url="https://example.com",
        )
        csl = ref.to_csl()
        assert csl["id"] == "smith2020"
        assert csl["type"] == "article-journal"
        assert csl["title"] == "A Paper"
        assert csl["author"] == [{"family": "Smith", "given": "J."}]
        assert csl["issued"] == {"raw": "2020"}
        assert csl["container-title"] == "Journal"
        assert csl["volume"] == "5"
        assert csl["issue"] == "2"
        assert csl["page"] == "1-10"
        assert csl["publisher"] == "Pub"
        assert csl["DOI"] == "10.1000/abc"
        assert csl["URL"] == "https://example.com"

    def test_empty_fields_excluded(self) -> None:
        ref = Reference(id="x", type="book")
        csl = ref.to_csl()
        assert csl == {"id": "x", "type": "book"}

    def test_knox_fields_not_in_csl(self) -> None:
        ref = Reference(
            id="x",
            raw="[1] Some citation",
            citing_source="paper.pdf",
            in_text_keys=["smith2020"],
        )
        csl = ref.to_csl()
        assert "raw" not in csl
        assert "citing_source" not in csl
        assert "in_text_keys" not in csl


# ---------------------------------------------------------------------------
# Reference.display_authors
# ---------------------------------------------------------------------------


class TestDisplayAuthors:
    def test_family_given(self) -> None:
        ref = Reference(author=[Author(family="Smith", given="J.")])
        assert ref.display_authors() == "J. Smith"

    def test_family_only(self) -> None:
        ref = Reference(author=[Author(family="Doe")])
        assert ref.display_authors() == "Doe"

    def test_literal(self) -> None:
        ref = Reference(author=[Author(literal="ACM")])
        assert ref.display_authors() == "ACM"

    def test_multiple_authors(self) -> None:
        ref = Reference(
            author=[
                Author(family="Smith", given="J."),
                Author(family="Doe", given="A."),
            ]
        )
        assert ref.display_authors() == "J. Smith, A. Doe"

    def test_no_authors(self) -> None:
        ref = Reference()
        assert ref.display_authors() == ""


# ---------------------------------------------------------------------------
# Reference.short_label
# ---------------------------------------------------------------------------


class TestShortLabel:
    def test_author_year(self) -> None:
        ref = Reference(
            author=[Author(family="Smith")],
            issued="2020",
        )
        assert ref.short_label() == "Smith 2020"

    def test_literal_author_year(self) -> None:
        ref = Reference(
            author=[Author(literal="ACM")],
            issued="2019",
        )
        assert ref.short_label() == "ACM 2019"

    def test_no_author_uses_year(self) -> None:
        ref = Reference(title="A Very Long Title About Stuff", issued="2020")
        # With no author but a year, short_label returns just the year
        assert ref.short_label() == "2020"

    def test_no_author_no_year_falls_back_to_title(self) -> None:
        ref = Reference(title="A Very Long Title About Stuff")
        label = ref.short_label()
        assert "A Very Long Title About Stuff"[:40] in label

    def test_completely_empty_falls_back_to_id(self) -> None:
        ref = Reference(id="ref42")
        assert ref.short_label() == "ref42"

    def test_extracts_year_from_longer_date(self) -> None:
        ref = Reference(author=[Author(family="X")], issued="Published 2021-03")
        assert ref.short_label() == "X 2021"


# ---------------------------------------------------------------------------
# ReferenceExtractionOutput
# ---------------------------------------------------------------------------


class TestReferenceExtractionOutput:
    def test_empty(self) -> None:
        out = ReferenceExtractionOutput()
        assert out.references == []

    def test_with_refs(self) -> None:
        refs = [
            Reference(id="a", type="book", title="A"),
            Reference(id="b", type="article-journal", title="B"),
        ]
        out = ReferenceExtractionOutput(references=refs)
        assert len(out.references) == 2
        assert out.references[0].id == "a"
        assert out.references[1].type == "article-journal"

    def test_validate_from_dict(self) -> None:
        out = ReferenceExtractionOutput.model_validate({"references": [{"id": "x", "type": "book", "title": "T"}]})
        assert len(out.references) == 1
        assert out.references[0].id == "x"
