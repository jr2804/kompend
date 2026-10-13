"""Tests for reference-section detection and excision."""

from __future__ import annotations

import pytest

from kompend.compilation.reference_excision import (
    excise_references,
    is_likely_reference_heading,
)

# ---------------------------------------------------------------------------
# is_likely_reference_heading
# ---------------------------------------------------------------------------


class TestIsLikelyReferenceHeading:
    @pytest.mark.parametrize(
        "line",
        [
            "# References",
            "## References",
            "### References",
            "# Bibliography",
            "## Works Cited",
            "### Literature Cited",
            "# References and Notes",
            "## Cited References",
            "# Sources",
            "10. References",
            "8. Bibliography",
            "10 References",
        ],
    )
    def test_matches(self, line: str) -> None:
        assert is_likely_reference_heading(line) is True

    @pytest.mark.parametrize(
        "line",
        [
            "# Introduction",
            "## Methods",
            "# Related Work",
            "Some text about references here.",
            "",
            "References are important.",
        ],
    )
    def test_no_match(self, line: str) -> None:
        assert is_likely_reference_heading(line) is False


# ---------------------------------------------------------------------------
# excise_references
# ---------------------------------------------------------------------------


class TestExciseReferences:
    def test_basic_atx_heading(self) -> None:
        text = (
            "# Introduction\n\nSome intro text.\n\n"
            "## Method\n\nWe did things.\n\n"
            "# References\n\n"
            "[1] Smith, J. (2020). A paper. Journal, 1(1), 1-10.\n"
            "[2] Doe, J. (2019). Another paper. arXiv.\n"
        )
        body, refs = excise_references(text)
        assert "Introduction" in body
        assert "Method" in body
        assert "References" not in body
        assert "[1] Smith" in refs
        assert "[2] Doe" in refs

    def test_numbered_heading(self) -> None:
        text = "Intro text.\n\n10. References\n\n[1] Foo, A. (2021). Title.\n"
        body, refs = excise_references(text)
        assert "Intro text" in body
        assert "10. References" in refs
        assert "[1] Foo" in refs

    def test_no_reference_section(self) -> None:
        text = "# Title\n\nJust some content without references.\n"
        body, refs = excise_references(text)
        assert body == text
        assert refs == ""

    def test_bibliography_variant(self) -> None:
        text = "Content here.\n\n## Bibliography\n\nItem 1.\nItem 2.\n"
        body, refs = excise_references(text)
        assert "Content here" in body
        assert "Bibliography" in refs
        assert "Item 1" in refs

    def test_empty_text(self) -> None:
        body, refs = excise_references("")
        assert body == ""
        assert refs == ""

    def test_reference_section_is_terminal(self) -> None:
        """Everything from the heading to the end is the ref section."""
        text = "# Intro\n\nBody.\n\n# References\n\n[1] Ref one.\n\n## Appendix\n\nMore stuff after refs.\n"
        body, refs = excise_references(text)
        assert "Intro" in body
        assert "Body" in body
        # The appendix (which comes after the references heading) is
        # included in the references section because bibliographies are
        # terminal — the excision is conservative.
        assert "Appendix" in refs
        assert "[1] Ref one" in refs

    def test_works_cited(self) -> None:
        text = "Body.\n\n# Works Cited\n\nSmith 2020. Title.\n"
        body, refs = excise_references(text)
        assert "Body" in body
        assert "Works Cited" in refs

    def test_preserves_body_whitespace(self) -> None:
        text = "# Intro\n\n   \n\n# References\n\n[1] X.\n"
        body, refs = excise_references(text)
        assert body == "# Intro"
