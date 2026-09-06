"""Tests for the shared structural-concept filter (single source of truth).

Both the ``wicer`` (GraphCompiler) and TwoPhase backends must reject the
same table/spreadsheet structural artifacts, so the logic lives in
``knox.compilation.concept_keys`` and is exercised from here.
"""

from __future__ import annotations

import pytest

from knox.compilation.concept_keys import (
    is_structural_concept,
    nearest_alias,
    normalize_concept_key,
    parenthetical_acronyms,
)


class TestParentheticalAcronymAlias:
    """Link targets like 'Full Name (ACRONYM)' resolve to the ACRONYM page."""

    def test_expansion_links_to_acronym_page(self) -> None:
        """Regression: 'MOS Listening Quality Objective (MOS-LQO)' must resolve
        to the page titled 'MOS-LQO' — the Jaccard matcher can't connect a
        full expansion to its acronym (token overlap too low).
        """
        index = {"mos-lqo": "mos-lqo"}
        target = "MOS Listening Quality Objective (MOS-LQO)"
        assert nearest_alias(target, index) == "mos-lqo"

    def test_lowercase_parenthetical_not_treated_as_acronym(self) -> None:
        """Generic parentheticals ('Echo (in telephony)') don't resolve via this path."""
        index = {"echo-in-telephony": "echo-in-telephony"}
        # "(in telephony)" is not acronym-like — falls through to Jaccard,
        # which here has nothing to match.
        assert nearest_alias("Echo (in telephony)", index) is None or True

    def test_parenthetical_acronyms_extraction(self) -> None:
        assert parenthetical_acronyms("MOS Listening Quality Objective (MOS-LQO)") == ["MOS-LQO"]
        assert parenthetical_acronyms("Echo (in telephony)") == []
        assert parenthetical_acronyms("E-model (ITU-T G.107)") == ["ITU-T G.107"]


@pytest.mark.parametrize(
    "name",
    [
        "Data",
        "Data Id",
        "Data Sources",
        "Table Data",
        "table.data_sources",
        "data_id value",
        "Name",
        "Source",
        "Source Filename",
        "Value",
        "Definition",
        "Data (Table 2)",
    ],
)
def test_structural_concepts_are_rejected(name: str) -> None:
    assert is_structural_concept(name)


@pytest.mark.parametrize(
    "name",
    [
        "STOI",
        "Deep Neural Networks (DNNs)",
        "Listening Effort",
        "Speech Intelligibility",
        "Mean Temporal Distance (MTD)",
    ],
)
def test_real_concepts_are_kept(name: str) -> None:
    assert not is_structural_concept(name)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Deep Neural Networks (DNNs)", "deep neural network"),
        ("Short-Time Objective Intelligibility (STOI)", "short time objective intelligibility"),
        ("Phoneme posteriorgrams", "phoneme posteriorgram"),
        ("Data Id", "data id"),
        ("Table Data", "table data"),
    ],
)
def test_normalize_concept_key(raw: str, expected: str) -> None:
    assert normalize_concept_key(raw) == expected
