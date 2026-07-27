"""Tests for the shared structural-concept filter (single source of truth).

Both the ``wicer`` (GraphCompiler) and TwoPhase backends must reject the
same table/spreadsheet structural artifacts, so the logic lives in
``knox.compilation.concept_keys`` and is exercised from here.
"""

from __future__ import annotations

import pytest

from knox.compilation.concept_keys import (
    is_structural_concept,
    normalize_concept_key,
)


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
