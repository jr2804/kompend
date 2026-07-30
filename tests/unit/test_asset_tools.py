"""Tests for ``build_asset_tools()`` — on-demand asset lookup tools.

Covers:
- ``get_table`` — returns pipe-delimited table text from AssetStore
- ``get_figure_description`` — returns caption/alt-text from store
- Edge cases: unknown asset id, wrong domain, missing metadata
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knox.compilation.asset_tools import build_asset_tools
from knox.core.models.assets import AssetRecord
from knox.core.models.enums import AssetKind
from knox.core.models.structured import StructuredFigure, StructuredTable
from knox.core.pipeline.assets import AssetStore

# ---------------------------------------------------------------------------
# build_asset_tools
# ---------------------------------------------------------------------------


class TestBuildAssetTools:
    """Tests for the ``build_asset_tools()`` factory."""

    def test_returns_two_tools(self, store: AssetStore) -> None:
        """Factory returns exactly 2 tools."""
        tools = build_asset_tools(store)
        assert len(tools) == 2

    def test_tools_have_expected_names(self, store: AssetStore) -> None:
        """Tool names match the wrapped function names."""
        tools = build_asset_tools(store)
        names = {t.name for t in tools}
        assert "get_table" in names
        assert "get_figure_description" in names

    def test_tools_descriptions_are_set(self, store: AssetStore) -> None:
        """Each tool has a non-empty description from the docstring."""
        tools = build_asset_tools(store)
        for t in tools:
            assert t.description, f"Tool '{t.name}' has no description"


# ---------------------------------------------------------------------------
# get_table tool
# ---------------------------------------------------------------------------


class TestGetTable:
    """Tests for the ``get_table`` tool."""

    def test_returns_table_text(self, store: AssetStore) -> None:
        """get_table returns pipe-delimited text for a known table."""
        tools = build_asset_tools(store)
        get_table = next(t for t in tools if t.name == "get_table")
        # Call the underlying function
        raw_func = get_table.function
        text = raw_func("tab-a1b2c3")
        assert "Name" in text
        assert "Value" in text
        assert "x | 1" in text or "x" in text
        assert "Test Table" in text

    def test_unknown_asset(self, store: AssetStore) -> None:
        """get_table returns error for unknown asset id."""
        tools = build_asset_tools(store)
        get_table = next(t for t in tools if t.name == "get_table")
        raw_func = get_table.function
        text = raw_func("tab-unknown")
        assert "unknown asset" in text.lower()

    def test_wrong_domain(self, store: AssetStore) -> None:
        """get_table returns error when asset is not a table."""
        tools = build_asset_tools(store)
        get_table = next(t for t in tools if t.name == "get_table")
        raw_func = get_table.function
        text = raw_func("fig-d4e5f6")
        assert "not a table" in text.lower()

    def test_empty_store(self, tmp_path: Path) -> None:
        """get_table returns error for empty store."""
        store = AssetStore(tmp_path / "empty.json")
        store.save()
        tools = build_asset_tools(store)
        get_table = next(t for t in tools if t.name == "get_table")
        raw_func = get_table.function
        text = raw_func("tab-none")
        assert "unknown asset" in text.lower()


# ---------------------------------------------------------------------------
# get_figure_description tool
# ---------------------------------------------------------------------------


class TestGetFigure:
    """Tests for the ``get_figure_description`` tool."""

    def test_returns_caption_and_alt_text(self, store: AssetStore) -> None:
        """get_figure_description returns caption and alt-text."""
        tools = build_asset_tools(store)
        get_fig = next(t for t in tools if t.name == "get_figure_description")
        raw_func = get_fig.function
        text = raw_func("fig-d4e5f6")
        assert "Caption: A bar chart" in text
        assert "Alt text: Bar chart" in text

    def test_unknown_asset(self, store: AssetStore) -> None:
        """get_figure_description returns error for unknown id."""
        tools = build_asset_tools(store)
        get_fig = next(t for t in tools if t.name == "get_figure_description")
        raw_func = get_fig.function
        text = raw_func("fig-unknown")
        assert "unknown asset" in text.lower()

    def test_wrong_domain(self, store: AssetStore) -> None:
        """get_figure_description returns error when asset is not a figure."""
        tools = build_asset_tools(store)
        get_fig = next(t for t in tools if t.name == "get_figure_description")
        raw_func = get_fig.function
        text = raw_func("tab-a1b2c3")
        assert "not a figure" in text.lower()

    def test_no_caption_or_alt_text(self, store: AssetStore) -> None:
        """get_figure_description handles figure without metadata gracefully."""
        tools = build_asset_tools(store)
        get_fig = next(t for t in tools if t.name == "get_figure_description")
        raw_func = get_fig.function
        text = raw_func("fig-abc123")
        assert "no caption" in text.lower() or "figure" in text.lower()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> AssetStore:
    """AssetStore pre-populated with one table and one figure."""
    s = AssetStore(tmp_path / "assets.json")
    table = AssetRecord(
        id="tab-a1b2c3",
        kind=AssetKind.TABLE,
        source="test",
        data=StructuredTable(
            columns=["Name", "Value"],
            rows=[{"Name": "x", "Value": "1"}, {"Name": "y", "Value": "2"}],
            caption="Test Table",
        ),
        summary="2 rows x 2 cols",
    )
    figure = AssetRecord(
        id="fig-d4e5f6",
        kind=AssetKind.FIGURE,
        source="test",
        data=StructuredFigure(
            caption="A bar chart showing results",
            alt_text="Bar chart with positive trend",
            description="",
        ),
        summary="Figure with bar chart",
    )
    figure_no_caption = AssetRecord(
        id="fig-abc123",
        kind=AssetKind.FIGURE,
        source="test",
        data=StructuredFigure(caption="", alt_text="", description=""),
        summary="Figure without metadata",
    )
    s.put(table)
    s.put(figure)
    s.put(figure_no_caption)
    s.save()
    return s
