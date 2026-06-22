"""Tests for structured-data extractors (extractors.py)."""

from __future__ import annotations

from knox.chunking.extractors import (
    extract_equation,
    extract_figure,
    extract_structured,
    extract_table,
)
from knox.core.models import (
    DocumentElement,
    ElementType,
    StructuredEquation,
    StructuredFigure,
    StructuredTable,
)

# ── extract_structured dispatch ──────────────────────────────────────────


class TestExtractStructuredDispatch:
    def test_table_dispatches(self) -> None:
        el = DocumentElement(
            type=ElementType.TABLE,
            content=[["A", "B"], ["1", "2"]],
        )
        result = extract_structured(el)
        assert isinstance(result, StructuredTable)

    def test_formula_dispatches(self) -> None:
        el = DocumentElement(type=ElementType.FORMULA, content=r"x = \alpha + \beta")
        result = extract_structured(el)
        assert isinstance(result, StructuredEquation)

    def test_image_dispatches(self) -> None:
        el = DocumentElement(type=ElementType.IMAGE, content="img.png", metadata={"caption": "Figure"})
        result = extract_structured(el)
        assert isinstance(result, StructuredFigure)

    def test_text_returns_none(self) -> None:
        el = DocumentElement(type=ElementType.TEXT, content="just text")
        assert extract_structured(el) is None

    def test_heading_returns_none(self) -> None:
        el = DocumentElement(type=ElementType.HEADING, content="Title", heading_level=1)
        assert extract_structured(el) is None


# ── extract_table ────────────────────────────────────────────────────────


class TestExtractTable:
    def test_basic_grid(self) -> None:
        el = DocumentElement(type=ElementType.TABLE, content=[["Name", "Value"], ["foo", "42"]])
        result = extract_table(el)
        assert result.columns == ["Name", "Value"]
        assert result.rows == [{"Name": "foo", "Value": "42"}]

    def test_caption_from_metadata(self) -> None:
        el = DocumentElement(
            type=ElementType.TABLE,
            content=[["A"], ["1"]],
            metadata={"caption": "Test Results"},
        )
        result = extract_table(el)
        assert result.caption == "Test Results"

    def test_empty_grid(self) -> None:
        el = DocumentElement(type=ElementType.TABLE, content=[])
        result = extract_table(el)
        assert result.columns == []
        assert result.rows == []

    def test_uneven_rows(self) -> None:
        el = DocumentElement(type=ElementType.TABLE, content=[["A", "B", "C"], ["1", "2"]])
        result = extract_table(el)
        assert result.rows[0] == {"A": "1", "B": "2", "C": ""}

    def test_non_list_content(self) -> None:
        el = DocumentElement(type=ElementType.TABLE, content="not a grid")
        result = extract_table(el)
        assert result.columns == []
        assert result.rows == []


# ── extract_equation ─────────────────────────────────────────────────────


class TestExtractEquation:
    def test_basic_latex(self) -> None:
        el = DocumentElement(type=ElementType.FORMULA, content=r"E = mc^2")
        result = extract_equation(el)
        assert result.latex == r"E = mc^2"

    def test_definition_variables(self) -> None:
        latex = r"\text{SNR} = 10 \log_{10}(P_s / P_n) \text{dB}"
        el = DocumentElement(type=ElementType.FORMULA, content=latex)
        result = extract_equation(el)
        assert any(v.name == "SNR" for v in result.variables)

    def test_definition_with_units(self) -> None:
        latex = r"\text{f} = 1000 \text{Hz}"
        el = DocumentElement(type=ElementType.FORMULA, content=latex)
        result = extract_equation(el)
        assert any(v.name == "f" for v in result.variables)

    def test_subscript_fallback(self) -> None:
        latex = r"x_{ij} + y_{ij} = z_{ij}"
        el = DocumentElement(type=ElementType.FORMULA, content=latex)
        result = extract_equation(el)
        names = {v.name for v in result.variables}
        assert "x_ij" in names
        assert "y_ij" in names
        assert "z_ij" in names

    def test_where_clause_from_caption(self) -> None:
        latex = r"P = 2\pi f"
        el = DocumentElement(
            type=ElementType.FORMULA,
            content=latex,
            metadata={"caption": "where f is the frequency in Hz"},
        )
        result = extract_equation(el)
        assert any(v.name == "f" and "frequency" in v.definition.lower() for v in result.variables)

    def test_empty_content(self) -> None:
        el = DocumentElement(type=ElementType.FORMULA, content=123)
        result = extract_equation(el)
        assert result.latex == ""
        assert result.variables == []

    def test_caption_from_metadata(self) -> None:
        el = DocumentElement(
            type=ElementType.FORMULA,
            content="x = y",
            metadata={"caption": "Equation 1"},
        )
        result = extract_equation(el)
        assert result.caption == "Equation 1"


# ── extract_figure ───────────────────────────────────────────────────────


class TestExtractFigure:
    def test_string_content(self) -> None:
        el = DocumentElement(type=ElementType.IMAGE, content="figures/diagram.png")
        result = extract_figure(el)
        assert result.image_ref == "figures/diagram.png"

    def test_binary_content(self) -> None:
        el = DocumentElement(type=ElementType.IMAGE, content=b"\x89PNG")
        result = extract_figure(el)
        assert "binary" in result.image_ref

    def test_metadata_extraction(self) -> None:
        el = DocumentElement(
            type=ElementType.IMAGE,
            content="img.png",
            metadata={
                "caption": "Block diagram",
                "alt_text": "System architecture",
                "description": "Shows data flow",
            },
        )
        result = extract_figure(el)
        assert result.caption == "Block diagram"
        assert result.alt_text == "System architecture"
        assert result.description == "Shows data flow"

    def test_empty_metadata(self) -> None:
        el = DocumentElement(type=ElementType.IMAGE, content="img.png")
        result = extract_figure(el)
        assert result.caption == ""
        assert result.alt_text == ""
        assert result.image_ref == "img.png"
