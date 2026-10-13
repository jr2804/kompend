"""Structured data types preserved as typed records (tables, equations, figures).

These are leaf content in the chunk tree — preserved with their structure
intact rather than flattened to pipe-delimited text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

from wenmode import MarkdownRenderer
from wenmode.nodes import Paragraph, Root, Strong, Text
from wenmode.nodes import Table as WTable
from wenmode.nodes import TableCell as WCell
from wenmode.nodes import TableRow as WRow

from kompend.core.models.enums import StructuredField

# Reusable renderer for table AST -> Markdown conversion.
# Module-level to avoid allocating one per call.
_MARKDOWN_RENDERER = MarkdownRenderer()


@dataclass
class EquationVariable:
    """A single variable in an equation with its definition."""

    name: str
    definition: str = ""
    units: str = ""


@dataclass
class StructuredTable:
    """Table preserved as typed records instead of pipe-delimited text."""

    columns: list[str]
    rows: list[dict[str, str]]
    caption: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            StructuredField._TYPE: StructuredField.TYPE_TABLE.value,
            StructuredField.COLUMNS: self.columns,
            StructuredField.ROWS: self.rows,
            StructuredField.CAPTION: self.caption,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StructuredTable:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            columns=d.get(StructuredField.COLUMNS, []),
            rows=d.get(StructuredField.ROWS, []),
            caption=d.get(StructuredField.CAPTION, ""),
        )


@dataclass
class StructuredEquation:
    """Equation preserved as LaTeX + parsed variables."""

    latex: str
    variables: list[EquationVariable] = field(default_factory=list)
    caption: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            StructuredField._TYPE: StructuredField.TYPE_EQUATION.value,
            StructuredField.LATEX: self.latex,
            StructuredField.VARIABLES: [
                {
                    StructuredField.NAME: variable.name,
                    StructuredField.DEFINITION: variable.definition,
                    StructuredField.UNITS: variable.units,
                }
                for variable in self.variables
            ],
            StructuredField.CAPTION: self.caption,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StructuredEquation:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        variables = [
            EquationVariable(
                name=v.get(StructuredField.NAME, ""),
                definition=v.get(StructuredField.DEFINITION, ""),
                units=v.get(StructuredField.UNITS, ""),
            )
            for v in d.get(StructuredField.VARIABLES, [])
        ]
        return cls(
            latex=d.get(StructuredField.LATEX, ""),
            variables=variables,
            caption=d.get(StructuredField.CAPTION, ""),
        )


@dataclass
class StructuredFigure:
    """Figure with caption, alt text, and optional VLM description."""

    caption: str
    alt_text: str = ""
    image_ref: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            StructuredField._TYPE: StructuredField.TYPE_FIGURE.value,
            StructuredField.CAPTION: self.caption,
            StructuredField.ALT_TEXT: self.alt_text,
            StructuredField.IMAGE_REF: self.image_ref,
            StructuredField.DESCRIPTION: self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StructuredFigure:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            caption=d.get(StructuredField.CAPTION, ""),
            alt_text=d.get(StructuredField.ALT_TEXT, ""),
            image_ref=d.get(StructuredField.IMAGE_REF, ""),
            description=d.get(StructuredField.DESCRIPTION, ""),
        )


StructuredData: TypeAlias = StructuredTable | StructuredEquation | StructuredFigure  # noqa: UP040


def structured_to_dict(s: StructuredData) -> dict[str, Any]:
    """Serialize a StructuredData variant with the canonical discriminator."""
    if isinstance(s, StructuredTable):
        return s.to_dict()
    if isinstance(s, StructuredEquation):
        return s.to_dict()
    if isinstance(s, StructuredFigure):
        return s.to_dict()
    raise TypeError(f"Unknown StructuredData type: {type(s).__name__}")


_structured_to_dict = structured_to_dict  # backward-compat alias


__all__ = [
    "EquationVariable",
    "StructuredData",
    "StructuredEquation",
    "StructuredFigure",
    "StructuredTable",
    "structured_to_text",
]


def structured_to_text(s: StructuredData) -> str:
    """Flatten a :class:`StructuredData` variant to a text representation.

    Lives here (not in ``kompend.chunking.summarizer``) because both the chunking
    and compilation tiers consume it — importing it from the chunker pulls in
    the agent stack and creates a circular import.

    Tables are rendered through wenmode's AST pipeline for consistent
    GFM-format pipe tables. Equations and figures remain text-based.
    """
    if isinstance(s, StructuredTable):
        root = Root()
        if s.caption:
            caption_para = Paragraph()
            caption_para.children.append(Strong(children=[Text(value=s.caption)]))
            root.children.append(caption_para)

        # Build header row with dashes
        tbl = WTable(align=[None] * len(s.columns))
        hdr = WRow()
        for col in s.columns:
            cell = WCell()
            cell.children.append(Text(value=col))
            hdr.children.append(cell)
        tbl.children.append(hdr)

        # Data rows
        for row_dict in s.rows:
            row = WRow()
            for col in s.columns:
                cell = WCell()
                cell.children.append(Text(value=row_dict.get(col, "")))
                row.children.append(cell)
            tbl.children.append(row)

        root.children.append(tbl)
        return _MARKDOWN_RENDERER.render(root).rstrip()
    if isinstance(s, StructuredEquation):
        parts = [s.latex]
        if s.variables:
            parts.append("where " + ", ".join(f"{v.name}={v.definition}" for v in s.variables if v.definition))
        if s.caption:
            parts.append(f"({s.caption})")
        return " ".join(parts)
    if isinstance(s, StructuredFigure):
        parts = []
        if s.caption:
            parts.append(s.caption)
        if s.alt_text:
            parts.append(f"[{s.alt_text}]")
        if s.image_ref:
            parts.append(f"({s.image_ref})")
        return " ".join(parts) if parts else "[figure]"
    return ""


def structured_from_dict(d: dict[str, Any]) -> StructuredData:
    """Deserialize a StructuredData variant using the canonical discriminator."""
    kind = d.get(StructuredField._TYPE)
    if kind == StructuredField.TYPE_TABLE.value:
        return StructuredTable.from_dict(d)
    if kind == StructuredField.TYPE_EQUATION.value:
        return StructuredEquation.from_dict(d)
    if kind == StructuredField.TYPE_FIGURE.value:
        return StructuredFigure.from_dict(d)
    raise ValueError(f"Unknown StructuredData _type: {kind!r}")
