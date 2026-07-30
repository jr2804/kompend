"""Structured-data extractors for atomic document elements.

Converts parsed ``DocumentElement`` content (table grids, LaTeX equations, image
metadata) into typed ``StructuredData`` records.  These records carry the
structure that would otherwise be lost when flattening to text.

Public API:
    extract_structured(el) -> StructuredData | None
    extract_table(grid, metadata) -> StructuredTable
    extract_equation(latex, metadata) -> StructuredEquation
    extract_figure(content, metadata) -> StructuredFigure
"""

from __future__ import annotations

import re
from collections.abc import Callable

from knox.core.models import (
    DocumentElement,
    ElementType,
    EquationVariable,
    StructuredData,
    StructuredEquation,
    StructuredFigure,
    StructuredTable,
)

# Regex patterns for equation variable extraction ──────────────────────────

# Inline variable definitions:  v = value   or   v := value
# Captures: variable name, definition text
_VAR_DEFINITION = re.compile(
    r"\\(?:text|mathrm|mathbf|mathit)\{([^}]+)\}\s*[:=]+\s*(.+?)(?=\\text|\\mathrm|$)",
)

# Subscripted variables:  x_{sub} or  x_sub
_SUBSCRIPT_VAR = re.compile(r"([A-Za-z])_\{?([A-Za-z0-9]+)\}?")

# "where x is ..." style definitions after the equation
_WHERE_CLAUSE = re.compile(
    r"(?:where|with|here)\s+(.+)",
    re.IGNORECASE,
)

# Comma-separated definitions in where-clauses: "x is the first variable, y is the second"
_WHERE_ITEM = re.compile(
    r"([A-Za-z][A-Za-z0-9_]*)\s+(?:is|represents|denotes|stands for)\s+([^,;]+)",
    re.IGNORECASE,
)


def extract_structured(el: DocumentElement) -> StructuredData | None:
    """Dispatch to the appropriate extractor based on element type.

    Returns ``None`` for non-atomic or unsupported element types.
    """
    extractor = _EXTRACTORS.get(el.type)
    if extractor is None:
        return None
    return extractor(el)


def extract_table(el: DocumentElement) -> StructuredTable:
    """Convert a table grid (``list[list[str]]``) to :class:`StructuredTable`.

    The first row is treated as the header.  Each subsequent row becomes a
    dictionary keyed by column name.
    """
    grid: list[list[str]] = []
    if isinstance(el.content, list):
        grid = el.content
    if not grid:
        return StructuredTable(columns=[], rows=[])
    columns = [str(c) for c in grid[0]]
    rows = [{col: str(row[j]) if j < len(row) else "" for j, col in enumerate(columns)} for row in grid[1:]]
    caption = ""
    if el.metadata:
        caption = str(el.metadata.get("caption", ""))
    return StructuredTable(columns=columns, rows=rows, caption=caption)


def extract_equation(el: DocumentElement) -> StructuredEquation:
    r"""Convert a LaTeX formula to :class:`StructuredEquation`.

    Attempts to extract variable definitions from common LaTeX patterns:
    - ``\text{VARIABLE} = definition``
    - Subscripted symbols ``x_{sub}``
    - ``where x is ...`` clauses from surrounding metadata
    """
    latex = el.content if isinstance(el.content, str) else ""
    caption = ""
    variables: list[EquationVariable] = []
    if el.metadata:
        caption = str(el.metadata.get("caption", ""))

    variables.extend(_extract_definition_vars(latex))
    if not variables:
        variables.extend(_extract_subscript_vars(latex))

    surrounding = ""
    if el.metadata:
        surrounding = str(el.metadata.get("context", "")) + " " + caption
    if surrounding:
        variables.extend(_extract_where_clause_vars(surrounding))

    return StructuredEquation(latex=latex, variables=variables, caption=caption)


def extract_figure(el: DocumentElement) -> StructuredFigure:
    """Convert image content + metadata to :class:`StructuredFigure`."""
    caption = ""
    image_ref = ""
    alt_text = ""
    description = ""
    if el.metadata:
        caption = str(el.metadata.get("caption", ""))
        alt_text = str(el.metadata.get("alt_text", ""))
        description = str(el.metadata.get("description", ""))
    if isinstance(el.content, str):
        image_ref = el.content
    elif isinstance(el.content, bytes):
        image_ref = f"[binary: {len(el.content)} bytes]"
    return StructuredFigure(
        caption=caption,
        alt_text=alt_text,
        image_ref=image_ref,
        description=description,
    )


_EXTRACTORS: dict[ElementType, Callable[[DocumentElement], StructuredData]] = {
    ElementType.TABLE: extract_table,
    ElementType.FORMULA: extract_equation,
    ElementType.IMAGE: extract_figure,
}


# ── internal helpers ─────────────────────────────────────────────────────


def _extract_definition_vars(latex: str) -> list[EquationVariable]:
    r"""Extract ``\text{name} = definition`` patterns from LaTeX."""
    result: list[EquationVariable] = []
    for match in _VAR_DEFINITION.finditer(latex):
        name = match.group(1).strip()
        definition = match.group(2).strip().rstrip(".,;")
        if name and definition:
            units = ""
            unit_match = re.search(r"\(([^)]+)\)\s*$", definition)
            if unit_match:
                units = unit_match.group(1)
                definition = definition[: unit_match.start()].strip()
            result.append(EquationVariable(name=name, definition=definition, units=units))
    return result


def _extract_subscript_vars(latex: str) -> list[EquationVariable]:
    """Extract subscripted symbols (e.g. ``x_{ij}`` → name ``x_ij``).

    Only used as a fallback when no explicit definitions are found.
    Returns unique names without definitions.
    """
    seen: set[str] = set()
    result: list[EquationVariable] = []
    for match in _SUBSCRIPT_VAR.finditer(latex):
        name = f"{match.group(1)}_{match.group(2)}"
        if name not in seen:
            seen.add(name)
            result.append(EquationVariable(name=name))
    return result


def _extract_where_clause_vars(text: str) -> list[EquationVariable]:
    """Extract ``where x is ...`` style definitions from surrounding text."""
    result: list[EquationVariable] = []
    where_match = _WHERE_CLAUSE.search(text)
    if not where_match:
        return result
    clause = where_match.group(1)
    for match in _WHERE_ITEM.finditer(clause):
        name = match.group(1).strip()
        definition = match.group(2).strip().rstrip(".,;")
        units = ""
        unit_match = re.search(r"\(([^)]+)\)\s*$", definition)
        if unit_match:
            units = unit_match.group(1)
            definition = definition[: unit_match.start()].strip()
        if name and definition:
            result.append(EquationVariable(name=name, definition=definition, units=units))
    return result
