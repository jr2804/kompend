"""The rumdl markdown fragment adapter: element_to_markdown / markdown_to_element.

The pair wraps an element's content as the minimal markdown block of the right
kind and strips it again, so an external formatter (rumdl) can run over
element contents without touching pages, reading order or metadata.

The backend-neutral Markdown→element converter that used to live beside this
pair (``markdown_to_elements``) went with the kompend-side markdown-producing
backends — parsecraft maps its IR to elements directly.
"""

from __future__ import annotations

from kompend.core.models import DocumentElement, ElementType
from kompend.preprocessing.parsers.markdown import (
    FENCE_RE,
    HEADING_RE,
    LIST_ITEM_RE,
    element_to_markdown,
    markdown_to_element,
)


def test_heading_regex_matches_atx_h1_through_h6() -> None:
    cases = {
        "# h1": (1, "h1"),
        "### h3 trailing": (3, "h3 trailing"),
        "###### h6": (6, "h6"),
        "####### not a heading": None,
    }
    for line, expected in cases.items():
        match = HEADING_RE.match(line)
        if expected is None:
            assert match is None
        else:
            assert match is not None
            assert len(match.group(1)) == expected[0]
            assert match.group(2).strip() == expected[1]


def test_fence_regex_accepts_three_or_more_chars_and_language() -> None:
    assert (m := FENCE_RE.match("```python")) is not None
    assert m.group(1) == "```"
    assert m.group(2) == "python"
    assert (m := FENCE_RE.match("~~~")) is not None
    assert m.group(1) == "~~~"
    assert FENCE_RE.match("`two") is None


def test_list_item_regex_matches_bullet_and_ordered() -> None:
    for marker in ("-", "*", "+"):
        assert LIST_ITEM_RE.match(f"{marker} item 1") is not None
    assert LIST_ITEM_RE.match("1. numbered") is not None
    assert LIST_ITEM_RE.match("2) alt numbered") is not None
    assert LIST_ITEM_RE.match("not a list") is None


# ── Round-trip with element_to_markdown / markdown_to_element ───────────


def test_round_trip_text() -> None:
    e = DocumentElement(type=ElementType.TEXT, content="hello   world")
    md = element_to_markdown(e)
    assert md == "hello   world"
    assert markdown_to_element(ElementType.TEXT, "hello   world") == "hello   world"


def test_round_trip_code_preserves_language() -> None:
    e = DocumentElement(type=ElementType.CODE, content="x = 1\ny = 2", metadata={"language": "python"})
    md = element_to_markdown(e)
    assert md == "```python\nx = 1\ny = 2\n```"
    assert markdown_to_element(ElementType.CODE, md) == "x = 1\ny = 2"


def test_round_trip_formula_strips_wrapper() -> None:
    e = DocumentElement(type=ElementType.FORMULA, content="E = m c^2")
    md = element_to_markdown(e)
    assert md is not None
    assert markdown_to_element(ElementType.FORMULA, md) == "E = m c^2"


def test_round_trip_keeps_table_and_image_untouched() -> None:
    image = DocumentElement(type=ElementType.IMAGE, content="data:...")
    table = DocumentElement(type=ElementType.TABLE, content=[["A", "B"]])
    assert element_to_markdown(image) is None
    assert element_to_markdown(table) is None
