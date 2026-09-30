"""The shared Markdown → DocumentElement converter and rumdl text adapter.

The converter is single-source-of-truth — every backend that emits Markdown
after its SDK call goes through :func:`markdown_to_elements` so equations,
fenced code, and list items are recognised (previously, only heading/image/
table/text were; the rest was swallowed into TEXT).

Backends ship only what is genuinely backend-specific: their SDK call and an
:image resolver that turns a ref into bytes (see :class:`ImageResolver`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from knox.core.models import DocumentElement, ElementType
from knox.preprocessing.parsers.markdown import (
    FENCE_RE,
    HEADING_RE,
    IMAGE_REF_RE,
    INLINE_MATH_RE,
    LIST_ITEM_RE,
    TABLE_SEPARATOR_RE,
    ImagePayload,
    MarkdownContext,
    build_image_element,
    element_to_markdown,
    markdown_to_element,
    markdown_to_elements,
    parse_table,
)


@dataclass(frozen=True, slots=True)
class _NoopResolver:
    """Resolver that returns no bytes — used when an image ref is non-fatal."""

    def __call__(self, alt_text: str, ref_url: str) -> ImagePayload | None:  # noqa: D401 — protocol shim
        return None


def _has_lines(value: str) -> bool:
    return "\n" in value


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


def test_image_ref_regex_handles_no_alt() -> None:
    assert (m := IMAGE_REF_RE.match("![](ref.png)")) is not None
    assert m.group(1) == ""
    assert m.group(2) == "ref.png"


def test_table_separator_regex_accepts_alignment_rows() -> None:
    assert TABLE_SEPARATOR_RE.match("| --- | :---: | :--- |")
    assert TABLE_SEPARATOR_RE.match("|---|---|")
    assert TABLE_SEPARATOR_RE.match("| --- |")
    assert not TABLE_SEPARATOR_RE.match("not a separator")


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


def test_inline_math_regex_matches_inline_double_dollar_and_bracket() -> None:
    assert (m := INLINE_MATH_RE.match("$$a+b$$")) is not None
    assert m.group(1) == "a+b"
    assert (m := INLINE_MATH_RE.match(r"\[e=mc^2\]")) is not None
    assert m.group(2) == "e=mc^2"
    assert INLINE_MATH_RE.match("$single$") is None


def test_parse_table_returns_grid_dropping_separator() -> None:
    lines = [
        "| Name | Score |",
        "|------|-------|",
        "| alice | 9 |",
        "| bob   | 7 |",
    ]
    grid, consumed = parse_table(lines, 0)
    assert consumed == 4
    assert grid == [["Name", "Score"], ["alice", "9"], ["bob", "7"]]


def test_parse_table_stops_at_non_pipe_line() -> None:
    lines = [
        "| A | B |",
        "|---|---|",
        "| 1 | 2 |",
        "",
        "next paragraph",
    ]
    _grid, consumed = parse_table(lines, 0)
    assert consumed == 3  # 3 pipe rows; the blank line is NOT consumed


def test_build_image_element_writes_data_uri_in_base64_mode(tmp_path: Path) -> None:
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    ctx = MarkdownContext(
        source_file=tmp_path / "doc.pdf",
        image_mode="base64",
        images_dir=images_dir,
    )
    payload = ImagePayload(b"\x89PNG\r\n\x1a\n", "png", "fig1")
    element = build_image_element(
        payload,
        alt_text="A figure caption",
        context=ctx,
        reading_order=5,
    )
    assert element.type is ElementType.IMAGE
    assert element.reading_order == 5
    assert str(element.content).startswith("data:image/png;base64,")
    assert element.metadata["alt_text"] == "A figure caption"
    assert list(images_dir.iterdir()) == []  # file mode is NOT used


def test_build_image_element_writes_file_and_alt_in_file_mode(tmp_path: Path) -> None:
    images_dir = tmp_path / "imgs"
    ctx = MarkdownContext(
        source_file=tmp_path / "doc.pdf",
        image_mode="file",
        images_dir=images_dir,
    )
    payload = ImagePayload(b"\x89PNG\r\n\x1a\n", "png", "fig1")
    element = build_image_element(payload, alt_text="caption", context=ctx, reading_order=0)
    assert element.type is ElementType.IMAGE
    assert isinstance(element.content, str)
    assert (images_dir / "doc_img_fig1.png").exists()  # stem drops the suffix


def test_markdown_to_elements_emits_all_kinds(tmp_path: Path) -> None:
    md = (
        "# Title\n"
        "\n"
        "A paragraph line one\n"
        "second line.\n"
        "\n"
        "## Subheading\n"
        "\n"
        "- first\n"
        "- second\n"
        "\n"
        "```python\n"
        "x = 1\n"
        "```\n"
        "\n"
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "\n"
        "$$\n"
        "E = m c^2\n"
        "$$\n"
        "\n"
        "Inline $$\\\\alpha$$ is here.\n"
    )
    elements, _order, first_h1 = markdown_to_elements(
        md,
        MarkdownContext(source_file=tmp_path / "d.pdf", image_mode="base64", images_dir=tmp_path / "i"),
    )
    # types we'd expect in document order
    seen = [e.type for e in elements]
    assert seen == [
        ElementType.HEADING,
        ElementType.TEXT,
        ElementType.HEADING,
        ElementType.LIST_ITEM,
        ElementType.LIST_ITEM,
        ElementType.CODE,
        ElementType.TABLE,
        ElementType.FORMULA,
        ElementType.TEXT,
    ]
    assert first_h1 == "Title"
    code = next(e for e in elements if e.type is ElementType.CODE)
    assert code.metadata["language"] == "python"
    formula = next(e for e in elements if e.type is ElementType.FORMULA)
    assert "E = m c^2" in str(formula.content)


def test_markdown_to_elements_skips_unresolved_image(tmp_path: Path) -> None:
    md = "![missing](nope.png)\n"
    elements, _order, _ = markdown_to_elements(
        md,
        MarkdownContext(source_file=tmp_path / "d.pdf", image_mode="base64", images_dir=tmp_path / "i", resolve_image=_NoopResolver()),
    )
    assert elements == []  # ref dropped, no empty IMAGE element


def test_unterminated_math_falls_back_to_text(tmp_path: Path) -> None:
    md = "Open $$$$\nstill math\n"
    elements, _, _ = markdown_to_elements(md, MarkdownContext(source_file=tmp_path / "x", image_mode="base64", images_dir=tmp_path / "i"))
    assert all(e.type is not ElementType.FORMULA for e in elements)


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
