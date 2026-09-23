r"""Backend-neutral Markdown → :class:`DocumentElement` conversion.

Every markdown-producing backend (liteparse, marker, mineru, the glm-ocr
fallback) needs the same thing after its SDK call: split page Markdown into
KNOX elements. That walk used to be copy-pasted per backend and recognised only
headings, images, tables and paragraphs — so **equations, fenced code and list
items were silently swallowed into TEXT**, even from backends that emit them
(marker and mineru routinely produce ``$$…$$`` math and ``` fences). The wiki
then compiled that content as prose.

This module is the single source of truth for that conversion. Backends keep
only what is genuinely backend-specific: the SDK call and an
:data:`ImageResolver` that turns a Markdown image ref into bytes.

Element kinds recognised, in match order:

============  ==================================================
Markdown      Element
============  ==================================================
````` ``` ````  :attr:`ElementType.CODE` (``metadata["language"]``)
``# …``       :attr:`ElementType.HEADING` (``heading_level``)
``![alt](u)`` :attr:`ElementType.IMAGE` (via the resolver)
``| a | b |`` :attr:`ElementType.TABLE` (cell grid)
``$$ … $$``   :attr:`ElementType.FORMULA` (also ``\\[ … \\]``)
``- …``/``1.``  :attr:`ElementType.LIST_ITEM` (marker stripped)
other         :attr:`ElementType.TEXT` (consecutive lines joined)
============  ==================================================
"""

from __future__ import annotations

import base64
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from knox.core.models import DocumentElement, ElementType, MetadataField

__all__ = [
    "HEADING_RE",
    "IMAGE_REF_RE",
    "MIME_BY_FMT",
    "TABLE_SEPARATOR_RE",
    "ImagePayload",
    "ImageResolver",
    "MarkdownContext",
    "build_image_element",
    "element_to_markdown",
    "markdown_to_element",
    "markdown_to_elements",
    "parse_table",
]

# ── Shared patterns ────────────────────────────────────────────────────

HEADING_RE: Final = re.compile(r"^(#{1,6})\s+(.*)$")
IMAGE_REF_RE: Final = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
TABLE_SEPARATOR_RE: Final = re.compile(r"^\|[\s:|-]+\|?\s*$")
#: ``` or ~~~ fence, with an optional info string (language).
FENCE_RE: Final = re.compile(r"^(`{3,}|~{3,})\s*([\w+#.-]*)\s*$")
#: ``- item``, ``* item``, ``+ item``, ``1. item``, ``1) item``.
LIST_ITEM_RE: Final = re.compile(r"^(?:[-*+]|\d+[.)])\s+(.*)$")
#: Single-line math: ``$$x$$`` or ``\[x\]``.
INLINE_MATH_RE: Final = re.compile(r"^(?:\$\$(.*?)\$\$|\\\[(.*?)\\\])$")
#: Block-math openers and their closers, in ``$$``/``\[`` pairs.
_MATH_DELIMITERS: Final[dict[str, str]] = {"$$": "$$", "\\[": "\\]"}

MIME_BY_FMT: Final[dict[str, str]] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}


# ── Inverse direction (for whole-part normalisation) ───────────────────
#
# ``element_to_markdown`` / ``markdown_to_element`` are the pair used when an
# external markdown formatter must run over individual element contents: each
# content is wrapped in the minimal markdown that makes it the right *kind* of
# block, formatted in one batch, then unwrapped. Only the content changes —
# page, reading order and metadata stay on the element.

_FRAGMENT_TYPE_SKIP: Final = frozenset({ElementType.TABLE, ElementType.IMAGE, ElementType.METADATA})


#: ``(alt_text, ref_url) -> payload | None``. ``None`` means the ref has no
#: backing bytes in this backend's result and the ref is skipped.
ImageResolver = Callable[[str, str], "ImagePayload | None"]


# ── Image plumbing ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ImagePayload:
    """Bytes plus naming metadata for one resolved figure.

    Backends differ only in *how* they look up bytes (by id, by filename, as a
    PIL image). Once looked up, everything downstream is identical, so the
    lookup result is normalized to this shape and the element construction
    lives in :func:`build_image_element`.
    """

    data: bytes
    format: str
    #: Bare stem for the written filename — no extension. Backends pass their
    #: image id / name without suffix.
    name: str


@dataclass(frozen=True, slots=True)
class MarkdownContext:
    """Everything the converter needs that is not the Markdown itself."""

    source_file: Path
    image_mode: str
    images_dir: Path
    page: int | None = None
    reading_order: int = 0
    #: Optional resolver. ``None`` means "no figures" — resolved lazily inside
    #: :func:`markdown_to_elements` so the default never names a module-level
    #: function (csort's stepdown ordering would move it after the dataclass).
    resolve_image: ImageResolver | None = None


# ── The converter ──────────────────────────────────────────────────────


def markdown_to_elements(  # noqa: PLR0912 — one branch per markdown block kind
    markdown: str,
    context: MarkdownContext,
) -> tuple[list[DocumentElement], int, str]:
    """Convert Markdown into KNOX elements.

    Returns ``(elements, next_reading_order, first_h1_title)`` — the shape every
    backend's ``parse()`` already consumes, so migrating a backend is a swap of
    the call, not a restructure.
    """
    lines = markdown.splitlines()
    elements: list[DocumentElement] = []
    first_h1 = ""
    reading_order = context.reading_order
    i = 0
    n = len(lines)

    def add(element: DocumentElement) -> None:
        elements.append(element)

    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue

        fence = FENCE_RE.match(stripped)
        if fence:
            code, language, i = _consume_fence(lines, i, fence)
            meta = {"language": language} if language else {}
            add(
                DocumentElement(
                    type=ElementType.CODE,
                    content=code,
                    page=context.page,
                    reading_order=reading_order,
                    metadata=meta,
                    source_file=context.source_file,
                )
            )
            reading_order += 1
            continue

        heading = HEADING_RE.match(stripped)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2).strip()
            if level == 1 and not first_h1:
                first_h1 = text
            add(
                DocumentElement(
                    type=ElementType.HEADING,
                    content=text,
                    page=context.page,
                    reading_order=reading_order,
                    heading_level=level,
                    source_file=context.source_file,
                )
            )
            reading_order += 1
            i += 1
            continue

        image = IMAGE_REF_RE.match(stripped)
        if image:
            resolve = context.resolve_image or _no_image
            payload = resolve(image.group(1), image.group(2))
            if payload is not None:
                add(
                    build_image_element(
                        payload,
                        alt_text=image.group(1),
                        context=context,
                        reading_order=reading_order,
                    )
                )
                reading_order += 1
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < n and TABLE_SEPARATOR_RE.match(lines[i + 1].strip()):
            grid, consumed = parse_table(lines, i)
            if grid:
                add(
                    DocumentElement(
                        type=ElementType.TABLE,
                        content=grid,
                        page=context.page,
                        reading_order=reading_order,
                        source_file=context.source_file,
                    )
                )
                reading_order += 1
            i += consumed
            continue

        single_line_math = INLINE_MATH_RE.match(stripped)
        if single_line_math:
            add(
                DocumentElement(
                    type=ElementType.FORMULA,
                    content=(single_line_math.group(1) or single_line_math.group(2) or "").strip(),
                    page=context.page,
                    reading_order=reading_order,
                    source_file=context.source_file,
                )
            )
            reading_order += 1
            i += 1
            continue

        math_block = _consume_math(lines, i)
        if math_block is not None:
            latex, i = math_block
            add(
                DocumentElement(
                    type=ElementType.FORMULA,
                    content=latex,
                    page=context.page,
                    reading_order=reading_order,
                    source_file=context.source_file,
                )
            )
            reading_order += 1
            continue

        item = LIST_ITEM_RE.match(stripped)
        if item:
            add(
                DocumentElement(
                    type=ElementType.LIST_ITEM,
                    content=item.group(1).strip(),
                    page=context.page,
                    reading_order=reading_order,
                    source_file=context.source_file,
                )
            )
            reading_order += 1
            i += 1
            continue

        # Default: a paragraph of consecutive plain-text lines.
        paragraph: list[str] = [stripped]
        j = i + 1
        while j < n and _continues_paragraph(lines[j]):
            paragraph.append(lines[j].strip())
            j += 1
        add(
            DocumentElement(
                type=ElementType.TEXT,
                content="\n".join(paragraph),
                page=context.page,
                reading_order=reading_order,
                source_file=context.source_file,
            )
        )
        reading_order += 1
        i = j

    return elements, reading_order, first_h1


def _no_image(_alt_text: str, _ref_url: str) -> None:
    """Default resolver: no images available (markdown-only backends)."""
    return None


def build_image_element(
    payload: ImagePayload,
    *,
    alt_text: str,
    context: MarkdownContext,
    reading_order: int,
) -> DocumentElement:
    """Write/encode figure bytes and wrap them in an IMAGE element.

    ``image_mode == "base64"`` inlines a data URI; anything else writes the
    bytes under ``context.images_dir`` and stores the path. ``metadata`` uses
    the :class:`MetadataField` symbols so the downstream extractor /
    VLM-enricher contract holds.
    """
    fmt = (payload.format or "png").lower()
    mime = MIME_BY_FMT.get(fmt, "image/png")

    if context.image_mode == "base64":
        data_uri = f"data:{mime};base64,{base64.b64encode(payload.data).decode('ascii')}"
        content: str = data_uri
        meta: dict[str, object] = {MetadataField.IMAGE_BASE64: data_uri, "image_mode": "base64"}
    else:
        context.images_dir.mkdir(parents=True, exist_ok=True)
        out_file = context.images_dir / f"{context.source_file.stem}_img_{payload.name}.{fmt}"
        out_file.write_bytes(payload.data)
        content = str(out_file)
        meta = {MetadataField.IMAGE_FILE: str(out_file), "image_mode": "file"}

    if alt_text:
        meta["alt_text"] = alt_text

    return DocumentElement(
        type=ElementType.IMAGE,
        content=content,
        page=context.page,
        reading_order=reading_order,
        metadata=meta,
        source_file=context.source_file,
    )


# ── Block helpers ──────────────────────────────────────────────────────


def parse_table(lines: Sequence[str], start: int) -> tuple[list[list[str]], int]:
    """Parse a GFM table starting at *start*.

    Returns ``(grid, consumed)`` where *grid* is a list of rows (header first,
    separator row dropped). The caller has already verified that
    ``lines[start + 1]`` matches :data:`TABLE_SEPARATOR_RE`.
    """
    block: list[str] = []
    j = start
    while j < len(lines) and lines[j].strip().startswith("|"):
        block.append(lines[j].strip())
        j += 1
    grid: list[list[str]] = []
    for row in block:
        if TABLE_SEPARATOR_RE.match(row):
            continue
        grid.append([cell.strip() for cell in row.strip("|").split("|")])
    return grid, j - start


def _consume_fence(
    lines: Sequence[str],
    start: int,
    fence: re.Match[str],
) -> tuple[str, str, int]:
    """Collect a fenced block. Returns ``(code, language, next_index)``."""
    marker = fence.group(1)
    language = fence.group(2)
    body: list[str] = []
    j = start + 1
    while j < len(lines) and not lines[j].strip().startswith(marker):
        body.append(lines[j])
        j += 1
    return "\n".join(body), language, min(j + 1, len(lines))


def _consume_math(lines: Sequence[str], start: int) -> tuple[str, int] | None:
    r"""Collect a ``$$``/``\[`` delimited block starting at *start*.

    Returns ``(latex, next_index)``, or ``None`` when *start* is not a math
    opener or the block is unterminated (then it stays plain text). Single-line
    ``$$x$$`` is handled by the caller via :data:`INLINE_MATH_RE`.
    """
    opener = lines[start].strip()
    closer = _MATH_DELIMITERS.get(opener)
    if closer is None:
        return None
    body: list[str] = []
    j = start + 1
    while j < len(lines) and lines[j].strip() != closer:
        body.append(lines[j])
        j += 1
    if j >= len(lines):  # unterminated → treat as plain text
        return None
    return "\n".join(body).strip(), j + 1


def _continues_paragraph(line: str) -> bool:
    """True when *line* is a continuation of the current paragraph."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped in _MATH_DELIMITERS:
        return False
    if HEADING_RE.match(stripped) or IMAGE_REF_RE.match(stripped):
        return False
    if FENCE_RE.match(stripped) or INLINE_MATH_RE.match(stripped):
        return False
    if LIST_ITEM_RE.match(stripped):
        return False
    return not stripped.startswith("|")


def element_to_markdown(element: DocumentElement) -> str | None:
    """Wrap an element's text content as a standalone markdown block.

    Returns ``None`` for non-text elements (tables are grids, images are
    references, metadata is invisible) — those are already canonical and must
    not be round-tripped through a markdown formatter.
    """
    if element.type in _FRAGMENT_TYPE_SKIP or not isinstance(element.content, str):
        return None
    content = element.content
    if not content.strip():
        return None
    if element.type is ElementType.HEADING:
        level = max(1, min(element.heading_level or 1, 6))
        return f"{'#' * level} {content}"
    if element.type is ElementType.LIST_ITEM:
        return f"- {content}"
    if element.type is ElementType.CODE:
        language = str((element.metadata or {}).get("language", "") or "")
        return f"```{language}\n{content}\n```"
    if element.type is ElementType.FORMULA:
        return f"$$\n{content}\n$$"
    return content


def markdown_to_element(element_type: ElementType, markdown: str) -> str | None:  # noqa: PLR0911 — one return per kind
    """Strip the wrapper added by :func:`element_to_markdown`.

    Returns the bare content, or ``None`` when *markdown* no longer has the
    expected shape (then the caller keeps the original content — never guess).
    """
    text = markdown.strip()
    if not text:
        return None
    if element_type is ElementType.HEADING:
        match = HEADING_RE.match(text.split("\n", 1)[0].strip())
        return match.group(2).strip() if match else None
    if element_type is ElementType.LIST_ITEM:
        match = LIST_ITEM_RE.match(text.split("\n", 1)[0].strip())
        return match.group(1).strip() if match else None
    if element_type is ElementType.CODE:
        lines = text.splitlines()
        if len(lines) < 2 or not FENCE_RE.match(lines[0].strip()):
            return None
        if not lines[-1].strip().startswith("```") and not lines[-1].strip().startswith("~~~"):
            return None
        return "\n".join(lines[1:-1])
    if element_type is ElementType.FORMULA:
        lines = text.splitlines()
        if len(lines) < 2 or lines[0].strip() not in _MATH_DELIMITERS or lines[-1].strip() != "$$":
            return None
        return "\n".join(lines[1:-1]).strip()
    return text
