"""LiteParse-based parser (run-llama/liteparse) — fast Rust-core alternative.

LiteParse reconstructs documents to Markdown with embedded image bytes. It has
no LLM/captioning of its own, so ``figure_enrichment`` is ``False`` and every
extracted figure still flows through ``VLMEnricher`` (per-figure bypass only
applies when a description is already present, which LiteParse never sets).

Office formats (DOC/DOCX/PPT/PPTX/XLS/XLSX) are converted via an external
LibreOffice install; PDF is handled natively via PDFium.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from liteparse import LiteParse

from knox.core.models import DocumentElement, ElementType, MetadataField, ParsedDocument
from knox.logging import get_logger
from knox.preprocessing.parsers._registry import ParserBuildOptions, register_backend
from knox.preprocessing.parsers.base import (
    IMAGE_SCALE_DEFAULT,
    BaseParser,
    FileExtension,
)

# ``![alt](img_<id>.png)`` — liteparse emits refs in this shape (older
# versions used an ``image_`` prefix; both are stripped by ``_ref_to_id``).
_IMAGE_REF_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
# GFM table separator row, e.g. ``| --- | :---: |``.
_TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|?\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

_MIME_BY_FMT: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}


#: Filename prefixes liteparse uses for image refs. Newer releases emit
#: ``img_<id>.png``; older ones ``image_<id>.png``. Missing the prefix meant
#: ``images_by_id`` lookup never matched and every figure was dropped.
_REF_PREFIXES: tuple[str, ...] = ("img_", "image_")

logger = get_logger(__name__)


class LiteParseParser(BaseParser):
    """Parser using LiteParse for fast Markdown reconstruction.

    LiteParse's Markdown output is walked line-by-line and mapped to KNOX
    ``ElementType`` values (headings, tables, images, text). Image bytes from
    ``ParseResult.images`` are written to ``images_dir`` (file mode) or encoded
    as data URIs (base64 mode), matching the ``DoclingParser`` contract so the
    downstream chunker/resolver/VLM path is unchanged.
    """

    supported_extensions: set[FileExtension] = {
        FileExtension.PDF,
        FileExtension.DOC,
        FileExtension.DOCX,
        FileExtension.PPTX,
        FileExtension.XLSX,
    }

    # No internal captioning — VLM still captions every figure.
    figure_enrichment: bool = False

    def __init__(
        self,
        *,
        image_mode: str = "file",
        images_dir: Path | None = None,
        images_scale: float = IMAGE_SCALE_DEFAULT,
        ocr_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the parser.

        Args:
            image_mode: ``"file"`` (write images to ``images_dir``) or
                ``"base64"`` (inline data URIs).
            images_dir: Directory for extracted images. When ``None``, defaults
                to ``<source>_images/`` next to each source file.
            images_scale: Image resolution multiplier (``1.0`` = 72 DPI).
                Converted to a DPI value for LiteParse.
            ocr_enabled: Run Tesseract OCR. OCR is the slowest phase — disable
                for born-digital text to maximize speed.
            **kwargs: Accepted and ignored (BaseParser protocol compliance).
        """
        self.image_mode = image_mode
        self.images_dir = images_dir
        self.images_scale = images_scale
        self.ocr_enabled = ocr_enabled

    def parse(self, source_file: Path) -> ParsedDocument:
        dpi = round(self.images_scale * 72.0)
        converter = LiteParse(
            output_format="markdown",
            image_mode="embed",
            dpi=dpi,
            ocr_enabled=self.ocr_enabled,
            # Full capability: without this ``result.images`` stays empty and
            # every figure is dropped (no IMAGE element ever reaches the
            # pipeline). Set explicitly rather than relying on the backend
            # default, which is ``False``.
            extract_images=True,
            quiet=True,
        )
        result = converter.parse(str(source_file))

        images_by_id = {img.id: img for img in result.images}
        elements: list[DocumentElement] = []
        reading_order = 0
        title = ""

        for page in result.pages:
            page_no = page.page_num
            page_md = page.markdown or ""
            added, reading_order, h1 = _markdown_to_elements(
                page_md,
                page_no,
                source_file,
                images_by_id,
                self.image_mode,
                self._resolve_images_dir(source_file),
                reading_order,
            )
            if h1 and not title:
                title = h1
            elements.extend(added)

        return ParsedDocument(
            source_file=source_file,
            elements=elements,
            title=title or source_file.stem,
            page_count=len(result.pages),
            metadata={"parser": "liteparse"},
        )

    def _resolve_images_dir(self, source_file: Path) -> Path:
        if self.images_dir is not None:
            return self.images_dir
        return source_file.parent / f"{source_file.stem}_images"


def _markdown_to_elements(
    markdown: str,
    page_no: int,
    source_file: Path,
    images_by_id: dict[str, Any],
    image_mode: str,
    images_dir: Path,
    reading_order: int,
) -> tuple[list[DocumentElement], int, str]:
    """Convert one page of Markdown into KNOX elements.

    Returns ``(elements, next_reading_order, first_h1_title)``.
    """
    lines = markdown.splitlines()
    elements: list[DocumentElement] = []
    first_h1 = ""
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2).strip()
            if level == 1 and not first_h1:
                first_h1 = text
            elements.append(
                DocumentElement(
                    type=ElementType.HEADING,
                    content=text,
                    page=page_no,
                    reading_order=reading_order,
                    heading_level=level,
                    source_file=source_file,
                )
            )
            reading_order += 1
            i += 1
            continue

        img_match = _IMAGE_REF_RE.match(stripped)
        if img_match:
            element = _build_image_element(
                img_match.group(1),
                img_match.group(2),
                page_no,
                reading_order,
                source_file,
                images_by_id,
                image_mode,
                images_dir,
            )
            if element is not None:
                elements.append(element)
                reading_order += 1
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1].strip()):
            grid, consumed = _parse_table(lines, i)
            if grid:
                elements.append(
                    DocumentElement(
                        type=ElementType.TABLE,
                        content=grid,
                        page=page_no,
                        reading_order=reading_order,
                        source_file=source_file,
                    )
                )
                reading_order += 1
            i += consumed
            continue

        # Default: gather a paragraph of consecutive plain-text lines.
        paragraph: list[str] = [stripped]
        j = i + 1
        while j < n:
            nxt = lines[j].strip()
            if not nxt or _HEADING_RE.match(nxt) or _IMAGE_REF_RE.match(nxt) or nxt.startswith("|"):
                break
            paragraph.append(nxt)
            j += 1
        elements.append(
            DocumentElement(
                type=ElementType.TEXT,
                content="\n".join(paragraph),
                page=page_no,
                reading_order=reading_order,
                source_file=source_file,
            )
        )
        reading_order += 1
        i = j

    return elements, reading_order, first_h1


def _build_image_element(
    alt_text: str,
    ref_url: str,
    page_no: int,
    reading_order: int,
    source_file: Path,
    images_by_id: dict[str, Any],
    image_mode: str,
    images_dir: Path,
) -> DocumentElement | None:
    """Resolve a markdown image ref to a KNOX IMAGE element with bytes on disk."""
    image = images_by_id.get(_ref_to_id(ref_url))
    if image is None:
        logger.debug("liteparse image ref %r has no matching bytes — skipped", ref_url)
        return None

    raw: bytes = bytes(image.bytes)
    fmt = (getattr(image, "format", "") or "png").lower()
    mime = _MIME_BY_FMT.get(fmt, "image/png")

    if image_mode == "base64":
        data_uri = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        content = data_uri
        meta: dict[str, Any] = {MetadataField.IMAGE_BASE64: data_uri, "image_mode": "base64"}
    else:
        target_dir = images_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        out_file = target_dir / f"{source_file.stem}_img_{image.id}.{fmt}"
        out_file.write_bytes(raw)
        content = str(out_file)
        meta = {MetadataField.IMAGE_FILE: str(out_file), "image_mode": "file"}

    if alt_text:
        meta["alt_text"] = alt_text

    return DocumentElement(
        type=ElementType.IMAGE,
        content=content,
        page=page_no,
        reading_order=reading_order,
        metadata=meta,
        source_file=source_file,
    )


def _ref_to_id(ref_url: str) -> str:
    """``img_p1_0.png`` / ``image_p1_0.png`` → ``p1_0`` (strip prefix + ext)."""
    name = Path(ref_url).name
    for prefix in _REF_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.rsplit(".", 1)[0]


def _parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    """Parse a GFM table starting at ``start``. Returns ``(grid, consumed)``."""
    block: list[str] = []
    j = start
    while j < len(lines) and lines[j].strip().startswith("|"):
        block.append(lines[j].strip())
        j += 1
    grid: list[list[str]] = []
    for row in block:
        if _TABLE_SEP_RE.match(row):
            continue
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        grid.append(cells)
    return grid, j - start


def _factory(options: ParserBuildOptions) -> LiteParseParser:
    return LiteParseParser(
        image_mode=options.image_mode,
        images_dir=options.images_dir,
        images_scale=options.images_scale,
    )


register_backend("liteparse", _factory, parser_cls=LiteParseParser)
