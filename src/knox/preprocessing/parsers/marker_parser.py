"""Marker parser (datalab-to/marker) — high-accuracy PDF/office/HTML parser.

Marker is a deep-learning document converter that renders PDF, DOCX, PPTX,
XLSX, EPUB, and HTML into Markdown with structured tables, equations, and
extracted figures. It reports the highest benchmark accuracy among the
KNOX-supported parsers (95.67 heuristic vs docling 86.7).

Two conversion modes:
* **``fast``** (default on CPU/MPS): lightweight rf-detr/onnx detectors for
  layout + tables; block-OCRs only garbled/empty content.
* **``balanced``** (default on CUDA GPU): full VLM layout model + full-page OCR.

The ``marker-pdf`` package is an optional dependency — install with::

    uv sync --group marker

Marker has no internal figure captioning (the ``LLMImageDescriptionProcessor``
is opt-in via ``use_llm=True`` and requires a configured LLM service), so
``figure_enrichment`` is ``False`` and every extracted figure still flows
through ``VLMEnricher``.
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.renderers.markdown import MarkdownOutput

from knox.core.models import DocumentElement, ElementType, MetadataField, ParsedDocument
from knox.logging import get_logger
from knox.preprocessing.parsers._registry import ParserBuildOptions, register_backend
from knox.preprocessing.parsers.base import (
    IMAGE_SCALE_DEFAULT,
    BaseParser,
    FileExtension,
)

# Marker emits a page separator of ``"-" * 48`` by default between pages.
_PAGE_SEP_RE = re.compile(r"\n-{48,}\n")
# ``![alt](image_id.png)`` — Marker image refs.
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

logger = get_logger(__name__)


class MarkerParser(BaseParser):
    """Parser using Marker for high-accuracy Markdown reconstruction.

    Marker's ``PdfConverter`` handles all supported formats (PDF, DOCX, PPTX,
    XLSX, EPUB, HTML, images) via internal provider dispatch. The returned
    ``MarkdownOutput`` is walked line-by-line and mapped to KNOX
    ``ElementType`` values (headings, tables, images, text).
    """

    supported_extensions: set[FileExtension] = {
        FileExtension.PDF,
        FileExtension.DOCX,
        FileExtension.PPTX,
        FileExtension.XLSX,
        FileExtension.EPUB,
        FileExtension.HTML,
        FileExtension.HTM,
        FileExtension.PNG,
        FileExtension.JPG,
        FileExtension.JPEG,
    }

    # No internal captioning — VLM still captions every figure.
    figure_enrichment: bool = False

    def __init__(
        self,
        *,
        image_mode: str = "file",
        images_dir: Path | None = None,
        images_scale: float = IMAGE_SCALE_DEFAULT,
        mode: str | None = None,
        use_llm: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the parser.

        Args:
            image_mode: ``"file"`` (write images to ``images_dir``) or
                ``"base64"`` (inline data URIs).
            images_dir: Directory for extracted images. When ``None``, defaults
                to ``<source>_images/`` next to each source file.
            images_scale: Unused by Marker (resolution is model-controlled);
                accepted for ``ParserBuildOptions`` protocol compliance.
            mode: Marker conversion mode — ``"fast"`` or ``"balanced"``.
                When ``None``, Marker picks a device-appropriate default.
            use_llm: Enable Marker's LLM processors (higher quality, requires a
                configured LLM service). Defaults to ``False`` (model-only).
            **kwargs: Accepted and ignored (BaseParser protocol compliance).
        """
        self.image_mode = image_mode
        self.images_dir = images_dir
        self.images_scale = images_scale
        self.mode = mode
        self.use_llm = use_llm

    def parse(self, source_file: Path) -> ParsedDocument:
        config: dict[str, Any] = {"use_llm": self.use_llm}
        if self.mode is not None:
            config["mode"] = self.mode

        converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config=config,
        )
        rendered = converter(str(source_file))
        if not isinstance(rendered, MarkdownOutput):  # pragma: no cover - defensive
            msg = f"Marker returned unexpected type {type(rendered).__name__}"
            raise TypeError(msg)

        markdown = rendered.markdown or ""
        images: dict[str, Any] = rendered.images or {}

        # Split into pages using the page separator. When pagination is off,
        # the whole document is a single page.
        page_chunks = _PAGE_SEP_RE.split(markdown) if _PAGE_SEP_RE.search(markdown) else [markdown]

        elements: list[DocumentElement] = []
        reading_order = 0
        title = ""

        for page_no, page_md in enumerate(page_chunks, start=1):
            added, reading_order, h1 = _markdown_to_elements(
                page_md,
                page_no,
                source_file,
                images=images,
                image_mode=self.image_mode,
                images_dir=self._resolve_images_dir(source_file),
                reading_order=reading_order,
            )
            if h1 and not title:
                title = h1
            elements.extend(added)

        page_count = self._extract_page_count(rendered) or len(page_chunks)

        return ParsedDocument(
            source_file=source_file,
            elements=elements,
            title=title or source_file.stem,
            page_count=page_count,
            metadata={"parser": "marker"},
        )

    def _resolve_images_dir(self, source_file: Path) -> Path:
        if self.images_dir is not None:
            return self.images_dir
        return source_file.parent / f"{source_file.stem}_images"

    @staticmethod
    def _extract_page_count(rendered: MarkdownOutput) -> int | None:
        meta = getattr(rendered, "metadata", None) or {}
        pages = meta.get("pages")
        if isinstance(pages, list):
            return len(pages)
        if isinstance(pages, int):
            return pages
        return None


def _markdown_to_elements(
    markdown: str,
    page_no: int,
    source_file: Path,
    *,
    images: dict[str, Any],
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
                images=images,
                image_mode=image_mode,
                images_dir=images_dir,
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
    *,
    images: dict[str, Any],
    image_mode: str,
    images_dir: Path,
) -> DocumentElement | None:
    """Resolve a markdown image ref to a KNOX IMAGE element with bytes on disk."""
    img_name = Path(ref_url).name
    pil_img = images.get(img_name) or images.get(ref_url)
    if pil_img is None:
        logger.debug("marker image ref %r has no matching bytes — skipped", ref_url)
        return None

    fmt = (getattr(pil_img, "format", "") or "png").lower()
    mime = _MIME_BY_FMT.get(fmt, "image/png")
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt if fmt != "jpg" else "jpeg")
    raw: bytes = buf.getvalue()

    if image_mode == "base64":
        data_uri = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        content = data_uri
        meta: dict[str, Any] = {
            MetadataField.IMAGE_BASE64: data_uri,
            "image_mode": "base64",
        }
    else:
        target_dir = images_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{source_file.stem}_{img_name}"
        out_path.write_bytes(raw)
        content = str(out_path)
        meta = {MetadataField.IMAGE_PATH: str(out_path), "image_mode": "file"}

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


def _factory(options: ParserBuildOptions) -> MarkerParser:
    return MarkerParser(
        image_mode=options.image_mode,
        images_dir=options.images_dir,
        images_scale=options.images_scale,
    )


register_backend("marker", _factory, parser_cls=MarkerParser)
