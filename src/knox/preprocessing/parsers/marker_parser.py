"""Marker parser (datalab-to/marker) — high-accuracy PDF/office/HTML parser.

Marker is a deep-learning document converter that renders PDF, DOCX, PPTX,
XLSX, EPUB, and HTML into Markdown with structured tables, equations, and
extracted figures. It reports the highest benchmark accuracy among the
KNOX-supported parsers (95.67 heuristic vs docling 86.7).

Two conversion modes:
* **``fast``** (default on CPU/MPS): lightweight rf-detr/onnx detectors for
  layout + tables; block-OCRs only garbled/empty content.
* **``balanced``** (default on CUDA GPU): full VLM layout model + full-page OCR.

The ``marker-pdf`` package is an optional dependency — install manually
(it conflicts with glmocr on pillow, so it cannot be a locked extra)::

    uv pip install marker-pdf

Marker has no internal figure captioning (the ``LLMImageDescriptionProcessor``
is opt-in via ``use_llm=True`` and requires a configured LLM service), so
``figure_enrichment`` is ``False`` and every extracted figure still flows
through ``VLMEnricher``.
"""

from __future__ import annotations

import importlib
import io
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from knox.core.models import DocumentElement, ParsedDocument
from knox.logging import get_logger
from knox.preprocessing.parsers._registry import ParserBuildOptions, register_backend
from knox.preprocessing.parsers.base import (
    IMAGE_SCALE_DEFAULT,
    BaseParser,
    FileExtension,
)
from knox.preprocessing.parsers.markdown import (
    ImagePayload,
    MarkdownContext,
    markdown_to_elements,
)

# Marker emits a page separator of ``"-" * 48`` by default between pages.
_PAGE_SEP_RE = re.compile(r"\n-{48,}\n")

PdfConverter = importlib.import_module("marker.converters.pdf").PdfConverter
create_model_dict = importlib.import_module("marker.models").create_model_dict
MarkdownOutput = importlib.import_module("marker.renderers.markdown").MarkdownOutput

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
        images_dir = self._resolve_images_dir(source_file)

        for page_no, page_md in enumerate(page_chunks, start=1):
            added, reading_order, h1 = markdown_to_elements(
                page_md,
                MarkdownContext(
                    source_file=source_file,
                    image_mode=self.image_mode,
                    images_dir=images_dir,
                    page=page_no,
                    reading_order=reading_order,
                    resolve_image=_marker_image_resolver(images),
                ),
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


def _marker_image_resolver(images: dict[str, Any]) -> Callable[[str, str], ImagePayload | None]:
    """Look up figure bytes in marker's ``rendered.images`` (name -> PIL image)."""

    def resolve(_alt_text: str, ref_url: str) -> ImagePayload | None:
        img_name = Path(ref_url).name
        pil_img = images.get(img_name) or images.get(ref_url)
        if pil_img is None:
            logger.debug("marker image ref %r has no matching bytes — skipped", ref_url)
            return None
        fmt = (getattr(pil_img, "format", "") or "png").lower()
        buf = io.BytesIO()
        pil_img.save(buf, format=fmt if fmt != "jpg" else "jpeg")
        return ImagePayload(data=buf.getvalue(), format=fmt, name=Path(img_name).stem)

    return resolve


def _factory(options: ParserBuildOptions) -> MarkerParser:
    return MarkerParser(
        image_mode=options.image_mode,
        images_dir=options.images_dir,
        images_scale=options.images_scale,
    )


register_backend("marker", _factory, parser_cls=MarkerParser)
