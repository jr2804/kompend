"""LiteParse-based parser (run-llama/liteparse) — fast Rust-core alternative.

LiteParse reconstructs documents to Markdown with embedded image bytes. It has
no LLM/captioning of its own, so ``figure_enrichment`` is ``False`` and every
extracted figure still flows through ``VLMEnricher`` (per-figure bypass only
applies when a description is already present, which LiteParse never sets).

Office formats (DOC/DOCX/PPT/PPTX/XLS/XLSX) are converted via an external
LibreOffice install; PDF is handled natively via PDFium.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from liteparse import LiteParse

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

#: Filename prefixes liteparse uses for image refs. Newer releases emit
#: ``img_<id>.png``; older ones ``image_<id>.png``. Missing the prefix meant
#: the id lookup never matched and every figure was dropped.
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
        images_dir = self._resolve_images_dir(source_file)
        resolve_image = _image_resolver(images_by_id)

        for page in result.pages:
            page_no = page.page_num
            page_md = page.markdown or ""
            added, reading_order, h1 = markdown_to_elements(
                page_md,
                MarkdownContext(
                    source_file=source_file,
                    image_mode=self.image_mode,
                    images_dir=images_dir,
                    page=page_no,
                    reading_order=reading_order,
                    resolve_image=resolve_image,
                ),
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


def _image_resolver(images_by_id: dict[str, Any]) -> Callable[[str, str], ImagePayload | None]:
    """Look up figure bytes by ref id in the liteparse result."""

    def resolve(_alt_text: str, ref_url: str) -> ImagePayload | None:
        image = images_by_id.get(_ref_to_id(ref_url))
        if image is None:
            logger.debug("liteparse image ref %r has no matching bytes — skipped", ref_url)
            return None
        return ImagePayload(
            data=bytes(image.bytes),
            format=(getattr(image, "format", "") or "png").lower(),
            name=str(image.id),
        )

    return resolve


def _ref_to_id(ref_url: str) -> str:
    """``img_p1_0.png`` / ``image_p1_0.png`` → ``p1_0`` (strip prefix + ext)."""
    name = Path(ref_url).name
    for prefix in _REF_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.rsplit(".", 1)[0]


def _factory(options: ParserBuildOptions) -> LiteParseParser:
    return LiteParseParser(
        image_mode=options.image_mode,
        images_dir=options.images_dir,
        images_scale=options.images_scale,
    )


register_backend("liteparse", _factory, parser_cls=LiteParseParser)
