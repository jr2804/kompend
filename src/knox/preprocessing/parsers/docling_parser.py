"""Docling-based PDF/DOCX parser with structure-preserving extraction.

Docling is a hard dependency for KNOX's primary parsing pipeline.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from knox.core.config import PdfConfig
from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser, FileExtension

logger = logging.getLogger(__name__)

_ITEM_TYPE_MAP: dict[str, ElementType] = {
    "TextItem": ElementType.TEXT,
    "SectionHeaderItem": ElementType.HEADING,
    "TitleItem": ElementType.HEADING,
    "TableItem": ElementType.TABLE,
    "PictureItem": ElementType.IMAGE,
    "FormulaItem": ElementType.FORMULA,
    "CodeItem": ElementType.CODE,
    "ListItem": ElementType.LIST_ITEM,
}


class DoclingParser(BaseParser):
    """Parser using Docling for structure-preserving extraction from PDF/DOCX.

    Uses Docling's ``iterate_items()`` API to walk the document body and map
    each item (TextItem, SectionHeaderItem, TableItem, PictureItem, etc.) to
    the corresponding KNOX ``ElementType`` with full structure preservation.

    When ``image_mode`` is ``"file"`` or ``"base64"``, the parser configures
    Docling's pipeline with ``generate_picture_images=True`` so actual pixel
    data is extracted from embedded PDF images, not just structure markers.

    Large PDFs (more than ``pdf_config.max_pages_before_chunking`` pages) are
    processed in sequential page-range segments to prevent ONNX/OCR OOM.
    """

    supported_extensions: set[FileExtension] = {
        FileExtension.PDF,
        FileExtension.DOCX,
        FileExtension.PPTX,
        FileExtension.HTML,
        FileExtension.MD,
        FileExtension.MARKDOWN,
        FileExtension.EPUB,
        FileExtension.TEX,
        FileExtension.LTX,
        FileExtension.CSV,
        FileExtension.XLSX,
        FileExtension.ADOC,
        FileExtension.EML,
        FileExtension.MSG,
    }

    def __init__(
        self,
        image_mode: str = "file",
        images_dir: Path | None = None,
        pdf_config: PdfConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the parser.

        Args:
            image_mode: How to handle images (``"file"`` or ``"base64"``).
            images_dir: Directory for extracted images. When ``None``, defaults
                to ``<source>_images/`` next to the source file.
            pdf_config: PDF processing thresholds. When ``None``, uses
                :class:`PdfConfig` defaults.
            **kwargs: Accepted and ignored (BaseParser protocol compliance).
        """
        self.image_mode = image_mode
        self.images_dir = images_dir
        self._pdf_config = pdf_config or PdfConfig()
        self._converter: DocumentConverter | None = None

    def parse(self, source_path: Path) -> ParsedDocument:
        converter = self._get_converter()

        if source_path.suffix.lower() == ".pdf" and self._pdf_config.max_pages_before_chunking > 0:
            page_count = self._count_pdf_pages(source_path)
            if page_count > self._pdf_config.max_pages_before_chunking:
                logger.info(
                    "Large PDF (%d pages) — processing in chunks of %d pages",
                    page_count,
                    self._pdf_config.page_chunk_size,
                )
                return self._parse_chunked(source_path, converter, page_count)

        result = converter.convert(str(source_path))
        doc = result.document
        elements, _ = self._extract_elements(doc, source_path)
        page_count = len(doc.pages) if doc.pages else 0
        title = doc.name if hasattr(doc, "name") else source_path.stem
        self._unload_result(result)

        return ParsedDocument(
            source_path=source_path,
            elements=elements,
            title=title,
            page_count=page_count,
            metadata={"parser": "docling"},
        )

    def _parse_chunked(
        self,
        source_path: Path,
        converter: DocumentConverter,
        page_count: int,
    ) -> ParsedDocument:
        """Process a large PDF in page-range chunks to prevent OOM.

        Docling's ``convert(page_range=...)`` accepts a 1-indexed
        ``(start, end)`` tuple. After each chunk the backend is unloaded
        to release ONNX/OCR caches that GC does not reclaim.
        """
        chunk_size = self._pdf_config.page_chunk_size
        all_elements: list[DocumentElement] = []
        reading_order = 0
        title = source_path.stem

        start_page = 1
        while start_page <= page_count:
            end_page = min(start_page + chunk_size - 1, page_count)
            logger.debug("Processing pages %d-%d of %d", start_page, end_page, page_count)

            result = converter.convert(
                str(source_path),
                page_range=(start_page, end_page),
            )
            doc = result.document
            if hasattr(doc, "name") and doc.name:
                title = doc.name

            elements, reading_order = self._extract_elements(
                doc,
                source_path,
                reading_order_start=reading_order,
            )
            all_elements.extend(elements)

            self._unload_result(result)
            start_page = end_page + 1

        return ParsedDocument(
            source_path=source_path,
            elements=all_elements,
            title=title,
            page_count=page_count,
            metadata={"parser": "docling", "chunked": True},
        )

    def _extract_elements(
        self,
        doc: Any,
        source_path: Path,
        reading_order_start: int = 0,
    ) -> tuple[list[DocumentElement], int]:
        """Walk Docling document items and build KNOX DocumentElements.

        Returns ``(elements, next_reading_order)``.
        """
        elements: list[DocumentElement] = []
        reading_order = reading_order_start

        for item, _level in doc.iterate_items():
            class_name = type(item).__name__
            element_type = _ITEM_TYPE_MAP.get(class_name)
            if element_type is None:
                logger.debug("Skipping unknown docling item type: %s", class_name)
                continue

            page_no = self._extract_page(item)

            if element_type == ElementType.HEADING:
                level = getattr(item, "level", None) or (1 if class_name == "TitleItem" else 2)
                content = getattr(item, "text", "")
                elements.append(
                    DocumentElement(
                        type=element_type,
                        content=content,
                        page=page_no,
                        reading_order=reading_order,
                        heading_level=level,
                        metadata={"docling_type": class_name},
                    )
                )

            elif element_type == ElementType.TABLE:
                table_data = self._extract_table(item)
                elements.append(
                    DocumentElement(
                        type=element_type,
                        content=table_data,
                        page=page_no,
                        reading_order=reading_order,
                        metadata={"docling_type": class_name},
                    )
                )

            elif element_type == ElementType.IMAGE:
                content = self._extract_image(item, source_path, reading_order)
                img_meta: dict[str, Any] = {
                    "docling_type": class_name,
                    "image_mode": self.image_mode,
                }
                if self.image_mode == "file" and content and not content.startswith("["):
                    img_meta["image_path"] = content
                elif self.image_mode == "base64" and content.startswith("data:"):
                    img_meta["image_base64"] = content
                captions = getattr(item, "captions", [])
                if captions:
                    img_meta["caption"] = str(captions[0])
                elements.append(
                    DocumentElement(
                        type=element_type,
                        content=content,
                        page=page_no,
                        reading_order=reading_order,
                        metadata=img_meta,
                    )
                )

            elif element_type == ElementType.FORMULA:
                text = getattr(item, "text", "")
                elements.append(
                    DocumentElement(
                        type=element_type,
                        content=text,
                        page=page_no,
                        reading_order=reading_order,
                        metadata={"docling_type": class_name},
                    )
                )

            else:
                text = getattr(item, "text", "")
                if not text.strip():
                    continue
                elements.append(
                    DocumentElement(
                        type=element_type,
                        content=text,
                        page=page_no,
                        reading_order=reading_order,
                        metadata={"docling_type": class_name},
                    )
                )

            reading_order += 1

        return elements, reading_order

    def _get_converter(self) -> DocumentConverter:
        if self._converter is None:
            if self.image_mode in ("file", "base64"):
                pipeline_opts = PdfPipelineOptions(generate_picture_images=True)
                pdf_opts = PdfFormatOption(pipeline_options=pipeline_opts)
                self._converter = DocumentConverter(format_options={InputFormat.PDF: pdf_opts})
            else:
                self._converter = DocumentConverter()
        return self._converter

    def _extract_image(self, item: Any, source_path: Path, idx: int) -> str:
        image_ref = getattr(item, "image", None)
        uri_str = None
        if image_ref is not None:
            uri = getattr(image_ref, "uri", None)
            if uri is not None:
                uri_str = str(uri)

        if self.image_mode == "base64":
            if uri_str and uri_str.startswith("data:"):
                return uri_str
            captions = getattr(item, "captions", [])
            if captions:
                return f"[Image: {captions[0]}]"
            return f"[Image #{idx}]"

        if self.image_mode == "file":
            captions = getattr(item, "captions", [])
            stem = source_path.stem
            images_dir = self.images_dir or (source_path.parent / f"{stem}_images")
            images_dir.mkdir(parents=True, exist_ok=True)
            out_path = images_dir / f"{stem}_img_{idx}.png"

            if uri_str and uri_str.startswith("data:"):
                try:
                    header, b64data = uri_str.split(",", 1)
                    raw = base64.b64decode(b64data)
                    out_path.write_bytes(raw)
                    return str(out_path.resolve())
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to save image from data URI for %s", out_path.name, exc_info=True)

            caption_text = str(captions[0]) if captions else ""
            return f"[Image: {caption_text if caption_text else 'figure'}]"

        return f"[Image #{idx}]"

    @staticmethod
    def _count_pdf_pages(source_path: Path) -> int:
        """Count pages in a PDF using pypdfium2 (lightweight, no OCR)."""
        document = pdfium.PdfDocument(str(source_path))
        count = len(document)
        document.close()
        return count

    @staticmethod
    def _unload_result(result: Any) -> None:
        """Unload Docling conversion result to free ONNX/OCR caches."""
        try:
            backend = getattr(getattr(result, "input", None), "_backend", None)
            if backend is not None:
                backend.unload()
        except Exception:  # noqa: BLE001
            logger.debug("Failed to unload Docling backend", exc_info=True)

    @staticmethod
    def _extract_page(item: Any) -> int | None:
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            return getattr(prov[0], "page_no", None)
        return None

    @staticmethod
    def _extract_table(item: Any) -> list[list[str]]:
        data = getattr(item, "data", None)
        if data is None:
            text = getattr(item, "text", "")
            return [[text]]

        cells = getattr(data, "table_cells", [])
        num_rows = getattr(data, "num_rows", 0)
        num_cols = getattr(data, "num_cols", 0)

        if not cells or num_rows == 0:
            text = getattr(item, "text", "")
            return [[text]]

        grid: list[list[str]] = [[""] * num_cols for _ in range(num_rows)]
        for cell in cells:
            row = getattr(cell, "start_row_offset_idx", 0) or 0
            col = getattr(cell, "start_col_offset_idx", 0) or 0
            text = getattr(cell, "text", "")
            if 0 <= row < num_rows and 0 <= col < num_cols:
                grid[row][col] = text
        return grid
