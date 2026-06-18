"""Docling-based PDF/DOCX parser with structure-preserving extraction.

Docling is a hard dependency for KNOX's primary parsing pipeline.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser, FileExtension

logger = logging.getLogger(__name__)

# Mapping from docling_core item classes to our ElementType.
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
        **kwargs: Any,
    ) -> None:
        self.image_mode = image_mode
        self.images_dir = images_dir
        self._converter = None

    def parse(self, source_path: Path) -> ParsedDocument:
        converter = self._get_converter()
        result = converter.convert(str(source_path))
        doc = result.document

        elements: list[DocumentElement] = []
        reading_order = 0

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
                # Store the path so VLMEnricher can read the file
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
                # TEXT, CODE, LIST_ITEM
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

        return ParsedDocument(
            source_path=source_path,
            elements=elements,
            title=doc.name if hasattr(doc, "name") else source_path.stem,
            page_count=len(doc.pages) if doc.pages else 0,
            metadata={"parser": "docling"},
        )

    def _get_converter(self) -> DocumentConverter:
        """Create or return the cached DocumentConverter.

        When ``image_mode`` is ``"file"`` or ``"base64"``, the PDF pipeline is
        configured with ``generate_picture_images=True`` so that
        :class:`PictureItem` instances carry actual pixel data (via the
        ``image.uri`` data-URI attribute) instead of empty structure markers.
        """
        if self._converter is None:
            if self.image_mode in ("file", "base64"):
                pipeline_opts = PdfPipelineOptions(generate_picture_images=True)
                pdf_opts = PdfFormatOption(pipeline_options=pipeline_opts)
                self._converter = DocumentConverter(format_options={InputFormat.PDF: pdf_opts})
            else:
                self._converter = DocumentConverter()
        return self._converter

    def _extract_image(self, item: Any, source_path: Path, idx: int) -> str:
        """Extract image content according to image_mode.

        Returns a string: base64 data-URI for 'base64', file path for 'file',
        or a placeholder description.

        Docling stores the actual pixel data as a base64 data URI in
        ``item.image.uri`` when ``generate_picture_images=True`` is enabled
        in the pipeline options.
        """
        # ── Try to get raw pixel data from the ImageRef URI ─────────────
        image_ref = getattr(item, "image", None)
        uri_str = None
        if image_ref is not None:
            uri = getattr(image_ref, "uri", None)
            if uri is not None:
                uri_str = str(uri)

        # ── base64 mode ────────────────────────────────────────────────
        if self.image_mode == "base64":
            if uri_str and uri_str.startswith("data:"):
                return uri_str
            # Fallback to text description
            captions = getattr(item, "captions", [])
            if captions:
                return f"[Image: {captions[0]}]"
            return f"[Image #{idx}]"

        # ── file mode ──────────────────────────────────────────────────
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

            # No pixel data available — return descriptive placeholder
            caption_text = str(captions[0]) if captions else ""
            return f"[Image: {caption_text if caption_text else 'figure'}]"

        # ── Default: placeholder ──────────────────────────────────────
        return f"[Image #{idx}]"

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_page(item: Any) -> int | None:
        """Extract page number from item provenance."""
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            return getattr(prov[0], "page_no", None)
        return None

    @staticmethod
    def _extract_table(item: Any) -> list[list[str]]:
        """Extract table cells as list-of-rows from a Docling TableItem."""
        data = getattr(item, "data", None)
        if data is None:
            # Fallback: export markdown representation
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
