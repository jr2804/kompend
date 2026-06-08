"""Docling-based PDF/DOCX parser with structure-preserving extraction.

Docling is a hard dependency for KNOX's primary parsing pipeline.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser

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
    """

    supported_extensions = (".pdf", ".docx")

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
                elements.append(
                    DocumentElement(
                        type=element_type,
                        content=content,
                        page=page_no,
                        reading_order=reading_order,
                        metadata={
                            "docling_type": class_name,
                            "image_mode": self.image_mode,
                        },
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

    def _get_converter(self):
        """Create or return the cached DocumentConverter."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    def _extract_image(self, item: Any, source_path: Path, idx: int) -> str:
        """Extract image content according to image_mode.

        Returns a string: base64 data-URI for 'base64', file path for 'file',
        or a placeholder description.
        """
        if self.image_mode == "base64":
            # Try to get image bytes from docling's PictureItem
            image = getattr(item, "image", None)
            if image is not None:
                pil_image = image.get_pil_image() if hasattr(image, "get_pil_image") else None
                if pil_image is not None:
                    buf = BytesIO()
                    pil_image.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    return f"data:image/png;base64,{b64}"
            # Fallback to text description
            captions = getattr(item, "captions", [])
            if captions:
                return f"[Image: {captions[0]}]"
            return f"[Image #{idx}]"

        if self.image_mode == "file":
            captions = getattr(item, "captions", [])
            captions[0] if captions else f"image_{idx}"
            stem = source_path.stem
            if self.images_dir:
                out_path = self.images_dir / f"{stem}_img_{idx}.png"
                return str(out_path)
            return f"{stem}_img_{idx}.png"

        # Default: placeholder
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
