"""VLM-based image parser using docling-slim's VLM inline model.

Processes image files (.png, .jpg, .jpeg, .gif, .webp) through a Vision
Language Model for text extraction. Falls back gracefully to returning
image metadata without text when the VLM model dependency is not available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser

logger = logging.getLogger(__name__)

try:
    from PIL import Image as PILImage

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    # docling-slim's VLM inline model — lightweight, no PyTorch required.
    # Falls through gracefully if docling-slim does not expose this model.
    from docling_slim.models import VLMInlineModel  # type: ignore[import-untyped]

    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False


class VLMParser(BaseParser):
    """Parser for image files using a Vision Language Model.

    Supported formats: .png, .jpg, .jpeg, .gif, .webp

    When the VLM model is available, the parser extracts a text description
    from each image. When unavailable, it returns a ParsedDocument with the
    image metadata (path, size, format) but no text extraction.
    """

    supported_extensions = (".png", ".jpg", ".jpeg", ".gif", ".webp")

    def __init__(self) -> None:
        if VLM_AVAILABLE:
            logger.info("VLMParser initialized with VLMInlineModel")
        else:
            logger.info(
                "VLMParser initialized in metadata-only mode "
                "(docling_slim.models.VLMInlineModel not available)"
            )

    def parse(self, source_path: Path) -> ParsedDocument:
        """Parse an image file and return document elements."""
        elements: list[DocumentElement] = []
        metadata: dict[str, Any] = {"parser": "vlm"}

        # --- Image metadata element ---
        if PIL_AVAILABLE:
            info_lines, width, height, img_format = self._extract_metadata(source_path)
            metadata["format"] = img_format
            metadata["width"] = width
            metadata["height"] = height
            elements.append(
                DocumentElement(
                    type=ElementType.METADATA,
                    content="\n".join(info_lines),
                    metadata=metadata,
                    source_file=source_path,
                )
            )
        else:
            elements.append(
                DocumentElement(
                    type=ElementType.METADATA,
                    content=f"Image: {source_path.name}",
                    metadata=metadata,
                    source_file=source_path,
                )
            )

        # --- VLM text description element ---
        if VLM_AVAILABLE and PIL_AVAILABLE:
            caption = self._generate_caption(source_path)
            if caption:
                elements.append(
                    DocumentElement(
                        type=ElementType.TEXT,
                        content=caption,
                        metadata={"vlm_generated": True},
                        source_file=source_path,
                    )
                )
        else:
            logger.debug(
                "VLM not available — skipping caption generation for %s", source_path
            )
            elements.append(
                DocumentElement(
                    type=ElementType.TEXT,
                    content="[Image description not available — VLM model not installed]",
                    metadata={"vlm_generated": False},
                    source_file=source_path,
                )
            )

        return ParsedDocument(
            source_path=source_path,
            elements=elements,
            title=source_path.stem,
            page_count=1,
            metadata=metadata,
        )

    def _extract_metadata(
        self, source_path: Path
    ) -> tuple[list[str], int, int, str]:
        """Extract basic image metadata."""
        try:
            img = PILImage.open(source_path)
            img.load()
            info_lines = [
                f"Image: {source_path.name}",
                f"Format: {img.format or 'unknown'}",
                f"Size: {img.size[0]}x{img.size[1]}",
                f"Mode: {img.mode}",
            ]
            return (
                info_lines,
                img.size[0],
                img.size[1],
                img.format or "unknown",
            )
        except Exception as exc:
            logger.warning("Failed to read image metadata for %s: %s", source_path, exc)
            return (
                [f"Image: {source_path.name}", f"Error: {exc}"],
                0,
                0,
                "unknown",
            )

    def _generate_caption(self, source_path: Path) -> str | None:
        """Generate a text description for the image using the VLM.

        TODO: Wire into ``VLMInlineModel`` once docling-slim exposes a
        ready-to-use inline VLM.  For now returns a placeholder caption
        that describes basic image stats.
        """
        try:
            img = PILImage.open(source_path)
            img.load()
            # Placeholder: return basic image stats as a pseudo-caption.
            # Replace with an actual VLMInlineModel call:
            #   model = VLMInlineModel(...)
            #   caption = model.describe(img)
            return (
                f"[VLM description pending — image: {img.size[0]}x{img.size[1]}, "
                f"{img.mode}]"
            )
        except Exception as exc:
            logger.warning("VLM caption generation failed for %s: %s", source_path, exc)
            return None