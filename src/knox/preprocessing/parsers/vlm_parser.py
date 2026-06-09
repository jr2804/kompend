"""VLM-based image parser for text extraction from image files.

Supported formats: .png, .jpg, .jpeg, .gif, .webp

When an :class:`knox.llm.LLMClient` is provided, the parser uses a
Vision Language Model to generate textual captions from images.
Without a client the parser falls back to descriptive placeholders
(basic image dimensions and format).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.preprocessing.parsers.base import BaseParser, FileExtension

logger = logging.getLogger(__name__)

_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class VLMParser(BaseParser):
    """Parser for image files using a Vision Language Model for captioning.

    Parameters
    ----------
    client :
        Optional :class:`knox.llm.client.LLMClient`. When provided,
        the VLM caption will be generated via the client's multimodal
        completion path.
    model :
        Model name (litellm format, e.g. ``"ollama/glm-ocr"``).
        Required when *client* is provided. Used for the VLM completion.
    """

    supported_extensions = {
        FileExtension.PNG,
        FileExtension.JPG,
        FileExtension.JPEG,
        FileExtension.GIF,
        FileExtension.WEBP,
    }

    def __init__(
        self,
        client: Any = None,
        model: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        if client is not None and model is None:
            logger.warning("VLMParser given a client but no model; captions will use the client's default")

    def parse(self, source_path: Path) -> ParsedDocument:
        """Parse an image file and return document elements."""
        elements: list[DocumentElement] = []
        metadata: dict[str, Any] = {"parser": "vlm"}

        # --- Image metadata element ---
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

        # --- VLM text description element ---
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
            elements.append(
                DocumentElement(
                    type=ElementType.TEXT,
                    content="[Image description not available]",
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

    def _generate_caption(self, source_path: Path) -> str | None:
        """Generate a text description for the image using the VLM.

        If a :class:`LLMClient` was provided at instantiation, a
        multimodal completion call is made. Otherwise a
        descriptive placeholder is returned with basic image stats.
        """
        if self._client is None:
            return self._fallback_caption(source_path)

        path = Path(source_path)
        if not path.is_file():
            logger.warning("Image file not found for caption: %s", path)
            return None

        try:
            raw_bytes = path.read_bytes()
            b64 = base64.b64encode(raw_bytes).decode("utf-8")
        except OSError as exc:
            logger.warning("Failed to read %s for VLM caption: %s", path, exc)
            return None

        mime = _MIME_BY_EXT.get(path.suffix.lower(), "image/png")

        try:
            result = self._client.try_complete(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in one sentence."},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        ],
                    },
                ],
                role="vlm",
                max_tokens=128,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM caption generation failed for %s: %s", path, exc)
            return None

        if result is None:
            return None

        caption = getattr(result, "content", None)
        if not caption:
            return None

        return caption.strip()

    @staticmethod
    def _extract_metadata(source_path: Path) -> tuple[list[str], int, int, str]:
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

    @staticmethod
    def _fallback_caption(source_path: Path) -> str | None:
        """Return a placeholder with image stats when no VLM is available."""
        try:
            img = PILImage.open(source_path)
            img.load()
            return f"[VLM caption unavailable — image: {img.size[0]}x{img.size[1]}, {img.mode}]"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fallback caption failed for %s: %s", source_path, exc)
            return None
