"""VLM-based image caption enricher for KNOX Tier 3.

Analyses IMAGE chunks and adds descriptive captions as entities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import litellm

from knox.core.models import ElementType, ResolvedChunk

logger = logging.getLogger(__name__)


class VLMEnricher:
    """Enrich IMAGE chunks by producing textual captions via a VLM.

    The enricher checks whether each resolved chunk contains IMAGE element
    types and, if so, attempts to load the image from ``metadata.image_path``
    and run it through a VLM for description generation.

    The default implementation uses ``docling-slim``'s capabilities when
    available; otherwise it falls back to a no-op.
    """

    def __init__(self, vlm_model: str = "phi-3-vision", provider: str = "ollama") -> None:
        """Initialise the VLM enricher.

        Args:
            vlm_model: Vision-language model name.
            provider: Provider prefix for litellm.
        """
        self._vlm_model = vlm_model
        self._provider = provider
        self._backend = "litellm"

    def enrich(self, resolved: ResolvedChunk) -> ResolvedChunk:
        """Add VLM-generated captions to IMAGE chunks.

        Args:
            resolved: A resolved chunk (possibly containing IMAGE elements).

        Returns:
            The same resolved chunk, potentially with added entities containing
            VLM captions. Never raises.
        """
        # ------------------------------------------------------------------
        # Fast no-op path: no IMAGE elements.
        # ------------------------------------------------------------------
        if ElementType.IMAGE not in resolved.chunk.element_types:
            return resolved

        # ------------------------------------------------------------------
        # Try to load the image and generate a caption.
        # ------------------------------------------------------------------
        image_path = resolved.chunk.metadata.get("image_path") or resolved.metadata.get("image_path")

        # If we have no path but the chunk text has image content, mark it.
        if not image_path:
            logger.debug("No image_path in metadata for chunk %s — adding placeholder", resolved.chunk.id)
            resolved.entities.append({
                "name": "vlm_placeholder",
                "type": "image_description",
                "context": "Image present — VLM caption not available (no image path)",
            })
            return resolved

        try:
            caption = self._generate_caption(image_path)
            resolved.entities.append({
                "name": f"vlm_caption_{resolved.chunk.id}",
                "type": "image_description",
                "context": caption,
            })
            resolved.metadata["vlm_enriched"] = True
            resolved.metadata["vlm_model"] = f"{self._provider}/{self._vlm_model}"
        except Exception:
            logger.exception("VLM caption generation failed for %s", image_path)
            resolved.entities.append({
                "name": "vlm_error",
                "type": "image_description",
                "context": "VLM caption generation failed",
            })

        return resolved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_caption(self, image_path: str | Path) -> str:
        """Run the VLM on *image_path* and return a text caption."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        if self._backend == "docling":
            return self._caption_via_docling(path)
        elif self._backend == "litellm":
            return self._caption_via_litellm(path)
        else:
            raise RuntimeError("No VLM backend configured")

    def _caption_via_docling(self, path: Path) -> str:
        """Use docling-slim's built-in VLM for captioning."""
        from docling_slim.datamodel import DocumentDescription

        desc = DocumentDescription.from_image(path)
        return desc.text if hasattr(desc, "text") else str(desc)

    def _caption_via_litellm(self, path: Path) -> str:
        """Use litellm vision endpoint for captioning (base64 image)."""
        import base64

        raw_bytes = path.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        mime = self._guess_mime(path)

        response = litellm.completion(
            model=f"{self._provider}/{self._vlm_model}",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one sentence."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }
            ],
            max_tokens=128,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _guess_mime(path: Path) -> str:
        """Guess MIME type from file extension."""
        ext = path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_map.get(ext, "image/png")