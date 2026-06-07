"""VLM-based image caption enricher for KNOX Tier 3.

Analyses IMAGE chunks and adds descriptive captions as entities. All
LLM calls go through the shared :class:`knox.llm.LLMClient` so VLM
failover and observability work the same way as text models.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from knox.core.models import ElementType, ResolvedChunk
from knox.llm.client import LLMClient

logger = logging.getLogger(__name__)


_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class VLMEnricher:
    """Enrich IMAGE chunks by producing textual captions via a VLM.

    Routes the multimodal call through the shared LLM client so the
    failover chain (Q12) and per-stage configuration (Q13) apply
    consistently to vision models.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        role: str = "vlm",
    ) -> None:
        self._client = client or LLMClient()
        self._role = role

    def enrich(self, resolved: ResolvedChunk) -> ResolvedChunk:
        """Add VLM-generated captions to IMAGE chunks. Never raises."""
        if ElementType.IMAGE not in resolved.chunk.element_types:
            return resolved

        image_path = resolved.chunk.metadata.get("image_path") or resolved.metadata.get("image_path")

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
        except FileNotFoundError as exc:
            logger.warning("VLM image not found for chunk %s: %s", resolved.chunk.id, exc)
            resolved.entities.append({
                "name": "vlm_error",
                "type": "image_description",
                "context": f"Image file not found: {image_path}",
            })
            return resolved

        if caption is None:
            resolved.entities.append({
                "name": "vlm_error",
                "type": "image_description",
                "context": "VLM caption generation failed",
            })
            return resolved

        resolved.entities.append({
            "name": f"vlm_caption_{resolved.chunk.id}",
            "type": "image_description",
            "context": caption,
        })
        resolved.metadata["vlm_enriched"] = True
        return resolved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_caption(self, image_path: str | Path) -> str | None:
        """Run the VLM on *image_path* and return a text caption.

        Returns ``None`` when the LLM client returns ``None`` (all
        models in the failover chain failed). Raises
        :class:`FileNotFoundError` when the image file is missing.
        """
        path = Path(image_path)
        if not path.is_file():
            msg = f"Image not found: {path}"
            raise FileNotFoundError(msg)

        raw_bytes = path.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        mime = _MIME_BY_EXT.get(path.suffix.lower(), "image/png")

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
            role=self._role,
            max_tokens=128,
        )
        if result is None:
            return None
        return result.content
