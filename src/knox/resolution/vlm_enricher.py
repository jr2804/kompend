"""VLM-based image caption enricher for KNOX Tier 3.

Analyses IMAGE chunks and adds descriptive captions as entities. Each
image is evaluated by the VLM for informativeness — logos, icons,
decorative elements, and blank images are skipped automatically.
All LLM calls go through the shared :class:`knox.llm.LLMClient`.
"""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

from knox.core.models import Chunk, ElementType, MetadataField, ResolvedChunk
from knox.llm.client import LLMClient
from knox.llm.roles import LLMRole
from knox.logging import get_logger
from knox.resolution.base import BaseResolver

logger = get_logger(__name__)


_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class VLMEnricher(BaseResolver):
    """Enrich IMAGE chunks by producing textual captions via a VLM.

    Routes the multimodal call through the shared LLM client so the
    failover chain (Q12) and per-stage configuration (Q13) apply
    consistently to vision models.

    Implements :class:`BaseResolver` so it can participate in a
    :class:`ResolverChain`. The :meth:`resolve` method wraps a bare
    :class:`Chunk` into a :class:`ResolvedChunk` before enriching.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        role: str = LLMRole.VLM,
        min_image_bytes: int = 2048,
        min_image_dimension: int = 64,
    ) -> None:
        self._client = client
        self._role = role
        self._min_bytes = min_image_bytes
        self._min_dim = min_image_dimension

    def resolve(self, chunk: Chunk) -> ResolvedChunk:
        """BaseResolver interface: wrap chunk then enrich."""
        resolved = ResolvedChunk(chunk=chunk)
        return self.enrich(resolved)

    def enrich(self, resolved: ResolvedChunk) -> ResolvedChunk:  # noqa: PLR0911
        """Add VLM-generated captions to IMAGE chunks. Never raises."""
        if ElementType.IMAGE not in resolved.chunk.element_types:
            return resolved

        description = resolved.chunk.metadata.get("description")
        if description:
            resolved.entities.append(
                {
                    "name": f"parser_caption_{resolved.chunk.id}",
                    "type": "image_description",
                    "context": str(description),
                }
            )
            resolved.metadata["figure_enriched"] = True
            return resolved

        image_path = resolved.chunk.metadata.get(MetadataField.IMAGE_PATH) or resolved.metadata.get(MetadataField.IMAGE_PATH)

        if not image_path:
            logger.debug("No image_path in metadata for chunk %s — adding placeholder", resolved.chunk.id)
            resolved.entities.append(
                {
                    "name": "vlm_placeholder",
                    "type": "image_description",
                    "context": "Image present — VLM caption not available (no image path)",
                }
            )
            return resolved

        ctx = resolved.chunk.heading_context or ""
        caption_text = resolved.chunk.metadata.get("caption", "")
        context_parts = [f"Section: {ctx}"] if ctx else []
        if caption_text:
            context_parts.append(f"Caption: {caption_text}")
        context_hint = " | ".join(context_parts) if context_parts else ""

        skip, reason = self._should_skip(image_path)
        if skip:
            logger.debug("VLM pre-filter skipping %s: %s", resolved.chunk.id, reason)
            resolved.entities.append(
                {
                    "name": "vlm_filtered",
                    "type": "image_description",
                    "context": f"Image filtered (decorative/too small): {reason}",
                }
            )
            resolved.metadata["vlm_filtered"] = True
            return resolved

        try:
            caption = self._generate_caption(image_path, context_hint=context_hint)
        except FileNotFoundError:
            resolved.entities.append(
                {
                    "name": "vlm_error",
                    "type": "image_description",
                    "context": f"Image file not found: {image_path}",
                }
            )
            return resolved
        if caption is None:
            resolved.entities.append(
                {
                    "name": "vlm_error",
                    "type": "image_description",
                    "context": "VLM caption generation failed",
                }
            )
            return resolved

        if caption.strip().upper().startswith("SKIP"):
            resolved.entities.append(
                {
                    "name": "vlm_skipped",
                    "type": "image_description",
                    "context": caption,
                }
            )
            resolved.metadata["vlm_skipped"] = True
            return resolved
        resolved.entities.append(
            {
                "name": f"vlm_caption_{resolved.chunk.id}",
                "type": "image_description",
                "context": caption,
            }
        )
        resolved.metadata["vlm_enriched"] = True
        return resolved

    def _should_skip(self, image_path: str | Path) -> tuple[bool, str]:
        """Cheap pre-filter: return (skip, reason) without calling the VLM.

        Filters tiny/decorative images (logos, icons, spacers, bullets) via
        file size and pixel dimensions. These are the bulk of extraction
        noise and would otherwise each consume a slow VLM call.
        """
        p = Path(image_path)
        if not p.is_file():
            return True, "image file missing"
        size = p.stat().st_size
        if size < self._min_bytes:
            return True, f"too small ({size} bytes < {self._min_bytes})"
        try:
            with Image.open(p) as img:
                w, h = img.size
        except Exception as exc:
            logger.debug("Could not read dimensions for %s: %s — allowing VLM call", p, exc)
            return False, ""
        if w < self._min_dim or h < self._min_dim:
            return True, f"too small ({w}×{h} < {self._min_dim}px)"
        return False, ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_caption(self, image_ref: str | Path, context_hint: str = "") -> str | None:
        """Run the VLM on *image_ref* and return a text caption.

        *context_hint* is prepended to the user message (e.g. section heading,
        page number, document caption) so the VLM can evaluate the image in
        context.

        Returns ``None`` when no LLM client is configured or all
        models in the failover chain failed. Raises
        :class:`FileNotFoundError` when the image file is missing.
        """
        if self._client is None:
            logger.debug("VLMEnricher has no LLM client — skipping caption for %s", image_ref)
            return None

        image_file = Path(image_ref)
        if not image_file.is_file():
            msg = f"Image not found: {image_file}"
            raise FileNotFoundError(msg)

        raw_bytes = image_file.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        mime = _MIME_BY_EXT.get(image_file.suffix.lower(), "image/png")

        prompt = (f"[Context: {context_hint}] " if context_hint else "") + (
            "Describe this image in one sentence. If it is a logo, icon, decorative "
            "element, background graphic, or contains no substantive information, "
            "respond with just SKIP."
        )

        result = self._client.try_complete(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                },
            ],
            role=self._role,
        )
        if result is None:
            return None
        return result.content
