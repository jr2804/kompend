"""VLM-based image caption enricher for kompend Tier 3.

Analyses IMAGE chunks and adds descriptive captions as entities. Each
image is evaluated by the VLM for informativeness — logos, icons,
decorative elements, and blank images are skipped automatically.

Two rules shape this module:

- **A caption is a caption.** Only a real caption (the parser's own
  description, or what the VLM answered) carries the entity type
  ``image_description``, which is the type the compiler publishes as the
  figure caption. A failure, a pre-filter or a deliberate "SKIP" carries
  the entity type ``image_caption_status`` instead, so a status string can
  never end up as the caption of a figure in the wiki.
- **A failed unit is recorded, not swallowed.** The VLM call raises; the
  enricher records it in a :class:`~kompend.agents.failure.UnitFailureLog`
  (the shared failure vocabulary every LLM-backed stage uses) and keeps the
  batch going. The resolve stage checks the log when the batch is done.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from kompend.agents.failure import UnitFailureLog
from kompend.agents.vlm import VlmAgent, VlmDeps
from kompend.core.models import Chunk, ElementType, MetadataField, ResolvedChunk
from kompend.core.templates import get_template_loader
from kompend.core.utils.exceptions import LLMOutputError
from kompend.logging import get_logger
from kompend.preprocessing.images import VectorRasterError, rasterise_image
from kompend.resolution.base import BaseResolver

#: The entity type a real caption carries. The compiler publishes an
#: entity of this type as the caption of the figure.
CAPTION_ENTITY_TYPE = "image_description"

#: The entity type a *status* carries — filtered, skipped, or failed. It is
#: not a caption, so it never reaches the figure caption.
STATUS_ENTITY_TYPE = "image_caption_status"

#: The template that renders what surrounds a figure for the VLM.
_CONTEXT_TEMPLATE = "vlm/context.md.j2"

logger = get_logger(__name__)


class VLMEnricher(BaseResolver):
    """Enrich IMAGE chunks by producing textual captions via a VLM.

    Implements :class:`BaseResolver` so it can participate in a
    :class:`ResolverChain`. The :meth:`resolve` method wraps a bare
    :class:`Chunk` into a :class:`ResolvedChunk` before enriching.

    The context the caption is written against comes from the resolve
    stage, which owns the chunk tree: the section the figure sits in, the
    neighbouring text, and — when the contextual enricher ran — the
    LLM-written summary of the neighbourhood. It arrives as
    ``chunk.metadata[MetadataField.VLM_CONTEXT]``.
    """

    def __init__(
        self,
        agent: VlmAgent | None = None,
        min_image_bytes: int = 2048,
        min_image_dimension: int = 64,
        failures: UnitFailureLog | None = None,
    ) -> None:
        self._agent = agent
        self._min_bytes = min_image_bytes
        self._min_dim = min_image_dimension
        self._failures = failures if failures is not None else UnitFailureLog(stage="vlm")

    @property
    def failures(self) -> UnitFailureLog:
        """The failure log of this enricher — always present."""
        return self._failures

    def resolve(self, chunk: Chunk) -> ResolvedChunk:
        """BaseResolver interface: wrap chunk then enrich."""
        resolved = ResolvedChunk(chunk=chunk)
        return self.enrich(resolved)

    def enrich(self, resolved: ResolvedChunk) -> ResolvedChunk:
        """Add VLM-generated captions to IMAGE chunks. Never raises.

        A failed caption is recorded in :attr:`failures`; the batch
        continues, and the resolve stage decides whether too much of the
        batch failed to be worth continuing with.
        """
        if ElementType.IMAGE not in resolved.chunk.element_types:
            return resolved

        self.failures.attempted()

        description = resolved.chunk.metadata.get(MetadataField.DESCRIPTION)
        if description:
            resolved.entities.append(
                {
                    "name": f"parser_caption_{resolved.chunk.id}",
                    "type": CAPTION_ENTITY_TYPE,
                    "context": str(description),
                }
            )
            resolved.metadata["figure_enriched"] = True
            return resolved

        image_file = resolved.chunk.metadata.get(MetadataField.IMAGE_FILE) or resolved.metadata.get(MetadataField.IMAGE_FILE)

        if not image_file:
            self._record(resolved, FileNotFoundError("the chunk metadata carries no image path"))
            resolved.entities.append(
                {
                    "name": "vlm_placeholder",
                    "type": STATUS_ENTITY_TYPE,
                    "context": "Image present — VLM caption not available (no image path)",
                }
            )
            return resolved

        try:
            caption_source = _caption_source(image_file)
        except (FileNotFoundError, VectorRasterError) as exc:
            logger.debug("No capturable figure for %s: %s", resolved.chunk.id, exc)
            self._record(resolved, exc)
            resolved.entities.append(
                {
                    "name": "vlm_error",
                    "type": STATUS_ENTITY_TYPE,
                    "context": f"Figure unavailable: {exc}",
                }
            )
            return resolved

        skip, reason = self._should_skip(caption_source)
        if skip:
            logger.debug("VLM pre-filter skipping %s: %s", resolved.chunk.id, reason)
            resolved.entities.append(
                {
                    "name": "vlm_filtered",
                    "type": STATUS_ENTITY_TYPE,
                    "context": f"Image filtered: {reason}",
                }
            )
            resolved.metadata["vlm_filtered"] = True
            return resolved

        try:
            caption = self._generate_caption(caption_source, context=_render_context(resolved.chunk))
        except Exception as exc:
            self._record(resolved, exc)
            resolved.entities.append(
                {
                    "name": "vlm_error",
                    "type": STATUS_ENTITY_TYPE,
                    "context": f"VLM caption failed: {exc}",
                }
            )
            return resolved

        if caption.strip().upper().startswith("SKIP"):
            resolved.entities.append(
                {
                    "name": "vlm_skipped",
                    "type": STATUS_ENTITY_TYPE,
                    "context": caption,
                }
            )
            resolved.metadata["vlm_skipped"] = True
            return resolved
        resolved.entities.append(
            {
                "name": f"vlm_caption_{resolved.chunk.id}",
                "type": CAPTION_ENTITY_TYPE,
                "context": caption,
            }
        )
        resolved.metadata["vlm_enriched"] = True
        return resolved

    def _record(self, resolved: ResolvedChunk, exc: BaseException) -> None:
        """Record one failed unit in the shared failure log."""
        self.failures.record(unit=resolved.chunk.id, exc=exc)

    def _should_skip(self, image_file: str | Path) -> tuple[bool, str]:
        """Cheap pre-filter: return (skip, reason) without calling the VLM.

        Filters tiny/decorative images (logos, icons, spacers, bullets) via
        file size and pixel dimensions. These are the bulk of extraction
        noise and would otherwise each consume a slow VLM call.

        Called with the rasterized copy, so a vector figure is measured by the
        pixels the model will actually see.
        """
        p = Path(image_file)
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

    def _generate_caption(self, image_ref: str | Path, context: str = "") -> str:
        """Run the VLM on *image_ref* and return a text caption.

        Raises whatever the agent raised: the caller records the failure
        against the shared failure vocabulary, which knows whether the
        request is worth repeating.
        """
        if self._agent is None:
            msg = "no VLM model is configured for image captioning"
            raise LLMOutputError(msg)
        deps = VlmDeps(image_file=str(image_ref), context=context)
        return self._agent.run(deps)


def _render_context(chunk: Chunk) -> str:
    """Render what surrounds the figure for the caption prompt.

    The pieces come from the resolve stage (which owns the chunk tree);
    the wording that introduces them lives in the template.
    """
    raw = chunk.metadata.get(MetadataField.VLM_CONTEXT)
    parts = raw if isinstance(raw, dict) else {}
    section = str(parts.get("section") or chunk.heading_context or "")
    nearby = str(parts.get("nearby") or "")
    document_context = str(parts.get("document_context") or "")
    return get_template_loader().render(
        _CONTEXT_TEMPLATE,
        section=section,
        nearby=nearby,
        document_context=document_context,
    )


def _caption_source(image_file: str | Path) -> Path:
    """The file the VLM is handed: the figure as pixels.

    A raster is used as it is; a vector figure is rendered to a PNG beside it,
    because a vision model answers ``.svg`` with an HTTP 400.

    Raises:
        FileNotFoundError: the producer named an image that is not there — a
            broken unit, not a decorative one.
        VectorRasterError: the figure exists but cannot become pixels.
    """
    path = Path(str(image_file))
    if not path.is_file():
        msg = f"image file not found: {image_file}"
        raise FileNotFoundError(msg)
    return rasterise_image(path)
