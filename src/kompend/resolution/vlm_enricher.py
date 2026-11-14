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
- **A relevance verdict is not a caption.** When a decision model is
  configured, a cheap text-context classification runs before the caption
  call: the surroundings the caption would be written against (section,
  neighbouring text, contextual summary) decide whether the figure is
  worth captioning — the image itself is never sent. A ``decorative``
  verdict skips the VLM call and is recorded under its own status name
  (``vlm_relevance_skipped``), distinct from the VLM's own SKIP. A failed
  classification is recorded in a separate ``vlm:relevance`` log and falls
  back to captioning — never to a skip.
- **Cheap evidence first.** When the heuristic filter is enabled, a bounded
  structural/pixel pass runs before the text gate: a thin line-like canvas,
  a near-uniform fill, or a near-blank frame is dropped with a specific
  reason in its ``image_caption_status``. Transparency and weak signals stay
  ``UNSURE`` and continue to the gate and the VLM. The pass never decodes
  more than the configured analysis budget, and a failed analysis is
  ``UNSURE`` — it can never suppress a figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image

from kompend.agents.decision import FigureRelevanceAgent, FigureRelevanceDeps
from kompend.agents.failure import UnitFailureLog
from kompend.agents.vlm import VlmAgent, VlmDeps
from kompend.core.models import Chunk, ElementType, MetadataField, ResolvedChunk
from kompend.core.templates import get_template_loader
from kompend.core.types import ReasonCode, StageOutcome
from kompend.core.utils.exceptions import LLMOutputError
from kompend.llm.decision import FigureRelevanceVerdict
from kompend.logging import get_logger
from kompend.preprocessing.images import MAX_RASTER_PIXELS, VectorRasterError, is_raster_image, rasterise_image
from kompend.resolution.base import BaseResolver
from kompend.resolution.image_heuristics import HeuristicConfig, HeuristicVerdict, ImageRelevance, assess_image

#: The entity type a real caption carries. The compiler publishes an
#: entity of this type as the caption of the figure.
CAPTION_ENTITY_TYPE = "image_description"

#: The entity type a *status* carries — filtered, skipped, or failed. It is
#: not a caption, so it never reaches the figure caption.
STATUS_ENTITY_TYPE = "image_caption_status"

#: The template that renders what surrounds a figure for the VLM.
_CONTEXT_TEMPLATE = "vlm/context.md.j2"

#: Backstop for vector sources. The renderer budgets its output at
#: ``MAX_RASTER_PIXELS`` (4 Mpx, i.e. 16 MiB of raw RGBA), so a vector file
#: bigger than that frame carries more XML than its own rendered pixels —
#: pathological by construction. SVG geometry cannot serve instead: the
#: declared canvas says nothing about the bytes the renderer will need.
_MAX_VECTOR_BYTES: Final[int] = 4 * MAX_RASTER_PIXELS

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CaptionOutcome:
    """What the enricher did to one figure, shaped for a stage-ledger row.

    ``outcome`` is the row's ``StageOutcome``; ``reason_code`` explains a
    non-``ok`` outcome in the shared :class:`ReasonCode` vocabulary;
    ``detail`` carries the enricher's own reason string.
    """

    outcome: StageOutcome
    reason_code: ReasonCode | None
    detail: str


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
        relevance_agent: FigureRelevanceAgent | None = None,
        relevance_failures: UnitFailureLog | None = None,
        heuristics: HeuristicConfig | None = None,
    ) -> None:
        self._agent = agent
        self._min_bytes = min_image_bytes
        self._min_dim = min_image_dimension
        self._failures = failures if failures is not None else UnitFailureLog(stage="vlm")
        self._relevance_agent = relevance_agent
        self._relevance_failures = relevance_failures if relevance_failures is not None else UnitFailureLog(stage="vlm:relevance")
        self._heuristics = heuristics

    @property
    def failures(self) -> UnitFailureLog:
        """The failure log of this enricher — always present."""
        return self._failures

    @property
    def relevance_failures(self) -> UnitFailureLog:
        """Failed relevance classifications (stage ``vlm:relevance``).

        Only failures land here — a failed classification falls back to
        captioning, so this log never counts attempts and never stops the
        batch. It exists so a failed classification is a recorded unit,
        never a silent skip.
        """
        return self._relevance_failures

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

        oversized = _oversized_reason(image_file) if image_file else None
        if not image_file or oversized is not None:
            if oversized is not None:
                logger.info("Skipping oversized figure %s for %s: %s", Path(str(image_file)).name, resolved.chunk.id, oversized)
                resolved.entities.append(
                    {
                        "name": "vlm_filtered",
                        "type": STATUS_ENTITY_TYPE,
                        "context": f"Image too large: {oversized}",
                    }
                )
                resolved.metadata["vlm_filtered"] = True
            else:
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
        heuristic = None if skip else self._heuristic_verdict(resolved.chunk, caption_source)
        skip_name: str | None = None
        skip_context = ""
        if skip:
            logger.debug("VLM pre-filter skipping %s: %s", resolved.chunk.id, reason)
            skip_name, skip_context = "vlm_filtered", f"Image filtered: {reason}"
        elif heuristic is not None:
            logger.debug("Image heuristics skipping %s: %s", resolved.chunk.id, heuristic.reason)
            skip_name, skip_context = "vlm_filtered", f"Image filtered: {heuristic.reason}"
        elif self._relevance_agent is not None and not self._is_worth_captioning(resolved.chunk.id, _render_context(resolved.chunk)):
            logger.debug("Relevance gate skipping %s: judged decorative", resolved.chunk.id)
            skip_name, skip_context = "vlm_relevance_skipped", "Figure judged decorative — caption deliberately skipped"
        if skip_name is not None:
            resolved.entities.append(
                {
                    "name": skip_name,
                    "type": STATUS_ENTITY_TYPE,
                    "context": skip_context,
                }
            )
            resolved.metadata[skip_name] = True
            if heuristic is not None:
                resolved.metadata["vlm_heuristic"] = heuristic.decision.value
                resolved.metadata["vlm_heuristic_reason"] = heuristic.reason
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

    def _is_worth_captioning(self, unit: str, context: str) -> bool:
        """Run the relevance gate: ``True`` captions, ``False`` skips.

        The judgement is made from the surrounding text alone — the figure
        image itself is never sent, so the decision sidecar's text-only
        transport applies unchanged.

        A failed classification is recorded in the ``vlm:relevance`` log and
        reads as ``True`` — the figure is captioned as if the gate were not
        there. A failed classification must never skip a figure.
        """
        agent = self._relevance_agent
        if agent is None:
            return True
        try:
            verdict = agent.run(FigureRelevanceDeps(context=context)).verdict
        except Exception as exc:
            self._relevance_failures.record(unit=unit, exc=exc)
            logger.debug("Relevance classification failed for %s (%s) — falling back to captioning", unit, exc)
            return True
        return verdict is FigureRelevanceVerdict.INFORMATIVE

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
        except Image.DecompressionBombError as exc:
            return True, f"dimensions trip PIL's pixel ceiling ({exc})"
        except Exception as exc:
            logger.debug("Could not read dimensions for %s: %s — allowing VLM call", p, exc)
            return False, ""
        if w < self._min_dim or h < self._min_dim:
            return True, f"too small ({w}×{h} < {self._min_dim}px)"
        return False, ""

    def _heuristic_verdict(self, chunk: Chunk, image_file: str | Path) -> HeuristicVerdict | None:
        """A confident structural/pixel drop, or ``None`` to fall through.

        The resolve stage already measured every raster figure to write the
        prompt hints; when that measurement produced a confident drop, it
        travels in the figure context and is reused here. A direct
        :meth:`enrich` call (no resolve stage) measures on demand.

        Only a verdict at or above the configured confidence drops the
        figure. ``UNSURE`` and sub-threshold drops are indistinguishable
        from no verdict here, so they continue to the text gate and the VLM.
        """
        if self._heuristics is None or not self._heuristics.enabled:
            return None
        precomputed = _precomputed_verdict(chunk)
        if precomputed is not None and precomputed.confidence >= self._heuristics.min_confidence:
            return precomputed
        verdict = assess_image(image_file, self._heuristics)
        if verdict.decision is ImageRelevance.DROP and verdict.confidence >= self._heuristics.min_confidence:
            return verdict
        return None

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


def caption_outcome(resolved: ResolvedChunk) -> CaptionOutcome | None:
    """Classify what the enricher did to one image chunk.

    ``None`` when the enricher did not act on the chunk (a text chunk
    passing through the chain). The classification reads the same metadata
    flags and status entities :meth:`VLMEnricher.enrich` writes, so a
    cached resolution classifies identically to a fresh one — the resolve
    stage can therefore record one honest ledger row per figure either way.
    """
    metadata = resolved.metadata
    status = next((e for e in resolved.entities if e.get("type") == STATUS_ENTITY_TYPE), None)
    context = str(status.get("context") or "") if status else ""
    if metadata.get("vlm_enriched"):
        return CaptionOutcome(StageOutcome.OK, None, "")
    if metadata.get("vlm_heuristic"):
        reason = str(metadata.get("vlm_heuristic_reason") or context)
        return CaptionOutcome(StageOutcome.SKIPPED, ReasonCode.VLM_FILTERED, reason)
    if metadata.get("vlm_filtered"):
        return CaptionOutcome(StageOutcome.SKIPPED, ReasonCode.VLM_FILTERED, context)
    if metadata.get("vlm_relevance_skipped"):
        return CaptionOutcome(StageOutcome.SKIPPED, ReasonCode.VLM_RELEVANCE_SKIPPED, context)
    if metadata.get("vlm_skipped"):
        return CaptionOutcome(StageOutcome.SKIPPED, ReasonCode.VLM_SKIPPED, context)
    if status is not None and status.get("name") == "vlm_placeholder":
        return CaptionOutcome(StageOutcome.DEGRADED, ReasonCode.NO_PARSER, context)
    if status is not None and status.get("name") == "vlm_error":
        return CaptionOutcome(StageOutcome.FAILED, ReasonCode.BACKEND_UNAVAILABLE, context)
    return None


def _precomputed_verdict(chunk: Chunk) -> HeuristicVerdict | None:
    """The drop the resolve stage measured into the figure context, if any."""
    raw = chunk.metadata.get(MetadataField.VLM_CONTEXT)
    if not isinstance(raw, dict):
        return None
    if raw.get("heuristic_decision") != ImageRelevance.DROP.value:
        return None
    try:
        confidence = float(str(raw.get("heuristic_confidence") or "0"))
    except ValueError:
        return None
    return HeuristicVerdict(
        ImageRelevance.DROP,
        str(raw.get("heuristic_reason") or "pre-filter drop"),
        confidence,
        str(raw.get("heuristic_stage") or "pixels"),
    )


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
    visual_hints = str(parts.get("visual_hints") or "")
    return get_template_loader().render(
        _CONTEXT_TEMPLATE,
        section=section,
        nearby=nearby,
        document_context=document_context,
        visual_hints=visual_hints,
    )


def _oversized_reason(image_file: str | Path) -> str | None:
    """Why this figure can never reach the VLM, or ``None`` when it may proceed.

    Raster inputs are measured by header dimensions against PIL's own
    decompression-bomb ceiling — opening a header is cheap, decoding is not.
    Vector inputs are measured by file size: their declared canvas says
    nothing about render cost, and the renderer already budgets its output,
    so this is a backstop for pathological attachments, not a classifier.
    Missing or unreadable files return ``None``: that verdict belongs to
    :func:`_caption_source`, not to this pre-check.
    """
    path = Path(str(image_file))
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if not is_raster_image(path):
        if size > _MAX_VECTOR_BYTES:
            return f"vector source too large ({size} bytes > {_MAX_VECTOR_BYTES})"
        return None
    try:
        with Image.open(path) as img:
            width, height = img.size
    except Image.DecompressionBombError:
        return f"dimensions trip PIL's {Image.MAX_IMAGE_PIXELS}-pixel ceiling"
    except Exception:
        return None
    ceiling = Image.MAX_IMAGE_PIXELS
    if ceiling is not None and width * height >= ceiling:
        return f"{width}×{height} trips PIL's {ceiling}-pixel ceiling"
    return None


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
