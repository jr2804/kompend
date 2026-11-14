"""kompend resolution tier — entity extraction, reference resolution, image enrichment."""

from __future__ import annotations

from kompend.resolution.image_heuristics import (
    HeuristicConfig,
    HeuristicVerdict,
    ImageObservations,
    ImageRelevance,
    observe_image,
    render_visual_hints,
    verdict_from_observations,
)
from kompend.resolution.vlm_enricher import CaptionOutcome, VLMEnricher, caption_outcome

__all__ = [
    "CaptionOutcome",
    "HeuristicConfig",
    "HeuristicVerdict",
    "ImageObservations",
    "ImageRelevance",
    "VLMEnricher",
    "caption_outcome",
    "observe_image",
    "render_visual_hints",
    "verdict_from_observations",
]
