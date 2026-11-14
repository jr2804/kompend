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
from kompend.resolution.vlm_enricher import VLMEnricher

__all__ = [
    "HeuristicConfig",
    "HeuristicVerdict",
    "ImageObservations",
    "ImageRelevance",
    "VLMEnricher",
    "observe_image",
    "render_visual_hints",
    "verdict_from_observations",
]
