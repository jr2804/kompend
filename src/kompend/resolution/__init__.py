"""kompend resolution tier — entity extraction, reference resolution, image enrichment."""

from __future__ import annotations

from kompend.resolution.image_heuristics import HeuristicConfig, HeuristicVerdict, ImageRelevance
from kompend.resolution.vlm_enricher import VLMEnricher

__all__ = [
    "HeuristicConfig",
    "HeuristicVerdict",
    "ImageRelevance",
    "VLMEnricher",
]
