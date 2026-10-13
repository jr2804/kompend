"""Pipeline orchestrator for the KOMPEND 5-tier architecture.

Tier identifiers, tier descriptions, and :class:`PipelineConfig` live in
:mod:`kompend.core.pipeline.config` so they can be shared with the CLI and
other callers without dragging in the orchestrator. This module
re-exports the legacy ``TIER_*`` string constants for backward
compatibility with downstream callers that still import them from
``kompend.core.pipeline``.
"""

from __future__ import annotations

from kompend.core.pipeline.config import PipelineConfig, Tier, TierInfo  # noqa: F401 — re-export
from kompend.core.pipeline.stages._factory import build_pipeline  # noqa: F401 — re-export
from kompend.core.pipeline.stages._pipeline import WikiPipeline  # noqa: F401 — re-export

# Legacy string-constant re-exports (kept for backward compatibility with
# downstream callers that imported these names from ``kompend.core.pipeline``).
# New code should import :class:`Tier` from :mod:`kompend.core.pipeline.config`.
TIER_INDEX: Tier = Tier.INDEX
TIER_CHUNK: Tier = Tier.CHUNK
TIER_RESOLVE: Tier = Tier.RESOLVE
TIER_COMPILE: Tier = Tier.COMPILE
TIER_OUTPUT: Tier = Tier.OUTPUT

# Backward-compat: ordered tuple of all tiers.
ALL_TIERS: tuple[Tier, ...] = tuple(ti.tier for ti in Tier.all_tiers())

# Backward-compat alias for the previously-private description map.
_TIER_DESCRIPTIONS: dict[Tier, str] = {ti.tier: ti.description for ti in Tier.all_tiers()}
