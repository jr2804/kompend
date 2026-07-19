"""Pipeline orchestrator for the KNOX 5-tier architecture.

Tier identifiers, tier descriptions, and :class:`PipelineConfig` live in
:mod:`knox.core.pipeline.config` so they can be shared with the CLI and
other callers without dragging in the orchestrator. This module
re-exports the legacy ``TIER_*`` string constants for backward
compatibility with downstream callers that still import them from
``knox.core.pipeline``.
"""

from __future__ import annotations

from knox.core.pipeline.config import ALL_TIERS, TIER_DESCRIPTIONS, PipelineConfig, Tier  # noqa: F401 — re-export
from knox.core.pipeline.stages._factory import build_pipeline  # noqa: F401 — re-export
from knox.core.pipeline.stages._pipeline import WikiPipeline  # noqa: F401 — re-export

# Legacy string-constant re-exports (kept for backward compatibility with
# downstream callers that imported these names from ``knox.core.pipeline``).
# New code should import :class:`Tier` from :mod:`knox.core.pipeline.config`.
TIER_INDEX: Tier = Tier.INDEX
TIER_CHUNK: Tier = Tier.CHUNK
TIER_RESOLVE: Tier = Tier.RESOLVE
TIER_COMPILE: Tier = Tier.COMPILE
TIER_OUTPUT: Tier = Tier.OUTPUT

# Backward-compat alias for the previously-private description map.
_TIER_DESCRIPTIONS = TIER_DESCRIPTIONS
