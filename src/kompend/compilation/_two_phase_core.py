"""Compatibility shim for the historical two-phase writer import path.

The implementation lives in :mod:`kompend.compilation.two_phase`; new code
imports the package root. This module re-exports the old names until the
remaining callers (generation, assembly, lint) are repointed by their owners.
"""

from __future__ import annotations

from kompend.compilation.concept_keys import is_structural_concept, normalize_concept_key
from kompend.compilation.two_phase.core import (
    _SLUG_MAX_LENGTH,
    _STATUS_ENTITY_NAMES,
    CONTEXT_SAFETY_FRACTION,
    CONTEXT_SAFETY_TOKENS,
    chars_to_tokens,
    compute_quality_score,
    context_safety_tokens,
    extract_images_from_chunks,
    rate_suffix,
    segment_sources,
    slugify,
    tokens_to_chars,
    try_pydanticai_extraction,
)
from kompend.compilation.two_phase.state import (
    _FAILURE_THRESHOLD_MIN,
    _FAILURE_THRESHOLD_RATIO,
    _RESERVED_CONCEPT_NAMES,
    BatchFailureTracker,
    Extraction,
    ExtractionSet,
    ProgressFn,
    TwoPhaseRuntime,
    merge_extractions,
)
from kompend.compilation.two_phase.writer import (
    _merge_node,
    _merge_sections,
    _split_sections,
    upsert_node,
)

# The historical surface: every name a caller could import from this module.
__all__ = [
    "CONTEXT_SAFETY_FRACTION",
    "CONTEXT_SAFETY_TOKENS",
    "_FAILURE_THRESHOLD_MIN",
    "_FAILURE_THRESHOLD_RATIO",
    "_RESERVED_CONCEPT_NAMES",
    "_SLUG_MAX_LENGTH",
    "_STATUS_ENTITY_NAMES",
    "BatchFailureTracker",
    "Extraction",
    "ExtractionSet",
    "ProgressFn",
    "TwoPhaseRuntime",
    "_merge_node",
    "_merge_sections",
    "_split_sections",
    "chars_to_tokens",
    "compute_quality_score",
    "context_safety_tokens",
    "extract_images_from_chunks",
    "is_structural_concept",
    "merge_extractions",
    "normalize_concept_key",
    "rate_suffix",
    "segment_sources",
    "slugify",
    "tokens_to_chars",
    "try_pydanticai_extraction",
    "upsert_node",
]
