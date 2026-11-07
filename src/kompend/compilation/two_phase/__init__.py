"""Two-phase compiler — extraction state, planning helpers, writer, compiler.

Public API. Import from this package root; the submodules are implementation
details. Everything is attached lazily: ``compiler`` imports the flows, and a
flow imports the compatibility shim, which imports this package.
"""

from __future__ import annotations

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "state": [
            "BatchFailureTracker",
            "Extraction",
            "ExtractionSet",
            "ProgressFn",
            "TwoPhaseRuntime",
            "merge_extractions",
        ],
        "core": [
            "CONTEXT_SAFETY_FRACTION",
            "CONTEXT_SAFETY_TOKENS",
            "chars_to_tokens",
            "compute_quality_score",
            "context_safety_tokens",
            "extract_images_from_chunks",
            "rate_suffix",
            "segment_sources",
            "slugify",
            "tokens_to_chars",
            "try_pydanticai_extraction",
        ],
        "writer": ["upsert_node"],
        "compiler": ["TwoPhaseCompiler"],
        "_compat": ["cosine_similarity", "merge_similar_concepts"],
    },
)
