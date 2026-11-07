"""Two-phase compiler — extraction state, planning helpers, writer, compiler.

Public API. Import from this package root; the submodules are implementation
details.
"""

from __future__ import annotations

from kompend.compilation.two_phase.compiler import (
    Extraction,
    ExtractionSet,
    TwoPhaseCompiler,
    chars_to_tokens,
    cosine_similarity,
    merge_similar_concepts,
    segment_sources,
    tokens_to_chars,
)

__all__ = [
    "Extraction",
    "ExtractionSet",
    "TwoPhaseCompiler",
    "chars_to_tokens",
    "cosine_similarity",
    "merge_similar_concepts",
    "segment_sources",
    "tokens_to_chars",
]
