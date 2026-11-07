"""Names the pre-package ``two_phase`` module re-exported from its neighbours.

Kept in a submodule so the package root stays lazy: ``assembly`` imports the
compatibility shim, which imports this package.
"""

from __future__ import annotations

from kompend.compilation.flows.assembly import merge_similar_concepts
from kompend.core.text.matching import cosine as cosine_similarity

__all__ = ["cosine_similarity", "merge_similar_concepts"]
