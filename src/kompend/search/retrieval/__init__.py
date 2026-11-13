"""Retrieval stages shared by the search entry point.

``retriever.py`` composes these; the package is the API surface for stage-level
use (tests, alternate retrievers). Stages run left to right:

``vector`` (the one store call) → ``assemble`` (hit → ``SearchResult``) →
``rank`` (store-less keyword scoring and graph expansion, used when the store
is absent or answers nothing) → ``synthesis`` (context formatting and the
LLM call).
"""

from __future__ import annotations

import lazy_loader as lazy

# Attached lazily: this package ``__init__`` runs before any submodule import,
# so eager re-exports would make ``import ...retrieval.assemble`` pull the
# synthesis stack (pydantic-ai) and defeat the stage boundary. Every stage
# module imports without the package root.
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "assemble": ["extract_snippet", "node_attributes", "node_figures", "node_tags", "result_from_hit"],
        "rank": ["TOKEN_SPLIT_RE", "graph_expansion", "keyword_search", "matches_filters", "tokenize"],
        "results": ["Citation", "SearchResult", "SynthesisResult", "hit_float", "hit_opt_str", "hit_str"],
        "synthesis": ["synthesize_results"],
        "vector": ["vector_search"],
    },
)
