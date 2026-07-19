"""Two-phase compilation flow modules.

- :mod:`assembly` — wiki assembly, edge building, concept merging (pure)
- :mod:`extraction` — LLM extraction from source documents
- :mod:`generation` — per-concept page generation and wiki assembly
"""

from knox.compilation.flows.assembly import (
    attach_tags_to_nodes,
    build_edges,
    build_tree_texts,
    merge_similar_concepts,
)

__all__ = [
    "attach_tags_to_nodes",
    "build_edges",
    "build_tree_texts",
    "merge_similar_concepts",
]
