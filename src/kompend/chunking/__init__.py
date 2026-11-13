"""kompend chunking tier."""

from __future__ import annotations

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "extractors": ["extract_structured"],
        "hierarchical": ["HierarchicalChunker"],
        "outline": [
            "OutlineNode",
            "build_outline",
            "estimate_tokens",
            "outline_to_prompt",
        ],
        "summarizer": ["NodeSummarizer", "SummarizerSessionPolicy"],
    },
)
