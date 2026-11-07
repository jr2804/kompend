"""Generation flow package for the two-phase compiler.

``flow`` owns the orchestration and the pass loop; the sibling modules own
segment planning, request accounting, prompt assembly and page writing.
"""

from __future__ import annotations

from kompend.compilation.flows.generation.budget import (
    _agentic_budget,
    _known_titles_windows,
)
from kompend.compilation.flows.generation.flow import (
    PlaceholderMarker,
    PlaceholderPageError,
    _allowed_kind,
    _consult_loop2,
    _format_uncovered_concepts,
    _placeholder_reasons,
    _reject_placeholder_pages,
    _resolve_page_kind,
    _sanitize_entity_type,
    _SegmentUnit,
    assemble_wiki,
    gen_cache_deps,
    generate_all_pages_agentic,
    generation_detail,
    learn_wikilink_aliases,
    prepare_concepts,
    prepare_source_segments,
    run_generation,
    run_generation_resumable,
)
from kompend.compilation.flows.generation.prompt import (
    _categories_block,
    build_known_titles,
)

__all__ = [
    "PlaceholderMarker",
    "PlaceholderPageError",
    "_SegmentUnit",
    "_agentic_budget",
    "_allowed_kind",
    "_categories_block",
    "_consult_loop2",
    "_format_uncovered_concepts",
    "_known_titles_windows",
    "_placeholder_reasons",
    "_reject_placeholder_pages",
    "_resolve_page_kind",
    "_sanitize_entity_type",
    "assemble_wiki",
    "build_known_titles",
    "gen_cache_deps",
    "generate_all_pages_agentic",
    "generation_detail",
    "learn_wikilink_aliases",
    "prepare_concepts",
    "prepare_source_segments",
    "run_generation",
    "run_generation_resumable",
]
