"""LLM-based file categorization for ``knox organize``.

This module provides the building blocks for reorganising the ``sources/``
directory based on the summaries produced by a prior compilation.  The
public entry point is :func:`organize_sources` in :mod:`knox.organize.engine`,
which ties the pieces together:

1. Load extractions from ``.knox/cache/extract/``.
2. Ask the LLM for a hierarchical category tree.
3. Validate the tree (max depth, minimum siblings, single-file collapse).
4. Move files into the new structure.
5. Update wiki links and metadata that reference the old paths.
"""

from knox.organize.categorize import (
    CategoryNode,
    CategoryTree,
    load_all_extractions,
    propose_categories,
    validate_tree,
)
from knox.organize.engine import OrganizeResult, organize_sources
from knox.organize.links import build_link_remap, update_wiki_links
from knox.organize.move import flatten_sources, materialize_tree, move_files

__all__ = [
    "CategoryNode",
    "CategoryTree",
    "OrganizeResult",
    "build_link_remap",
    "flatten_sources",
    "load_all_extractions",
    "materialize_tree",
    "move_files",
    "organize_sources",
    "propose_categories",
    "update_wiki_links",
    "validate_tree",
]
