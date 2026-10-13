"""LLM-based file categorization for ``kompend organize``.

This module provides the building blocks for reorganising the ``sources/``
directory based on the summaries produced by a prior compilation.  The
public entry point is :func:`organize_sources` in :mod:`kompend.organize.engine`,
which ties the pieces together:

1. Load extractions from ``.kompend/cache/extract/``.
2. Ask the LLM for a hierarchical category tree.
3. Validate the tree (max depth, minimum siblings, single-file collapse).
4. Move files into the new structure.
5. Update wiki links and metadata that reference the old paths.
"""

from kompend.organize.categorize import (
    CategoryNode,
    CategoryTree,
    load_all_extractions,
    propose_categories,
    validate_tree,
)
from kompend.organize.engine import OrganizeResult, organize_sources
from kompend.organize.moves import build_link_remap, flatten_sources, materialize_tree, move_files, update_wiki_links

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
