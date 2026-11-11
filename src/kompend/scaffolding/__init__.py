"""Public API for the kompend.scaffolding package."""

from kompend.scaffolding._core import (
    AUTHORED_TEMPLATES,
    PROJECT_MARKERS,
    RAIL_FILES,
    TEMPLATE_NEXT_STEPS,
    ScaffoldError,
    ScaffoldResult,
    find_existing_project,
    scaffold_project,
)
from kompend.scaffolding.gitignore import (
    BLOCK_BEGIN,
    BLOCK_END,
    dir_name,
    is_current,
    managed_block,
    merge_gitignore,
    render_gitignore,
)

__all__ = [
    "AUTHORED_TEMPLATES",
    "BLOCK_BEGIN",
    "BLOCK_END",
    "PROJECT_MARKERS",
    "RAIL_FILES",
    "TEMPLATE_NEXT_STEPS",
    "ScaffoldError",
    "ScaffoldResult",
    "dir_name",
    "find_existing_project",
    "is_current",
    "managed_block",
    "merge_gitignore",
    "render_gitignore",
    "scaffold_project",
]
