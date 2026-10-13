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

__all__ = [
    "AUTHORED_TEMPLATES",
    "PROJECT_MARKERS",
    "RAIL_FILES",
    "TEMPLATE_NEXT_STEPS",
    "ScaffoldError",
    "ScaffoldResult",
    "find_existing_project",
    "scaffold_project",
]
