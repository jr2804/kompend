"""CategorizeAgent — PydanticAI agent for hierarchical source file categorization."""

from __future__ import annotations

from typing import ClassVar

from kompend.agents.base import KompendAgentBase
from kompend.agents.categorize_models import CategoryTree
from kompend.agents.deps import CategorizeDeps


class CategorizeAgent(KompendAgentBase[CategoryTree, CategorizeDeps]):
    """PydanticAI agent for categorizing source files into a hierarchical tree.

    Overrides :meth:`render_system_prompt` because ``categorize_system.md.j2``
    is the one system template that declares variables (``max_depth``,
    ``min_subcategories``, ``min_files_per_category``) — unlike every other
    system template, which is static. The values come from the
    :class:`CategorizeDeps` config fields.
    """

    template_name: ClassVar[str] = "categorize_prompt.md.j2"
    system_template_name: ClassVar[str] = "categorize_system.md.j2"
    template_prefix: ClassVar[str] = "organize/"
    temperature: float | None = 0.0

    def render_system_prompt(self, deps: CategorizeDeps) -> str:
        """Render the categorize system prompt with tree-constraint values."""
        return self._template_loader.render(
            f"{self.template_prefix}{self.system_template_name}",
            max_depth=deps.max_depth,
            min_subcategories=deps.min_subcategories,
            min_files_per_category=deps.min_files_per_category,
        )
