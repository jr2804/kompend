"""CategorizeAgent — PydanticAI agent for hierarchical source file categorization."""

from __future__ import annotations

from typing import Any, ClassVar, Final

from kompend.agents.base import AgentSpec, KompendAgentBase
from kompend.agents.categorize_models import CategoryTree
from kompend.agents.deps import CategorizeDeps


def _categorize_system_prompt(agent: Any, deps: CategorizeDeps) -> str:
    """Render the categorize system prompt with tree-constraint values."""
    return agent._template_loader.render(  # noqa: SLF001
        f"{agent.template_prefix}{agent.system_template_name}",
        max_depth=deps.max_depth,
        min_subcategories=deps.min_subcategories,
        min_files_per_category=deps.min_files_per_category,
    )


SPEC: Final[AgentSpec] = AgentSpec(
    name="categorize",
    result_type=CategoryTree,
    deps_type=CategorizeDeps,
    template_name="categorize_prompt.md.j2",
    system_template_name="categorize_system.md.j2",
    template_prefix="organize/",
    temperature=0.0,
    system_prompt=_categorize_system_prompt,
)


class CategorizeAgent(KompendAgentBase[CategoryTree, CategorizeDeps]):
    """PydanticAI agent for categorizing source files into a hierarchical tree.

    Overrides :meth:`render_system_prompt` because ``categorize_system.md.j2``
    is the one system template that declares variables (``max_depth``,
    ``min_subcategories``, ``min_files_per_category``) — unlike every other
    system template, which is static. The values come from the
    :class:`CategorizeDeps` config fields.
    """

    spec: ClassVar[AgentSpec | None] = SPEC
