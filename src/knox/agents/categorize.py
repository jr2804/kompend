"""CategorizeAgent — PydanticAI agent for hierarchical source file categorization."""

from __future__ import annotations

from knox.agents.base import KnoxAgentBase
from knox.agents.categorize_models import CategoryTree
from knox.agents.deps import PipelineAgentDeps


class CategorizeAgent(KnoxAgentBase[CategoryTree, PipelineAgentDeps]):
    """PydanticAI agent for categorizing source files into a hierarchical tree."""

    template_name: str = "categorize_prompt.md.j2"
    system_template_name: str = "categorize_system.md.j2"
    template_prefix: str = "organize/"
    result_type: type = CategoryTree
    deps_type: type[PipelineAgentDeps] = PipelineAgentDeps
    temperature: float | None = 0.0
