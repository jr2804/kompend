"""AgenticCategorizeAgent — multi-turn categorization with tools.

One session builds the whole category tree: the agent pulls file
batches via ``get_file_batch()`` and records categories via
``add_category()``.
"""

from __future__ import annotations

from typing import ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.types import CategorizeAgentDeps


class AgenticCategorizeAgent(KnoxAgentBase[str, CategorizeAgentDeps]):
    """PydanticAI agent that categorizes files into a tree via tools."""

    template_name: ClassVar[str] = "categorize_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "categorize_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "organize/"
