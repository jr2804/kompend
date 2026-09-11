"""AgenticDistillAgent — multi-turn distillation with on-demand tools.

One session processes all tag groups: the agent pulls groups via
``get_group()`` and records pages via ``save_page()``.
"""

from __future__ import annotations

from typing import ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.types import DistillAgentDeps


class AgenticDistillAgent(KnoxAgentBase[str, DistillAgentDeps]):
    """PydanticAI agent that distills notes into wiki pages via tools."""

    template_name: ClassVar[str] = "distill_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "distill_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "capture/"
