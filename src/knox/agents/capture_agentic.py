"""AgenticCaptureAgent — multi-turn extraction with on-demand tools.

One session processes all input texts: the agent pulls texts via
``get_input()`` and records items via ``save_item()``.
"""

from __future__ import annotations

from typing import ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.types import CaptureAgentDeps


class AgenticCaptureAgent(KnoxAgentBase[str, CaptureAgentDeps]):
    """PydanticAI agent that captures knowledge items from multiple texts via tools."""

    template_name: ClassVar[str] = "capture_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "capture_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "capture/"
