"""AgenticQueryAgent — multi-turn query synthesis with tools.

One session covers a complex question: the agent decomposes it
into sub-questions, searches/wiki-reads via tools, and synthesizes
the final answer.
"""

from __future__ import annotations

from typing import ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.types import QueryAgentDeps


class AgenticQueryAgent(KnoxAgentBase[str, QueryAgentDeps]):
    """PydanticAI agent that synthesizes answers via tools."""

    template_name: ClassVar[str] = "query_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "query_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "search/"
