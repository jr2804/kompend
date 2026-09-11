"""AgenticSummarizerAgent — multi-turn summarizer with tools.

One session covers all internal nodes of a chunk tree: the agent pulls
nodes via ``next_node()`` and records each summary via
``summarize_node(node_id, summary)``.
"""

from __future__ import annotations

from typing import ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.types import SummarizerDeps


class AgenticSummarizerAgent(KnoxAgentBase[str, SummarizerDeps]):
    """PydanticAI agent that summarizes chunk-tree nodes via tools."""

    template_name: ClassVar[str] = "summarizer_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "summarizer_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "chunking/"
