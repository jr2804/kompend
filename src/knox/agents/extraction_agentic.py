"""AgenticExtractionAgent — multi-turn extraction with on-demand tools.

One session covers all sections of a source: the agent pulls
sections via ``get_section()`` and records each extraction via
``save_extraction()``. Memory and compaction handle cross-section
context through a single pydantic-ai agent run.
"""

from __future__ import annotations

from typing import ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.types import ExtractionAgentDeps


class AgenticExtractionAgent(KnoxAgentBase[str, ExtractionAgentDeps]):
    """PydanticAI agent that extracts from source sections via tools."""

    template_name: ClassVar[str] = "extraction_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "extraction_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "compilation/"
    use_delta_retry: bool = True
