"""AgenticExtractionAgent — multi-turn extraction with on-demand tools.

One session covers all sections of a source: the agent pulls
sections via ``get_section()`` and records each extraction via
``save_extraction()``. Memory and compaction handle cross-section
context through a single pydantic-ai agent run.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic_ai.agent.abstract import AgentRetries

from knox.agents.agentic_harness import AgenticHarness
from knox.agents.base import KnoxAgentBase
from knox.agents.types import ExtractionAgentDeps
from knox.compilation.extraction_tools import build_extraction_tools
from knox.core.config._configs import RlmConfig


class AgenticExtractionAgent(KnoxAgentBase[str, ExtractionAgentDeps]):
    """PydanticAI agent that extracts from source sections via tools."""

    template_name: ClassVar[str] = "extraction_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "extraction_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "compilation/"
    use_delta_retry: bool = True


def make_agentic_extraction_agent(
    model_string: str,
    *,
    timeout: float = 120.0,
    skills: list[str] | None = None,
    retries: int = 2,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
) -> AgenticExtractionAgent:
    """Factory building the agentic extractor with harness capabilities."""
    harness = AgenticHarness(context_window=context_window, memory_agent_name="extraction")
    agent_retries: int | AgentRetries = retries
    if output_retries is not None:
        agent_retries = AgentRetries(tools=retries, output=output_retries)
    return AgenticExtractionAgent(
        model=model_string,
        timeout=timeout,
        skills=skills,
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        tools=build_extraction_tools(),
        extra_capabilities=harness.build_capabilities(),  # type: ignore[arg-type]
    )
