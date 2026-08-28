"""AgenticSummarizerAgent — multi-turn summarizer with tools.

One session covers all internal nodes of a chunk tree: the agent pulls
nodes via ``next_node()`` and records each summary via
``summarize_node(node_id, summary)``.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic_ai.agent.abstract import AgentRetries

from knox.agents.agentic_harness import AgenticHarness
from knox.agents.base import KnoxAgentBase
from knox.agents.types import SummarizerDeps
from knox.compilation.summarizer_tools import build_summarizer_tools
from knox.core.config._configs import RlmConfig


class AgenticSummarizerAgent(KnoxAgentBase[str, SummarizerDeps]):
    """PydanticAI agent that summarizes chunk-tree nodes via tools."""

    template_name: ClassVar[str] = "summarizer_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "summarizer_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "chunking/"


def make_agentic_summarizer_agent(
    model_string: str,
    *,
    timeout: float = 120.0,
    skills: list[str] | None = None,
    retries: int = 2,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
) -> AgenticSummarizerAgent:
    """Factory building the agentic summarizer with harness capabilities."""
    harness = AgenticHarness(context_window=context_window, memory_agent_name="summarizer")
    agent_retries: int | AgentRetries = retries
    if output_retries is not None:
        agent_retries = AgentRetries(tools=retries, output=output_retries)
    return AgenticSummarizerAgent(
        model=model_string,
        timeout=timeout,
        skills=skills,
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        tools=build_summarizer_tools(),
        extra_capabilities=harness.build_capabilities(),  # type: ignore[arg-type]
    )
