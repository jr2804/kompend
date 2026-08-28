"""AgenticCategorizeAgent — multi-turn categorization with tools.

One session builds the whole category tree: the agent pulls file
batches via ``get_file_batch()`` and records categories via
``add_category()``.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic_ai.agent.abstract import AgentRetries

from knox.agents.agentic_harness import AgenticHarness
from knox.agents.base import KnoxAgentBase
from knox.agents.types import CategorizeAgentDeps
from knox.compilation.categorize_tools import build_categorize_tools
from knox.core.config._configs import RlmConfig


class AgenticCategorizeAgent(KnoxAgentBase[str, CategorizeAgentDeps]):
    """PydanticAI agent that categorizes files into a tree via tools."""

    template_name: ClassVar[str] = "categorize_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "categorize_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "organize/"


def make_agentic_categorize_agent(
    model_string: str,
    *,
    timeout: float = 120.0,
    skills: list[str] | None = None,
    retries: int = 2,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
) -> AgenticCategorizeAgent:
    """Factory building the agentic categorizer with harness capabilities."""
    harness = AgenticHarness(context_window=context_window, memory_agent_name="categorize")
    agent_retries: int | AgentRetries = retries
    if output_retries is not None:
        agent_retries = AgentRetries(tools=retries, output=output_retries)
    return AgenticCategorizeAgent(
        model=model_string,
        timeout=timeout,
        skills=skills,
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        tools=build_categorize_tools(),
        extra_capabilities=harness.build_capabilities(),  # type: ignore[arg-type]
    )
