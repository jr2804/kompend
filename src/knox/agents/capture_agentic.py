"""AgenticCaptureAgent — multi-turn extraction with on-demand tools.

One session processes all input texts: the agent pulls texts via
``get_input()`` and records items via ``save_item()``.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic_ai.agent.abstract import AgentRetries

from knox.agents.agentic_harness import AgenticHarness
from knox.agents.base import KnoxAgentBase
from knox.agents.types import CaptureAgentDeps
from knox.compilation.capture_tools import build_capture_tools
from knox.core.config._configs import RlmConfig


class AgenticCaptureAgent(KnoxAgentBase[str, CaptureAgentDeps]):
    """PydanticAI agent that captures knowledge items from multiple texts via tools."""

    template_name: ClassVar[str] = "capture_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "capture_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "capture/"


def make_agentic_capture_agent(
    model_string: str,
    *,
    timeout: float = 120.0,
    skills: dict[str, str] | None = None,
    retries: int = 2,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
) -> AgenticCaptureAgent:
    """Factory building the agentic capture agent with harness capabilities."""
    harness = AgenticHarness(context_window=context_window, memory_agent_name="capture")
    agent_retries: int | AgentRetries = retries
    if output_retries is not None:
        agent_retries = AgentRetries(tools=retries, output=output_retries)
    return AgenticCaptureAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        tools=build_capture_tools(),
        extra_capabilities=harness.build_capabilities(),  # type: ignore[arg-type]
    )
