"""AgenticDistillAgent — multi-turn distillation with on-demand tools.

One session processes all tag groups: the agent pulls groups via
``get_group()`` and records pages via ``save_page()``.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic_ai.agent.abstract import AgentRetries

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import KompendAgentBase
from kompend.agents.types import DistillAgentDeps
from kompend.compilation.distill_tools import build_distill_tools
from kompend.core.config._configs import RlmConfig


class AgenticDistillAgent(KompendAgentBase[str, DistillAgentDeps]):
    """PydanticAI agent that distills notes into wiki pages via tools."""

    template_name: ClassVar[str] = "distill_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "distill_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "capture/"


def make_agentic_distill_agent(
    model_string: str,
    *,
    timeout: float = 180.0,
    skills: dict[str, str] | None = None,
    retries: int = 2,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
) -> AgenticDistillAgent:
    """Build an :class:`AgenticDistillAgent` with tools + harness capabilities.

    Registers the ``get_group()`` / ``save_page()`` tools the system prompt
    instructs the model to call, and injects the standard agentic
    capabilities via :class:`AgenticHarness`.
    """
    agent_retries: int | AgentRetries = {"tools": retries, "output": output_retries} if output_retries is not None else retries
    harness = AgenticHarness(context_window=context_window, memory_agent_name="distill")
    return AgenticDistillAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        tools=build_distill_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
