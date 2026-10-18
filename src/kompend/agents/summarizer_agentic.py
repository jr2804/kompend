"""AgenticSummarizerAgent — multi-turn summarizer with tools.

One session covers all internal nodes of a chunk tree: the agent pulls
nodes via ``next_node()`` and records each summary via
``summarize_node(node_id, summary)``.
"""

from __future__ import annotations

import math
from typing import ClassVar

from pydantic_ai import UsageLimits
from pydantic_ai.agent.abstract import AgentRetries

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import KompendAgentBase
from kompend.agents.types import SummarizerDeps
from kompend.compilation.summarizer_tools import build_summarizer_tools
from kompend.core.config._configs import RlmConfig


class AgenticSummarizerAgent(KompendAgentBase[str, SummarizerDeps]):
    """PydanticAI agent that summarizes chunk-tree nodes via tools."""

    template_name: ClassVar[str] = "summarizer_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "summarizer_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "chunking/"

    def _usage_limits_for(
        self,
        deps: SummarizerDeps,
        *,
        requests_per_node: int = 2,
        headroom_frac: float = 1.1,
    ) -> UsageLimits | None:
        """Honest request budget for one tree session.

        ``requests_per_node`` × node count is the base budget (one
        ``next_node()`` turn plus one ``summarize_node()`` turn), scaled by
        ``headroom_frac`` (ceil, so small trees get at least one spare
        request) for tool/output retries. The pydantic-ai implicit default
        of 50 requests aborts trees with more than ~24 eligible nodes.
        """
        if self._rlm_usage_limits is not None:
            return self._rlm_usage_limits
        return UsageLimits(request_limit=math.ceil(len(deps.nodes) * requests_per_node * headroom_frac))


def make_agentic_summarizer_agent(
    model_string: str,
    *,
    timeout: float = 120.0,
    skills: dict[str, str] | None = None,
    retries: int = 2,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
) -> AgenticSummarizerAgent:
    """Build an :class:`AgenticSummarizerAgent` with tools + harness capabilities.

    Registers the ``next_node()`` / ``summarize_node()`` tools the system prompt
    instructs the model to call, and injects the standard agentic capabilities
    via :class:`AgenticHarness`.
    """
    agent_retries: int | AgentRetries = {"tools": retries, "output": output_retries} if output_retries is not None else retries
    harness = AgenticHarness(context_window=context_window, memory_agent_name="summarizer")
    return AgenticSummarizerAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        tools=build_summarizer_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
