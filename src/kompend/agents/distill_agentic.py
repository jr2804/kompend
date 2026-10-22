"""AgenticDistillAgent — multi-turn distillation with on-demand tools.

One session processes all tag groups: the agent pulls groups via
``get_group()`` and records pages via ``save_page()``.
"""

from __future__ import annotations

import math
from typing import ClassVar

from pydantic_ai import UsageLimits

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT, KompendAgentBase, resolve_retry_budget
from kompend.agents.types import DistillAgentDeps
from kompend.compilation.distill_tools import build_distill_tools
from kompend.core.config._configs import LLMConfig, RlmConfig


class AgenticDistillAgent(KompendAgentBase[str, DistillAgentDeps]):
    """PydanticAI agent that distills notes into wiki pages via tools."""

    template_name: ClassVar[str] = "distill_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "distill_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "capture/"

    def _usage_limits_for(
        self,
        deps: DistillAgentDeps,
        *,
        requests_per_group: int = 3,
        headroom_frac: float = 1.1,
        session_overhead: int = 2,
    ) -> UsageLimits | None:
        """Honest request budget for one multi-group distill session.

        ``requests_per_group`` per group (a ``get_group()`` turn, a
        ``save_page()`` turn, and one turn of slack for re-reads or retries),
        plus ``session_overhead`` for the session's fixed costs — the final
        "no more groups" probe and the closing turn.  Without that constant a
        single-group session is budgeted at exactly its own cost and dies on
        the closing turn.

        ``headroom_frac`` then covers retries. The pydantic-ai implicit
        default of 50 requests aborts sessions with more than ~16 groups.
        """
        if self._rlm_usage_limits is not None:
            return self._rlm_usage_limits
        planned = len(deps.groups) * requests_per_group + session_overhead
        return UsageLimits(request_limit=math.ceil(planned * headroom_frac))


def make_agentic_distill_agent(
    model_string: str,
    *,
    timeout: float = UNSET_TIMEOUT,
    skills: dict[str, str] | None = None,
    retries: int = UNSET_RETRIES,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
    llm: LLMConfig | None = None,
) -> AgenticDistillAgent:
    """Build an :class:`AgenticDistillAgent` with tools + harness capabilities.

    Registers the ``get_group()`` / ``save_page()`` tools the system prompt
    instructs the model to call, and injects the standard agentic
    capabilities via :class:`AgenticHarness`.
    """
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    harness = AgenticHarness(
        context_window=context_window,
        memory_agent_name="distill",
        memory=bool(llm and llm.agent_memory),
    )
    return AgenticDistillAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        llm=llm,
        tools=build_distill_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
