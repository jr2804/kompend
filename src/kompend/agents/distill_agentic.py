"""AgenticDistillAgent — multi-turn distillation with on-demand tools.

One session processes all tag groups: the agent pulls groups via
``get_group()`` and records pages via ``save_page()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic_ai import UsageLimits

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT, KompendAgentBase, resolve_retry_budget
from kompend.agents.types import DistillAgentDeps
from kompend.agents.vault_memory import build_notebook_store
from kompend.compilation.distill_tools import build_distill_tools
from kompend.core.config._configs import LLMConfig, RlmConfig


@dataclass
class AgenticDistillAgent(KompendAgentBase[str, DistillAgentDeps]):
    """PydanticAI agent that distills notes into wiki pages via tools."""

    template_name: ClassVar[str] = "distill_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "distill_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "capture/"

    #: LLM requests budgeted per tag group. The distiller sets this from
    #: ``budget.distill_requests_per_group`` - small local models spend prose
    #: turns and memory-tool turns before they reach ``save_page()``, and a
    #: one-group session budgeted at the historical 3 dies on its own closing
    #: turn.
    requests_per_group: int = 3

    def _usage_limits_for(
        self,
        deps: DistillAgentDeps,
        *,
        requests_per_group: int | None = None,
        headroom_frac: float = 1.1,
        session_overhead: int = 2,
    ) -> UsageLimits | None:
        """Honest request budget for one multi-group distill session.

        ``requests_per_group`` per group (a ``get_group()`` turn, a
        ``save_page()`` turn, and one turn of slack for re-reads or memory
        tools), plus ``session_overhead`` for the session's fixed costs — the
        final "no more groups" probe and the closing turn.  Without that
        constant a single-group session is budgeted at exactly its own cost
        and dies on the closing turn.

        The per-group factor defaults to the instance value (the distiller
        passes ``budget.distill_requests_per_group``); an explicit kwarg
        overrides it for probing. The shared arithmetic - overhead, retry
        capacity, headroom - lives in :meth:`KompendAgentBase._session_limits`
        and adds the resolved retry budgets: every retried tool call or
        rejected output costs another request, so a session with retries
        configured but none budgeted aborts mid-work (``request_limit of 6``
        for one group with four output retries). The pydantic-ai implicit
        default of 50 requests aborts sessions with more than ~14 groups.
        """
        per_group = self.requests_per_group if requests_per_group is None else requests_per_group
        return self._session_limits(
            units=len(deps.groups),
            per_unit=per_group,
            headroom_frac=headroom_frac,
            session_overhead=session_overhead,
        )


def make_agentic_distill_agent(
    model_string: str,
    *,
    timeout: float = UNSET_TIMEOUT,
    skills: dict[str, str] | None = None,
    retries: int = UNSET_RETRIES,
    output_retries: int | None = None,
    context_window: int | None = None,
    requests_per_group: int = 3,
    rlm: RlmConfig | None = None,
    llm: LLMConfig | None = None,
    memory_dir: Path | None = None,
) -> AgenticDistillAgent:
    """Build an :class:`AgenticDistillAgent` with tools + harness capabilities.

    Registers the ``get_group()`` / ``save_page()`` tools the system prompt
    instructs the model to call, and injects the standard agentic
    capabilities via :class:`AgenticHarness`.

    ``requests_per_group`` comes from ``budget.distill_requests_per_group``.
    ``memory_dir`` is the vault's agent-memory directory; the notebook
    persists under ``memory_dir/distill/`` when ``llm.memory_store``
    is ``vault``.
    """
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    harness = AgenticHarness(
        context_window=context_window,
        memory_agent_name="distill",
        memory=bool(llm and llm.agent_memory),
        memory_store=build_notebook_store(llm, memory_dir),
    )
    return AgenticDistillAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        requests_per_group=requests_per_group,
        rlm=rlm,
        llm=llm,
        tools=build_distill_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
