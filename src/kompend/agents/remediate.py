"""RemediationAgent - PydanticAI agent for lint-driven concept-page generation."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Final

from pydantic_ai import UsageLimits

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT, AgentSpec, KompendAgentBase, resolve_retry_budget
from kompend.agents.types import RemediationDeps
from kompend.agents.vault_memory import build_notebook_store
from kompend.compilation.remediate_tools import build_remediation_tools
from kompend.core.config._configs import LLMConfig, RlmConfig

SPEC: Final[AgentSpec] = AgentSpec(
    name="remediation",
    result_type=str,
    deps_type=RemediationDeps,
    template_name="remediate_session_prompt.md.j2",
    system_template_name="remediate_session_system.md.j2",
    template_prefix="lint/",
    supports_delta_retry=True,
)


class RemediationAgent(KompendAgentBase[str, RemediationDeps]):
    """PydanticAI agent that generates wiki concept pages for coverage gaps.

    One session shape: the agent discovers its work queue via the
    ``fetch_gaps()`` tool, generates pages and submits them via
    ``save_page()``, and marks unresolvable gaps via ``mark_unfixable()``.
    A single-gap session degenerates to one batch — same agent, fewer
    turns.
    """

    spec: ClassVar[AgentSpec | None] = SPEC

    #: Requests one gap costs: ``fetch_gaps()``, a source read, and
    #: ``save_page()`` / ``mark_unfixable()``.
    requests_per_gap: ClassVar[int] = 3

    def _usage_limits_for(self, deps: RemediationDeps) -> UsageLimits | None:
        """Budget the one session that works through the gap queue."""
        return self._session_limits(units=deps.gaps.remaining(), per_unit=self.requests_per_gap)


def make_remediation_agent(
    *,
    model_string: str,
    timeout: float = UNSET_TIMEOUT,
    skills: dict[str, str] | None = None,
    retries: int = UNSET_RETRIES,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
    llm: LLMConfig | None = None,
    memory_dir: Path | None = None,
) -> RemediationAgent:
    """Factory for :class:`RemediationAgent` with harness capabilities.

    Injects standard session capabilities via :class:`AgenticHarness`.
    Retry budgets are split: ``retries`` governs tool-call retries;
    ``output_retries`` governs output-validation retries.

    ``memory_dir`` is the vault's agent-memory directory; the notebook
    persists under ``memory_dir/remediation/`` when ``llm.memory_store``
    is ``vault``.
    """
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    harness = AgenticHarness(
        context_window=context_window,
        memory_agent_name="remediation",
        memory=bool(llm and llm.agent_memory),
        memory_store=build_notebook_store(llm, memory_dir),
    )
    return RemediationAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        llm=llm,
        tools=build_remediation_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
