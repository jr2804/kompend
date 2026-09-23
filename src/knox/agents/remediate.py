"""RemediationAgent - PydanticAI agents for lint-driven concept-page generation."""

from __future__ import annotations

from typing import ClassVar

from pydantic_ai.agent.abstract import AgentRetries

from knox.agents.agentic_harness import AgenticHarness
from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps
from knox.agents.types import RemediationDeps
from knox.compilation.remediate_tools import build_remediation_tools
from knox.core.config._configs import RlmConfig


class RemediationAgent(KnoxAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent that generates a wiki concept page from extraction data.

    Used by tier-2 remediation (``lint/remediate_concept.md.j2``) to fill
    coverage gaps. Returns free-text Markdown — distinct from
    :class:`LintRuleAgent`, which returns structured JSON for lint checks.

    The prompt is rendered by the remediation logic and passed via
    ``deps.text``, so ``render_user_prompt`` is overridden to return it
    directly.
    """

    template_name: ClassVar[str] = ""  # not used; render_user_prompt is overridden
    system_template_name: ClassVar[str] = ""  # not used; system prompt is empty

    def render_user_prompt(self, deps: PipelineAgentDeps) -> str:  # noqa: PLR6301
        """Return the pre-rendered prompt from deps.text."""
        return deps.text


class AgenticRemediationAgent(KnoxAgentBase[str, RemediationDeps]):
    """PydanticAI agent for tool-based agentic remediation.

    Discovers coverage-gap concepts via ``fetch_gaps()``, generates wiki
    pages via ``save_page()``, and marks unresolvable gaps via
    ``mark_unfixable()``. Returns a plain-text summary.
    """

    template_name: ClassVar[str] = "remediate_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "remediate_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "lint/"
    use_delta_retry: bool = True


def make_agentic_remediation_agent(
    *,
    model_string: str,
    timeout: float = 120.0,
    skills: dict[str, str] | None = None,
    retries: int = 0,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
) -> AgenticRemediationAgent:
    """Factory for :class:`AgenticRemediationAgent` with harness capabilities.

    Injects standard agentic capabilities via :class:`AgenticHarness`.
    Retry budgets are split: ``retries`` governs tool-call retries;
    ``output_retries`` governs output-validation retries.
    """
    agent_retries: int | AgentRetries = {"tools": retries, "output": output_retries} if output_retries is not None else retries
    harness = AgenticHarness(
        context_window=context_window,
        memory_agent_name="remediation",
    )
    return AgenticRemediationAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        tools=build_remediation_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
