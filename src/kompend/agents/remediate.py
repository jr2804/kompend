"""RemediationAgent - PydanticAI agents for lint-driven concept-page generation."""

from __future__ import annotations

from typing import ClassVar

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT, KompendAgentBase, resolve_retry_budget
from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.types import RemediationDeps
from kompend.compilation.remediate_tools import build_remediation_tools
from kompend.core.config._configs import LLMConfig, RlmConfig


class RemediationAgent(KompendAgentBase[str, PipelineAgentDeps]):
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


class AgenticRemediationAgent(KompendAgentBase[str, RemediationDeps]):
    """PydanticAI agent for tool-based agentic remediation.

    Discovers coverage-gap concepts via ``fetch_gaps()``, generates wiki
    pages via ``save_page()``, and marks unresolvable gaps via
    ``mark_unfixable()``. Returns a plain-text summary.
    """

    template_name: ClassVar[str] = "remediate_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "remediate_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "lint/"
    supports_delta_retry: ClassVar[bool] = True


def make_agentic_remediation_agent(
    *,
    model_string: str,
    timeout: float = UNSET_TIMEOUT,
    skills: dict[str, str] | None = None,
    retries: int = UNSET_RETRIES,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
    llm: LLMConfig | None = None,
) -> AgenticRemediationAgent:
    """Factory for :class:`AgenticRemediationAgent` with harness capabilities.

    Injects standard agentic capabilities via :class:`AgenticHarness`.
    Retry budgets are split: ``retries`` governs tool-call retries;
    ``output_retries`` governs output-validation retries.
    """
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    harness = AgenticHarness(
        context_window=context_window,
        memory_agent_name="remediation",
        memory=bool(llm and llm.agent_memory),
    )
    return AgenticRemediationAgent(
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
