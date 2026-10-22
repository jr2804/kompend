"""VerificationAgent — PydanticAI agents for claim/citation verification."""

from __future__ import annotations

from typing import Any, ClassVar

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT, KompendAgentBase, resolve_retry_budget
from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.types import VerificationDeps
from kompend.compilation.verification_tools import build_verification_tools
from kompend.core.config._configs import BudgetConfig, LLMConfig, RlmConfig


class VerificationAgent(KompendAgentBase[dict[str, Any], PipelineAgentDeps]):
    """PydanticAI agent for verifying claims against source excerpts."""

    template_name: ClassVar[str] = "verify_claim.md.j2"
    system_template_name: ClassVar[str] = "verify_system.md.j2"
    output_budget_key: ClassVar[str] = "verify_output_max"
    template_prefix: ClassVar[str] = "verification/"


class AgenticVerificationAgent(KompendAgentBase[str, VerificationDeps]):
    """Agentic claim-verification agent.

    Discovers claims via ``get_claim()``, searches sources via
    ``search_sources()`` / ``read_page()``, and records verdicts via
    ``mark_verified()`` / ``mark_contradicted()`` / ``mark_unverifiable()``.
    Runs a single multi-turn session over all claims instead of one
    LLM call per claim.
    """

    template_name: ClassVar[str] = "verify_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "verify_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "lint/"
    supports_delta_retry: ClassVar[bool] = True
    output_budget_key: ClassVar[str] = "verify_output_max"


def make_agentic_verification_agent(
    *,
    model_string: str,
    timeout: float = UNSET_TIMEOUT,
    skills: dict[str, str] | None = None,
    retries: int = UNSET_RETRIES,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
    llm: LLMConfig | None = None,
    budget: BudgetConfig | None = None,
) -> AgenticVerificationAgent:
    """Factory for :class:`AgenticVerificationAgent`.

    Uses :class:`AgenticHarness` for Memory + SlidingWindowCompaction +
    ClampOversizedMessages. Retry budgets are split: ``retries`` governs
    tool-call retries; ``output_retries`` governs output-validation.
    """
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    harness = AgenticHarness(
        context_window=context_window,
        memory_agent_name="verification",
        memory=bool(llm and llm.agent_memory),
    )
    return AgenticVerificationAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        llm=llm,
        budget=budget,
        tools=build_verification_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
