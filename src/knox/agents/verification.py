"""VerificationAgent — PydanticAI agents for claim/citation verification."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic_ai.agent.abstract import AgentRetries

from knox.agents.agentic_harness import AgenticHarness
from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps
from knox.agents.types import VerificationDeps
from knox.compilation.verification_tools import build_verification_tools
from knox.core.config._configs import RlmConfig


class VerificationAgent(KnoxAgentBase[dict[str, Any], PipelineAgentDeps]):
    """PydanticAI agent for verifying claims against source excerpts."""

    template_name: ClassVar[str] = "verify_claim.md.j2"
    system_template_name: ClassVar[str] = "verify_system.md.j2"
    template_prefix: ClassVar[str] = "verification/"


class AgenticVerificationAgent(KnoxAgentBase[str, VerificationDeps]):
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
    use_delta_retry: bool = True


def make_agentic_verification_agent(
    *,
    model_string: str,
    timeout: float = 120.0,
    skills: dict[str, str] | None = None,
    retries: int = 0,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
) -> AgenticVerificationAgent:
    """Factory for :class:`AgenticVerificationAgent`.

    Uses :class:`AgenticHarness` for Memory + SlidingWindowCompaction +
    ClampOversizedMessages. Retry budgets are split: ``retries`` governs
    tool-call retries; ``output_retries`` governs output-validation.
    """
    agent_retries: int | AgentRetries = {"tools": retries, "output": output_retries} if output_retries is not None else retries
    harness = AgenticHarness(context_window=context_window, memory_agent_name="verification")
    return AgenticVerificationAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        tools=build_verification_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
