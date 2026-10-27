"""WikiIndexAgent — PydanticAI agents for wiki index/overview generation."""

from __future__ import annotations

from typing import Any, ClassVar

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT, KompendAgentBase, resolve_retry_budget
from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.types import OverviewDeps
from kompend.compilation.overview_tools import build_overview_tools
from kompend.core.config._configs import LLMConfig, RlmConfig


class WikiIndexAgent(KompendAgentBase[dict[str, Any], PipelineAgentDeps]):
    """PydanticAI agent for generating wiki overview/index text."""

    template_prefix: ClassVar[str] = "output/"
    template_name: ClassVar[str] = "synthesize.md.j2"
    system_template_name: ClassVar[str] = "synthesize.md.j2"


class AgenticOverviewAgent(KompendAgentBase[str, OverviewDeps]):
    """Agentic overview-synthesis agent.

    Discovers pages via ``list_pages()``, reads full content via
    ``read_page()``, and submits the final overview via
    ``save_overview()``. Returns the overview Markdown (also collected
    on ``deps.overview`` by the tool).
    """

    template_name: ClassVar[str] = "synthesize_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "synthesize_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "output/"
    supports_delta_retry: ClassVar[bool] = True


def make_agentic_overview_agent(
    *,
    model_string: str,
    timeout: float = UNSET_TIMEOUT,
    skills: dict[str, str] | None = None,
    retries: int = UNSET_RETRIES,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
    llm: LLMConfig | None = None,
) -> AgenticOverviewAgent:
    """Factory for :class:`AgenticOverviewAgent` with harness capabilities.

    Uses :class:`AgenticHarness` for Memory + SlidingWindowCompaction +
    ClampOversizedMessages. Retry budgets are split: ``retries`` governs
    tool-call retries; ``output_retries`` governs output-validation.
    """
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    harness = AgenticHarness(context_window=context_window)
    return AgenticOverviewAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        llm=llm,
        tools=build_overview_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
