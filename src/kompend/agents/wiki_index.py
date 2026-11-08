"""WikiIndexAgent — PydanticAI agents for wiki index/overview generation."""

from __future__ import annotations

from typing import Any, ClassVar, Final

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT, AgentSpec, KompendAgentBase, resolve_retry_budget
from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.types import OverviewDeps
from kompend.compilation.overview_tools import build_overview_tools
from kompend.core.config._configs import LLMConfig, RlmConfig

SPEC_WIKI_INDEX: Final[AgentSpec] = AgentSpec(
    name="wiki_index",
    result_type=dict[str, Any],
    deps_type=PipelineAgentDeps,
    template_name="synthesize.md.j2",
    system_template_name="synthesize.md.j2",
    template_prefix="output/",
)


class WikiIndexAgent(KompendAgentBase[dict[str, Any], PipelineAgentDeps]):
    """PydanticAI agent for generating wiki overview/index text."""

    spec: ClassVar[AgentSpec | None] = SPEC_WIKI_INDEX


SPEC_AGENTIC_OVERVIEW: Final[AgentSpec] = AgentSpec(
    name="agentic_overview",
    result_type=str,
    deps_type=OverviewDeps,
    template_name="synthesize_agentic_prompt.md.j2",
    system_template_name="synthesize_agentic_system.md.j2",
    template_prefix="output/",
    supports_delta_retry=True,
)


class AgenticOverviewAgent(KompendAgentBase[str, OverviewDeps]):
    """Agentic overview-synthesis agent.

    Discovers pages via ``list_pages()``, reads full content via
    ``read_page()``, and submits the final overview via
    ``save_overview()``. Returns the overview Markdown (also collected
    on ``deps.overview`` by the tool).
    """

    spec: ClassVar[AgentSpec | None] = SPEC_AGENTIC_OVERVIEW


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
