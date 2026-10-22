"""WriteAgent — PydanticAI agent for wiki page synthesis (kompend write)."""

from __future__ import annotations

from typing import ClassVar

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT, KompendAgentBase, resolve_retry_budget
from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.types import WriteDeps
from kompend.compilation.write_tools import build_write_tools
from kompend.core.config._configs import BudgetConfig, LLMConfig, RlmConfig


class WriteAgent(KompendAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent that synthesises source texts into a wiki page.

    Returns free-text Markdown. Used by the ``kompend write`` command to
    generate a continuous-prose wiki page from extracted source texts.
    The prompt is rendered by the caller and passed via ``deps.text``,
    so ``render_user_prompt`` is overridden to return it directly.
    """

    template_name: ClassVar[str] = ""  # not used; render_user_prompt is overridden
    system_template_name: ClassVar[str] = ""  # no system prompt needed

    def render_user_prompt(self, deps: PipelineAgentDeps) -> str:  # noqa: PLR6301
        """Return the pre-rendered prompt from deps.text."""
        return deps.text


class AgenticWriteAgent(KompendAgentBase[str, WriteDeps]):
    """Agentic write agent — sources loaded on demand via tools."""

    template_name: ClassVar[str] = "write_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "write_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "capture/"

    def render_user_prompt(self, deps: WriteDeps) -> str:  # noqa: PLR6301
        """Return the page title as a one-line task description."""
        return f"Write the wiki page titled: {deps.page_title}"


def make_agentic_write_agent(
    model_string: str,
    *,
    timeout: float = UNSET_TIMEOUT,
    skills: dict[str, str] | None = None,
    retries: int = UNSET_RETRIES,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
    llm: LLMConfig | None = None,
    budget: BudgetConfig | None = None,
) -> AgenticWriteAgent:
    """Build an :class:`AgenticWriteAgent` with the standard agentic capabilities."""
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    harness = AgenticHarness(context_window=context_window, memory_agent_name="write")
    return AgenticWriteAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        llm=llm,
        budget=budget,
        tools=build_write_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
