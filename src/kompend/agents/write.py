"""WriteAgent — PydanticAI agent for wiki page synthesis (kompend write)."""

from __future__ import annotations

from typing import Any, ClassVar, Final

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT, AgentSpec, KompendAgentBase, resolve_retry_budget
from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.types import WriteDeps
from kompend.compilation.write_tools import build_write_tools
from kompend.core.config._configs import LLMConfig, RlmConfig


def _write_user_prompt(agent: Any, deps: PipelineAgentDeps) -> str:
    """Return the pre-rendered prompt from deps.text."""
    return deps.text


def _agentic_write_user_prompt(agent: Any, deps: WriteDeps) -> str:
    """Return the page title as a one-line task description."""
    return f"Write the wiki page titled: {deps.page_title}"


SPEC: Final[AgentSpec] = AgentSpec(
    name="write",
    result_type=str,
    deps_type=PipelineAgentDeps,
    template_name="",
    system_template_name="",
    user_prompt=_write_user_prompt,
)


SPEC_AGENTIC: Final[AgentSpec] = AgentSpec(
    name="agentic_write",
    result_type=str,
    deps_type=WriteDeps,
    template_name="write_agentic_prompt.md.j2",
    system_template_name="write_agentic_system.md.j2",
    template_prefix="capture/",
    user_prompt=_agentic_write_user_prompt,
)


class WriteAgent(KompendAgentBase[str, PipelineAgentDeps]):
    """PydanticAI agent that synthesises source texts into a wiki page.

    Returns free-text Markdown. Used by the ``kompend write`` command to
    generate a continuous-prose wiki page from extracted source texts.
    The prompt is rendered by the caller and passed via ``deps.text``,
    so ``render_user_prompt`` is overridden to return it directly.
    """

    spec: ClassVar[AgentSpec | None] = SPEC


class AgenticWriteAgent(KompendAgentBase[str, WriteDeps]):
    """Agentic write agent — sources loaded on demand via tools."""

    spec: ClassVar[AgentSpec | None] = SPEC_AGENTIC


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
) -> AgenticWriteAgent:
    """Build an :class:`AgenticWriteAgent` with the standard agentic capabilities."""
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    harness = AgenticHarness(context_window=context_window)
    return AgenticWriteAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        rlm=rlm,
        llm=llm,
        tools=build_write_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
