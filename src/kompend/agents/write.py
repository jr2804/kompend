"""WriteAgent - PydanticAI agent for wiki page synthesis (kompend write)."""

from __future__ import annotations

from typing import Any, ClassVar, Final

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT, AgentSpec, KompendAgentBase, resolve_retry_budget
from kompend.agents.types import WriteDeps
from kompend.compilation.write_tools import build_write_tools
from kompend.core.config._configs import LLMConfig, RlmConfig


def _write_user_prompt(agent: Any, deps: WriteDeps) -> str:
    """Return the page title as a one-line task description."""
    return f"Write the wiki page titled: {deps.page_title}"


SPEC: Final[AgentSpec] = AgentSpec(
    name="write",
    result_type=str,
    deps_type=WriteDeps,
    template_name="",
    system_template_name="write_session_system.md.j2",
    template_prefix="capture/",
    user_prompt=_write_user_prompt,
)


class WriteAgent(KompendAgentBase[str, WriteDeps]):
    """PydanticAI agent that synthesises source texts into a wiki page.

    One session shape: the sources travel in the deps and the agent pulls
    them on demand via the ``read_source`` / ``list_sources`` tools,
    paging through windows so nothing is dropped. A run that never calls
    a tool is still a valid session — the tools are there when the model
    wants them.
    """

    spec: ClassVar[AgentSpec | None] = SPEC


def make_write_agent(
    model_string: str,
    *,
    timeout: float = UNSET_TIMEOUT,
    skills: dict[str, str] | None = None,
    retries: int = UNSET_RETRIES,
    output_retries: int | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
    llm: LLMConfig | None = None,
) -> WriteAgent:
    """Build a :class:`WriteAgent` with the standard session capabilities."""
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    harness = AgenticHarness(context_window=context_window)
    return WriteAgent(
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
