r"""QueryAgent — PydanticAI wrapper for the wiki query/synthesis stage.

Replaces the raw direct LLM call in ``kompend/search/retriever.py``
``TokenOverlapRetriever.synthesize()`` with a typed PydanticAI agent.
Unlike ExtractionAgent (which returns structured
Pydantic models), the query agent returns **free text** — the answer
string. PydanticAI's ``output_type=str`` handles this natively.

Usage::

    from kompend.agents.query import QueryAgent
    from kompend.agents.deps import QueryDeps

    agent = QueryAgent(model_string="ollama/gpt-oss:20b")
    answer: str = agent.run(
        QueryDeps(
            question="What is VAD?",
            context="[1] Voice Activity Detection\\n    ...",
            source_id="query-1",
        )
    )
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

from kompend.agents.base import RLM_CONTEXT_PLACEHOLDER, AgentSpec, KompendAgentBase
from kompend.agents.deps import QueryDeps


def _query_user_prompt(agent: Any, deps: QueryDeps) -> str:
    """The user prompt is the raw question — no template needed."""
    return deps.question


def _query_system_prompt(agent: Any, deps: QueryDeps) -> str:
    """Render the system prompt from the synthesis template + skills."""
    ctx = deps.context
    if agent.rlm:
        ctx = RLM_CONTEXT_PLACEHOLDER
    prompt = agent._template_loader.render(  # noqa: SLF001
        f"{agent.template_prefix}{agent.system_template_name}",
        context=ctx,
        verbosity=deps.verbosity,
    )
    if agent.skills:
        skill_blocks = [f"## Skill: {name}\n\n{instructions}" for name, instructions in agent.skills.items()]
        prompt += "\n\n" + "\n\n---\n\n".join(skill_blocks)
    return prompt


SPEC: Final[AgentSpec] = AgentSpec(
    name="query",
    result_type=str,
    deps_type=QueryDeps,
    template_name="",
    system_template_name="synthesize_system.md.j2",
    template_prefix="query/",
    user_prompt=_query_user_prompt,
    system_prompt=_query_system_prompt,
)


class QueryAgent(KompendAgentBase[str, QueryDeps]):
    """PydanticAI agent for the wiki query/synthesis stage.

    The agent's system prompt comes from the existing Jinja2 template
    ``query/synthesize_system.md.j2`` so the LLM-facing surface is
    identical to the current pydantic-ai path. The user prompt is the raw
    question string. Output is free text (a string), not structured
    JSON.
    """

    # QueryAgent has no user-prompt template — the user prompt is the
    # raw question. We override render_user_prompt to return it directly.
    spec: ClassVar[AgentSpec | None] = SPEC
