r"""QueryAgent — PydanticAI wrapper for the wiki query/synthesis stage.

Replaces the raw litellm call in ``knox/search/retriever.py``
``TokenOverlapRetriever.synthesize()`` with a typed PydanticAI agent.
Unlike ExtractionAgent and GenerationAgent (which return structured
Pydantic models), the query agent returns **free text** — the answer
string. PydanticAI's ``output_type=str`` handles this natively.

Usage::

    from knox.agents.query import QueryAgent
    from knox.agents.deps import QueryDeps

    agent = QueryAgent(litellm_model="ollama/gpt-oss:20b")
    answer: str = agent.run(
        QueryDeps(
            question="What is VAD?",
            context="[1] Voice Activity Detection\\n    ...",
            source_id="query-1",
        )
    )
"""

from __future__ import annotations

from pydantic_ai import Agent

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import QueryDeps
from knox.agents.litellm_model import make_litellm_model


class QueryAgent(KnoxAgentBase[str, QueryDeps]):
    """PydanticAI agent for the wiki query/synthesis stage.

    The agent's system prompt comes from the existing Jinja2 template
    ``query/synthesize_system.md.j2`` so the LLM-facing surface is
    identical to the current litellm path. The user prompt is the raw
    question string. Output is free text (a string), not structured
    JSON.
    """

    # QueryAgent has no user-prompt template — the user prompt is the
    # raw question. We override render_user_prompt to return it directly.
    template_name: str = ""  # not used; render_user_prompt is overridden
    system_template_name: str = "synthesize_system.md.j2"
    result_type: type = str
    deps_type: type[QueryDeps] = QueryDeps

    def render_user_prompt(self, deps: QueryDeps) -> str:  # noqa: PLR6301
        """The user prompt is the raw question — no template needed."""
        return deps.question

    def render_system_prompt(self, deps: QueryDeps) -> str:
        """Render the system prompt from the synthesis template + skills."""
        prompt = self._template_loader.render(
            f"query/{self.system_template_name}",
            context=deps.context,
            verbosity=deps.verbosity,
        )
        if self._skills:
            skill_blocks = []
            for name, instructions in self._skills.items():
                skill_blocks.append(f"## Skill: {name}\n\n{instructions}")
            prompt += "\n\n" + "\n\n---\n\n".join(skill_blocks)
        return prompt

    def _build_agent(self) -> Agent[QueryDeps, str]:
        """Construct the PydanticAI agent with ``output_type=str``."""
        model_settings = {"timeout": self._timeout}
        if self._max_tokens is not None:
            model_settings["max_tokens"] = self._max_tokens
        return Agent(
            make_litellm_model(self._litellm_model),
            deps_type=self.deps_type,
            output_type=str,
            model_settings=model_settings,  # type: ignore[arg-type]
            retries=self._retries,
        )
