r"""QueryAgent — PydanticAI wrapper for the wiki query/synthesis stage.

Replaces the raw direct LLM call in ``knox/search/retriever.py``
``TokenOverlapRetriever.synthesize()`` with a typed PydanticAI agent.
Unlike ExtractionAgent and GenerationAgent (which return structured
Pydantic models), the query agent returns **free text** — the answer
string. PydanticAI's ``output_type=str`` handles this natively.

Usage::

    from knox.agents.query import QueryAgent
    from knox.agents.deps import QueryDeps

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

from typing import ClassVar

from knox.agents.base import RLM_CONTEXT_PLACEHOLDER, KnoxAgentBase
from knox.agents.deps import QueryDeps


class QueryAgent(KnoxAgentBase[str, QueryDeps]):
    """PydanticAI agent for the wiki query/synthesis stage.

    The agent's system prompt comes from the existing Jinja2 template
    ``query/synthesize_system.md.j2`` so the LLM-facing surface is
    identical to the current pydantic-ai path. The user prompt is the raw
    question string. Output is free text (a string), not structured
    JSON.
    """

    # QueryAgent has no user-prompt template — the user prompt is the
    # raw question. We override render_user_prompt to return it directly.
    template_name: ClassVar[str] = ""  # not used; render_user_prompt is overridden
    system_template_name: ClassVar[str] = "synthesize_system.md.j2"

    def render_user_prompt(self, deps: QueryDeps) -> str:  # noqa: PLR6301
        """The user prompt is the raw question — no template needed."""
        return deps.question

    def render_system_prompt(self, deps: QueryDeps) -> str:
        """Render the system prompt from the synthesis template + skills.

        When RLM is enabled the inlined context is replaced with a
        placeholder — the full context stays in the deps (REPL ``context``
        variable) so the model explores it via code instead.
        """
        ctx = deps.context
        if self.rlm:
            ctx = RLM_CONTEXT_PLACEHOLDER
        prompt = self._template_loader.render(
            f"query/{self.system_template_name}",
            context=ctx,
            verbosity=deps.verbosity,
        )
        if self.skills:
            skill_blocks = []
            for name, instructions in self.skills.items():
                skill_blocks.append(f"## Skill: {name}\n\n{instructions}")
            prompt += "\n\n" + "\n\n---\n\n".join(skill_blocks)
        return prompt
