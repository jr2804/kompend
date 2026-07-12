r"""GenerationAgent — PydanticAI wrapper for the two-phase generation stage.

Replaces the raw litellm call in ``knox/compilation/two_phase.py``
generation stage with a typed PydanticAI agent. The agent's
``output_type`` is :class:`~knox.compilation.schema.GenerationOutput`,
so the LLM is asked to produce that exact Pydantic model. PydanticAI
handles JSON extraction + validation + auto-retry on mismatch
internally.

Usage::

    from knox.agents.generation import GenerationAgent
    from knox.agents.deps import GenerationDeps

    agent = GenerationAgent(litellm_model="ollama/gpt-oss:20b")
    result = agent.run(GenerationDeps(
        concepts_json="[...]",
        content="### src\\n...",
        source_id="batch-1",
        pinned_facts=[],
    ))
    for page in result.pages:
        print(page.title, page.content)
"""

from __future__ import annotations

from pydantic_ai import Agent

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import GenerationDeps
from knox.compilation.schema import GenerationOutput


class GenerationAgent(KnoxAgentBase[GenerationOutput, GenerationDeps]):
    """PydanticAI agent for the compilation generation stage.

    The agent's system + user prompts come from the existing Jinja2
    templates (``generation_system.md.j2``, ``generation_prompt.md.j2``)
    so the LLM-facing surface is identical to the current litellm
    path. Only the dispatch and validation are upgraded.
    """

    template_name: str = "generation_prompt.md.j2"
    system_template_name: str = "generation_system.md.j2"
    result_type: type[GenerationOutput] = GenerationOutput
    deps_type: type[GenerationDeps] = GenerationDeps

    def _build_agent(self) -> Agent[GenerationDeps, GenerationOutput]:
        """Construct the PydanticAI agent.

        The system prompt is rendered per-call in
        :meth:`~knox.agents.base.KnoxAgentBase.run` and passed as
        ``instructions=`` to ``run_sync``. This keeps construction cheap
        and avoids template-resolution failures at init time.
        """
        return Agent(
            self._pydanticai_model,
            deps_type=self.deps_type,
            output_type=self.result_type,
            model_settings={"timeout": self._timeout},  # type: ignore[arg-type]
        )
