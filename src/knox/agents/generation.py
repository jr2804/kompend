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
    result = agent.run(
        GenerationDeps(
            concepts_json="[...]",
            content="### src\\n...",
            source_id="batch-1",
            pinned_facts=[],
        )
    )
    for page in result.pages:
        print(page.title, page.content)
"""

from __future__ import annotations

from pydantic_ai import ModelRetry, RunContext

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import GenerationDeps
from knox.compilation.schema import GenerationOutput


def _validate_generation_output(
    ctx: RunContext[GenerationDeps],
    output: GenerationOutput,
) -> GenerationOutput:
    """Semantic output validator for generation results.

    Checks for common issues that Pydantic structural validation can't
    catch: empty pages list, pages with empty content, empty slugs,
    and duplicate slugs.  Raises ``ModelRetry`` with actionable
    feedback so the model can fix the problem.
    """
    if not output.pages:
        raise ModelRetry(
            "You returned an empty pages list. Generate at least one "
            "wiki page with a title, slug, and content."
        )

    seen_slugs: set[str] = set()
    issues: list[str] = []

    for i, page in enumerate(output.pages):
        if not page.content.strip():
            issues.append(f"page {i} ('{page.title}') has empty content")
        if not page.slug.strip():
            issues.append(f"page {i} ('{page.title}') has an empty slug")
        if page.slug in seen_slugs:
            issues.append(f"duplicate slug '{page.slug}' on page {i} ('{page.title}')")
        seen_slugs.add(page.slug)

    if issues:
        raise ModelRetry(
            "Fix the following issues and return the complete output again:\n"
            + "\n".join(f"- {issue}" for issue in issues)
        )

    return output


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

    #: Semantic output validator: catches empty pages, empty content,
    #: empty slugs, and duplicate slugs before they reach the pipeline.
    output_validator_func = _validate_generation_output
