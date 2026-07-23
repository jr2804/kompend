r"""GenerationAgent — PydanticAI wrapper for the two-phase generation stage.

Replaces the raw direct LLM call in ``knox/compilation/two_phase.py``
generation stage with a typed PydanticAI agent. The agent's
``output_type`` is :class:`~knox.compilation.schema.GenerationOutput`,
so the LLM is asked to produce that exact Pydantic model. PydanticAI
handles JSON extraction + validation + auto-retry on mismatch
internally.

Usage::

    from knox.agents.generation import GenerationAgent
    from knox.agents.deps import GenerationDeps

    agent = GenerationAgent(model_string="ollama/gpt-oss:20b")
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

import re
from typing import Any

from pydantic_ai import ModelRetry, RunContext

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import GenerationDeps
from knox.compilation.schema import GenerationOutput


def _slugify_title(title: str) -> str:
    """Convert a title to a filesystem-safe slug (lowercase, hyphenated)."""
    slug = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug or "concept"


def _fix_slugs_locally(pages: list[Any]) -> None:
    """Auto-fix empty and duplicate slugs without triggering a model retry.

    Small models (Qwen3.5-2B) interpret delta-only retry prompts literally —
    when asked to "provide corrected slugs ONLY", they return only
    slug/title and **drop all content**.  Since slugs are deterministic
    (derived from the title), we fix them locally instead of asking the
    model to fix them.
    """
    seen: set[str] = set()
    for page in pages:
        if not page.slug.strip():
            page.slug = _slugify_title(page.title)
        # De-duplicate: append numeric suffix if slug already seen.
        if page.slug in seen:
            base = page.slug
            counter = 2
            while page.slug in seen:
                page.slug = f"{base}-{counter}"
                counter += 1
        seen.add(page.slug)


def _validate_generation_output(
    ctx: RunContext[GenerationDeps],
    output: GenerationOutput,
) -> GenerationOutput:
    """Semantic output validator for generation results.

    Checks for empty pages list (the only issue that warrants a model retry).
    Empty and duplicate slugs are auto-fixed locally — asking the model to
    fix them triggers content loss on small models (Qwen3.5-2B interprets
    "provide slugs ONLY" literally and drops all content).
    """
    if not output.pages:
        raise ModelRetry("You returned an empty pages list. Generate at least one wiki page with a title, slug, and content.")

    _fix_slugs_locally(output.pages)
    return output


class GenerationAgent(KnoxAgentBase[GenerationOutput, GenerationDeps]):
    """PydanticAI agent for the compilation generation stage.

    The agent's system + user prompts come from the existing Jinja2
    templates (``generation_system.md.j2``, ``generation_prompt.md.j2``)
    so the LLM-facing surface is identical to the current direct LLM
    path. Only the dispatch and validation are upgraded.
    """

    template_name: str = "generation_prompt.md.j2"
    system_template_name: str = "generation_system.md.j2"
    result_type: type[GenerationOutput] = GenerationOutput
    deps_type: type[GenerationDeps] = GenerationDeps

    #: Semantic output validator: catches empty pages, empty content,
    #: empty slugs, and duplicate slugs before they reach the pipeline.
    output_validator_func = _validate_generation_output
    use_delta_retry = True
