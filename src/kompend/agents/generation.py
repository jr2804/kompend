r"""Generation agents — PydanticAI wrappers for the two-phase generation stage.



``AgenticGenerationAgent`` (built via :func:`make_agentic_generation_agent`)

is the live generation path: a tool-using agent whose ``output_type`` is

:class:`~kompend.compilation.schema.GenerationOutput`, so the LLM is asked to

produce that exact Pydantic model. PydanticAI handles JSON extraction +

validation + auto-retry on mismatch internally.

"""

from __future__ import annotations

from typing import ClassVar

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.agent.abstract import AgentRetries

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import KompendAgentBase
from kompend.agents.types import ToolGenerationDeps
from kompend.compilation.gen_tools import build_gen_tools
from kompend.compilation.schema import GenerationOutput, GenerationPage
from kompend.core.config._configs import RlmConfig
from kompend.core.text.slug import slugify
from kompend.llm.usage import UsageTracker


class AgenticGenerationAgent(KompendAgentBase[GenerationOutput, ToolGenerationDeps]):
    """PydanticAI agent for tool-based agentic generation.

    Unlike a static-prompt generation agent, this agent discovers concepts via

    ``fetch_concepts()`` and source content via ``read_segment()`` tools.
    It uses :class:`~kompend.agents.deps.ToolGenerationDeps` and the
    ``generation_agentic_*`` template pair.

    The agent is equipped with:
    - ``SlidingWindowCompaction`` — keeps history bounded
    - ``ClampOversizedMessages`` — prevents runaway tool outputs
    """

    template_name: ClassVar[str] = "generation_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "generation_agentic_system.md.j2"
    supports_delta_retry: ClassVar[bool] = True
    output_budget_key: ClassVar[str] = "gen_output_max"

    @staticmethod
    def validate_output(ctx: RunContext[ToolGenerationDeps], output: GenerationOutput) -> GenerationOutput:
        """Validate generation output for the agentic flow."""
        if not output.pages:
            raise ModelRetry("You returned an empty pages list. Generate at least one wiki page.")
        _fix_slugs_locally(output.pages)
        return output


def make_agentic_generation_agent(
    *,
    model_string: str,
    timeout: float = UNSET_TIMEOUT,
    max_tokens: int | None = None,
    skills: dict[str, str] | None = None,
    retries: int = UNSET_RETRIES,
    output_retries: int | None = None,
    delta_retry_enabled: bool | None = None,
    reasoning_effort: str | None = None,
    context_manager: str | bool | None = None,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
    usage_tracker: UsageTracker | None = None,
    llm: LLMConfig | None = None,
    budget: BudgetConfig | None = None,
) -> AgenticGenerationAgent:
    """Factory for :class:`AgenticGenerationAgent` with harness capabilities.

    Injects standard agentic capabilities via :class:`AgenticHarness`
    (``SlidingWindowCompaction`` + ``ClampOversizedMessages``; the generation
    flow runs without cross-segment memory) and registers the
    ``read_segment()`` / ``fetch_concepts()`` tools.

    Retry budgets are passed as a pydantic-ai ``AgentRetries`` dict so output
    validation and tool calls get independent ceilings. ``retries`` is the
    tool-call retry budget (default = legacy ``compile.llm.output_retries``);
    ``output_retries`` is the output-validation retry budget for agentic
    agents (default = ``compile.llm.agentic_output_retries``, higher because
    agentic turns invalidate more often on small models).
    """
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    harness = AgenticHarness(context_window=context_window, memory_agent_name="generation", memory=False)
    return AgenticGenerationAgent(
        model_string=model_string,
        timeout=timeout,
        max_tokens=max_tokens,
        skills=skills or {},
        retries=agent_retries,
        delta_retry_enabled=delta_retry_enabled,
        reasoning_effort=reasoning_effort,
        context_manager=context_manager,
        context_window=context_window,
        rlm=rlm,
        usage_tracker=usage_tracker,
        llm=llm,
        budget=budget,
        tools=build_gen_tools(),
        extra_capabilities=harness.build_capabilities(),
    )


def _fix_slugs_locally(pages: list[GenerationPage]) -> None:
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
            page.slug = slugify(page.title, default="concept")
        # De-duplicate: append numeric suffix if slug already seen.
        if page.slug in seen:
            base = page.slug
            counter = 2
            while page.slug in seen:
                page.slug = f"{base}-{counter}"
                counter += 1
        seen.add(page.slug)
