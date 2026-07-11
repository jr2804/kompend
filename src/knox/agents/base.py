"""KnoxAgent base class — KNOX-side wrapper around pydantic_ai.Agent.

This is the only piece in the ``knox/agents/`` package that knows about
KNOX config, litellm model resolution, and the existing template loader.
Every concrete agent (extraction, generation, query, wicer) inherits
from it and supplies:

- ``template_name`` — the Jinja2 template under
  ``templates/compilation/<name>.md.j2`` whose frontmatter declares
  the system prompt role and (optional) JSON output schema
- ``result_type`` — the Pydantic model the LLM must return
- Optional ``deps_type`` (default: :class:`PipelineAgentDeps`)

The base class handles the boring-but-load-bearing concerns:

- **Model resolution** — accept the litellm model format that the
  rest of KNOX uses (``provider/model``) and convert it to
  PydanticAI's format (``provider:model``).
- **Template rendering** — load the system + user prompt templates
  via the existing ``TemplateLoader`` and render with the agent's
  ``Deps`` instance.
- **LLM call** — call ``agent.run_sync(prompt, deps=...)`` and return
  the typed result. On failure, raise :class:`LLMOutputError` so the
  pipeline's failure tracker treats the agent like the litellm path
  (3 attempts with budget bump, global hard-fail threshold).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent

from knox.agents.deps import PipelineAgentDeps
from knox.compilation.schema import LLMOutputError
from knox.core.template_loader import get_template_loader
from knox.logging import get_logger

logger = get_logger(__name__)

#: Per-stage timeout floor. Agents in KNOX are synchronous single-turn;
#: PydanticAI's run_sync is fast but provider-side latency can spike.
_DEFAULT_TIMEOUT_SECONDS: float = 120.0


def _litellm_to_pydanticai_model(litellm_model: str) -> str:
    """Convert a litellm-format model string to pydantic-ai format.

    Litellm uses ``<provider>/<model>`` (slash separator). PydanticAI
    uses ``<provider>:<model>`` (colon separator). The slash appears
    inside the model name for some providers (e.g. HuggingFace repos)
    so we only replace the FIRST slash.

    Examples:
        ``ollama/gpt-oss:20b``       -> ``ollama:gpt-oss:20b``
        ``openai/gpt-4o``            -> ``openai:gpt-4o``
        ``anthropic/claude-3-5-sonnet`` -> ``anthropic:claude-3-5-sonnet``
    """
    if "/" not in litellm_model:
        return litellm_model
    provider, _, model = litellm_model.partition("/")
    return f"{provider}:{model}"


ResultT = TypeVar("ResultT", bound=BaseModel)
DepsT = TypeVar("DepsT", bound=PipelineAgentDeps)


class KnoxAgentBase[ResultT: BaseModel, DepsT: PipelineAgentDeps](ABC):
    """KNOX-side wrapper around a PydanticAI agent.

    Subclasses MUST set:

    - ``template_name`` — the filename under ``templates/compilation/``,
      e.g. ``"extraction_prompt.md"``.
    - ``system_template_name`` — the filename of the system prompt
      template under ``templates/compilation/``, e.g.
      ``"extraction_system.md"``.
    - ``result_type`` — a Pydantic model class that the LLM output
      will be validated against.

    Subclasses MAY override :meth:`build_agent` to inject tools,
    capabilities, or custom model settings.
    """

    template_name: ClassVar[str]
    system_template_name: ClassVar[str]
    result_type: ClassVar[type[BaseModel]]
    deps_type: ClassVar[type[PipelineAgentDeps]] = PipelineAgentDeps

    def __init__(self, litellm_model: str, *, timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> None:
        self._litellm_model = litellm_model
        self._timeout = timeout
        self._pydanticai_model = _litellm_to_pydanticai_model(litellm_model)
        # The PydanticAI Agent is constructed lazily (on first run() call)
        # to avoid provider-resolution failures at init time. Tests can
        # construct agents without a live provider.
        self._agent: Agent[DepsT, ResultT] | None = None
        # The template loader must be available for render_system_prompt
        # and render_user_prompt, which are called at run() time.
        self._template_loader = get_template_loader()

    @abstractmethod
    def _build_agent(self) -> Agent[DepsT, ResultT]:
        """Construct the wrapped PydanticAI agent. Called once at init."""

    def render_user_prompt(self, deps: DepsT) -> str:
        """Render the user-prompt template with the agent's deps.

        The template receives the deps' fields as Jinja2 variables
        (e.g. ``{{ text }}``, ``{{ source_id }}``). Subclasses can
        override to pass extra variables.
        """
        return self._template_loader.render(
            f"compilation/{self.template_name}",
            **self._deps_to_template_vars(deps),
        )

    def render_system_prompt(self, deps: DepsT) -> str:
        """Render the system-prompt template. Defaults to static text."""
        return self._template_loader.render(
            f"compilation/{self.system_template_name}",
            **self._deps_to_template_vars(deps),
        )

    @staticmethod
    def _deps_to_template_vars(deps: DepsT) -> dict[str, Any]:
        """Convert a Deps instance to a dict for Jinja2 rendering.

        Excludes the system/dataclass metadata that Jinja2 cannot
        consume. Subclasses that hold extra context should override
        this method.
        """
        if hasattr(deps, "__dict__"):
            return {k: v for k, v in deps.__dict__.items() if not k.startswith("_")}
        return {}

    def run(self, deps: DepsT) -> ResultT:
        """Single-turn agent invocation. Returns the typed result.

        The underlying PydanticAI Agent is constructed lazily on the first
        call, so tests can create agents without a live provider.

        The system prompt is rendered per-call from :meth:`render_system_prompt`
        and passed as ``instructions=`` to PydanticAI's ``run_sync``. This keeps
        agent construction cheap (no template resolution at init time) and
        allows future agents to vary the system prompt per ``Deps``.

        Raises:
            LLMOutputError: if the LLM call fails or returns
                structurally invalid output. The pipeline's
                ``_BatchFailureTracker`` and retry loop treat this the
                same as a failure in the litellm path.
        """
        if self._agent is None:
            self._agent = self._build_agent()
        user_prompt = self.render_user_prompt(deps)
        system_prompt = self.render_system_prompt(deps)
        try:
            result = self._agent.run_sync(
                user_prompt,
                deps=deps,
                instructions=system_prompt,
            )
        except Exception as exc:
            logger.warning(
                "PydanticAI agent '%s' failed on '%s': %s",
                type(self).__name__,
                getattr(deps, "source_id", "?"),
                exc,
            )
            raise LLMOutputError(
                f"PydanticAI agent failed for model '{self._litellm_model}': {exc}"
            ) from exc
        return result.output

    @property
    def litellm_model(self) -> str:
        """The original KNOX-format model string."""
        return self._litellm_model

    @property
    def pydanticai_model(self) -> str:
        """The pydantic-ai-format model string actually used."""
        return self._pydanticai_model
