"""KnoxAgent base class — KNOX-side wrapper around pydantic_ai.Agent.

Every concrete agent (extraction, generation, query, wicer) inherits
from this base and supplies:

- ``template_name`` — the Jinja2 template under
  ``templates/<dir>/<name>.md.j2``
- ``system_template_name`` — the system prompt template name
- ``result_type`` — the Pydantic model (or ``str``) the LLM must return
- ``deps_type`` — the typed context dataclass

The base class handles:

- **Model bridge** — creates a :class:`~pydantic_ai.models.function.FunctionModel`
  backed by ``litellm.completion()``, preserving litellm's full provider
  dispatch chain (custom providers, API key resolution, monkey-patches).
- **Template rendering** — loads system + user prompt templates via the
  existing ``TemplateLoader`` and renders with the agent's ``Deps``.
- **Skill injection** — appends user-defined skill instructions to the
  system prompt.
- **Error mapping** — wraps failures as :class:`LLMOutputError` so the
  pipeline's failure tracker handles them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

from pydantic_ai import Agent

from knox.agents.deps import PipelineAgentDeps
from knox.core.exception_format import LLMOutputError
from knox.core.template_loader import get_template_loader
from knox.logging import get_logger

logger = get_logger(__name__)

#: Per-stage timeout floor. Agents in KNOX are synchronous single-turn;
#: PydanticAI's run_sync is fast but provider-side latency can spike.
_DEFAULT_TIMEOUT_SECONDS: float = 120.0

ResultT = TypeVar("ResultT")


@runtime_checkable
class AgentDeps(Protocol):
    """Structural type for deps objects passed to :class:`KnoxAgentBase`.

    Any dataclass with a ``source_id: str`` and ``timeout: float`` field
    satisfies this protocol.
    """

    source_id: str
    timeout: float


DepsT = TypeVar("DepsT", bound=AgentDeps)


class KnoxAgentBase[ResultT, DepsT: AgentDeps](ABC):
    """KNOX-side wrapper around a PydanticAI agent.

    Subclasses MUST set:

    - ``template_name`` — the filename under ``templates/compilation/``
      (or ``templates/query/`` for the query agent).
    - ``system_template_name`` — the filename of the system prompt.
    - ``result_type`` — a Pydantic model class (or ``str`` for free-text
      output) that the LLM output will be validated against.

    Subclasses MUST implement :meth:`_build_agent`.
    """

    template_name: ClassVar[str]
    system_template_name: ClassVar[str]
    result_type: ClassVar[type]
    deps_type: ClassVar[type[AgentDeps]] = PipelineAgentDeps

    def __init__(
        self,
        litellm_model: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        retries: int = 2,
        max_tokens: int | None = None,
        skills: dict[str, str] | None = None,
    ) -> None:
        self._litellm_model = litellm_model
        self._timeout = timeout
        self._retries = retries
        self._max_tokens = max_tokens
        # User-defined skills (skill_name -> instructions, injected into system prompt)
        self._skills: dict[str, str] = skills or {}
        # The PydanticAI Agent is constructed lazily (on first run() call)
        # so tests can construct agents without a live provider.
        self._agent: Agent[DepsT, ResultT] | None = None
        # The template loader is available for render_system_prompt
        # and render_user_prompt, which are called at run() time.
        self._template_loader = get_template_loader()

    def render_user_prompt(self, deps: DepsT) -> str:
        """Render the user-prompt template with the agent's deps.

        Only variables declared in the template's frontmatter are passed,
        avoiding warnings about unused variables.
        """
        template_path = f"compilation/{self.template_name}"
        meta = self._template_loader.get_meta(template_path)
        declared = meta.declared_variable_names()
        vars_dict = self._deps_to_template_vars(deps)
        filtered = {k: v for k, v in vars_dict.items() if k in declared}
        return self._template_loader.render(template_path, **filtered)

    def render_system_prompt(self, deps: DepsT) -> str:
        """Render the system-prompt template, appending any skill instructions.

        System templates are static instructions — deps variables are not passed
        since no system template uses them.
        """
        prompt = self._template_loader.render(
            f"compilation/{self.system_template_name}",
        )
        if self._skills:
            skill_blocks = []
            for name, instructions in self._skills.items():
                skill_blocks.append(f"## Skill: {name}\n\n{instructions}")
            prompt += "\n\n" + "\n\n---\n\n".join(skill_blocks)
        return prompt

    def run(self, deps: DepsT) -> ResultT:
        """Single-turn agent invocation. Returns the typed result.

        Raises:
            LLMOutputError: if the LLM call fails or returns
                structurally invalid output.
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
                "Agent '%s' failed on '%s': %s",
                type(self).__name__,
                getattr(deps, "source_id", "?"),
                exc,
            )
            raise LLMOutputError(f"Agent failed for model '{self._litellm_model}': {exc}") from exc
        return result.output

    @property
    def litellm_model(self) -> str:
        """The KNOX-format (litellm) model string."""
        return self._litellm_model

    @abstractmethod
    def _build_agent(self) -> Agent[DepsT, ResultT]:
        """Construct the wrapped PydanticAI agent. Called lazily on first run()."""

    @staticmethod
    def _deps_to_template_vars(deps: DepsT) -> dict[str, Any]:
        """Convert a Deps instance to a dict for Jinja2 rendering."""
        if hasattr(deps, "__dict__"):
            return {k: v for k, v in deps.__dict__.items() if not k.startswith("_")}
        return {}
