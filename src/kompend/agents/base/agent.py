"""KompendAgent base class — kompend-side wrapper around pydantic_ai.Agent.

Every concrete agent (extraction, generation, query, wicer) inherits
from this base and supplies:

- ``template_name`` — the Jinja2 template under
  ``templates/<dir>/<name>.md.j2``
- ``system_template_name`` — the system prompt template name
- Generic type params ``[ResultT, DepsT]`` — the Pydantic model (or ``str``)
  the LLM must return, and the typed context dataclass.

``result_type`` and ``deps_type`` are derived automatically from the
generic type parameters — subclasses do NOT set them as class attributes.

The base class handles:

- **Model bridge** — creates a :class:`KompendLiteLLMModel`
  backed by a native pydantic-ai model via ``build_model()``
  dispatch chain (custom providers, API key resolution, monkey-patches).
- **Template rendering** — loads system + user prompt templates via the
  existing ``TemplateLoader`` and renders with the agent's ``Deps``.
- **Skill injection** — appends user-defined skill instructions to the
  system prompt.
- **Error mapping** — wraps failures as :class:`LLMOutputError` so the
  pipeline's failure tracker handles them uniformly.
"""

from __future__ import annotations

import dataclasses
import re
import typing
from abc import ABC
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, cast

from pydantic_ai import Agent, NativeOutput, RunContext, Tool, UsageLimits
from pydantic_ai.agent import AgentModelSettings
from pydantic_ai.agent.abstract import AgentRetries
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.output import OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness.compaction import ContextUsage, ReportContextUsage

from kompend.agents.base.context import (
    on_usage_update,
    resolve_context_capability,
    resolve_rlm_capability,
)
from kompend.agents.base.delta_retry import (
    make_delta_retry_hook,
)
from kompend.agents.base.deps import (
    _DEFAULT_RETRIES,
    UNSET_RETRIES,
    UNSET_TIMEOUT,
    AgentDeps,
    DepsT,
    ResultT,
    _coerce_type,
)
from kompend.agents.base.models import (
    build_top_level_model,
    ceiling_model_settings,
    estimate_prompt_tokens,
    model_max_output,
    output_ceiling,
)
from kompend.agents.base.retries import (
    failure,
    invoke_agent,
    retry_requests,
    session_limits,
    transient_backoff,
    transient_retries,
    usage_limits_for,
)
from kompend.agents.base.spec import AgentSpec, install_hooks
from kompend.agents.instrument import configure_observability
from kompend.agents.rlm import (
    RLMCapability,
)
from kompend.agents.types import PipelineAgentDeps
from kompend.core.config._configs import ContextManagerPreset, LLMConfig, LlmDataFormat, ReasoningEffort, RlmConfig
from kompend.core.templates import TemplateLoader, get_template_loader
from kompend.core.utils.exceptions import (
    LLMOutputError,
)
from kompend.llm.models import requires_output_ceiling
from kompend.llm.usage import UsageTracker
from kompend.logging import get_logger
from kompend.types import JSONValue

#: Per-stage timeout floor. Agents in kompend are synchronous single-turn;
#: PydanticAI's run_sync is fast but provider-side latency can spike.
_DEFAULT_TIMEOUT_SECONDS: float = 120.0

#: Sliding-window defaults live in :mod:`kompend.agents.context_manager` so
#: the ``sliding`` preset and the summarizer's fallback cannot drift apart.

#: Placeholder replacing the inlined context in prompts when RLM is
#: enabled — the context stays available in the deps (REPL ``context``
#: variable) but is not packed into the prompt.
RLM_CONTEXT_PLACEHOLDER: str = (
    "(The full context is not inlined in this prompt. It is pre-loaded as the `context` variable of the execute_code REPL — explore it there.)"
)

logger = get_logger(__name__)


@dataclass
class KompendAgentBase[ResultT, DepsT: AgentDeps](ABC):
    """kompend-side wrapper around a PydanticAI agent.

    Subclasses MUST set:

    - ``template_name`` — the filename under ``templates/<prefix>/``
      where ``prefix`` is set by ``template_prefix`` (default ``compilation/``).
    - ``system_template_name`` — the filename of the system prompt.
    - Generic type params ``[ResultT, DepsT]`` — ``result_type`` and
      ``deps_type`` are derived automatically from these.

    Subclasses MUST implement :meth:`_build_agent`.
    """

    model_string: str
    timeout: float = UNSET_TIMEOUT
    retries: int | AgentRetries = UNSET_RETRIES
    skills: dict[str, str] = field(default_factory=dict)
    usage_tracker: UsageTracker | None = None
    fallback_models: list[str] | None = None
    #: Reasoning effort from ``compile.llm.reasoning_effort``; ``None`` leaves
    #: the provider's default in place (same as ``ReasoningEffort.DEFAULT``).
    reasoning_effort: ReasoningEffort | None = None
    use_native_output: bool = False
    #: Context-manager preset from ``compile.llm.context_manager``. ``False``
    #: disables compaction; ``None`` defers to the config, then to disabled.
    context_manager: ContextManagerPreset | bool | None = None
    context_window: int | None = None
    delta_retry_enabled: bool | None = None
    tools: list[Tool[DepsT]] = field(default_factory=list)
    extra_capabilities: Sequence[AbstractCapability[DepsT]] = field(default_factory=list)
    temperature: float | None = None
    #: RLM configuration from the ``compile.llm.rlm`` group
    #: (:class:`~kompend.core.config._configs.RlmConfig`); ``None`` disables.
    rlm: RlmConfig | None = None
    #: Deps attribute holding the document context for the RLM REPL.
    rlm_context_field: str = "text"
    #: The ``compile.llm`` config group. When a construction site hands it
    #: over, every limit the site did not set explicitly is taken from the
    #: config (see :meth:`_apply_config_defaults`).
    llm: LLMConfig | None = None
    #: Format used for structured context injected into this agent's prompts
    #: (``compile.llm.data_format``). Compact formats (TOON/ISON) shrink the
    #: input, which leaves more context window for the answer.
    data_format: LlmDataFormat = LlmDataFormat.JSON

    # Class-level template config (overridden by subclasses, NOT dataclass fields)
    template_name: ClassVar[str] = ""
    system_template_name: ClassVar[str] = ""
    #: Subdirectory under ``templates/`` where this agent's templates live.
    #: Override in subclasses whose templates are not under ``compilation/``
    #: (e.g. ``"search/"``, ``"capture/"``, ``"organize/"``).
    template_prefix: ClassVar[str] = "compilation/"
    #: Opt-in for the delta-retry salvage hook. Subclasses set this to
    #: ``True`` to recover partially valid structured output (see
    #: :meth:`_make_delta_retry_hook`). The runtime switch is
    #: ``delta_retry_enabled``; this flag only marks the agent as
    #: capable of the hook.
    supports_delta_retry: ClassVar[bool] = False
    #: Optional model factory for agents whose model is not a chat model
    #: (the decision sidecar speaks the SystemOne protocol). ``None`` uses
    #: :func:`kompend.llm.models.build_model`.
    model_builder: ClassVar[Callable[[str], Model] | None] = None
    #: Declarative definition, when the agent is built from an
    #: :class:`~kompend.agents.base.spec.AgentSpec`. Specs own the type
    #: parameters and prompt wiring; a ``None`` spec keeps the legacy
    #: generic-parameter resolution.
    spec: ClassVar[AgentSpec | None] = None

    # Runtime-resolved type params (set in __post_init__, not constructor params)
    result_type: type = field(default=str, init=False, repr=False)
    deps_type: type[AgentDeps] = field(default=PipelineAgentDeps, init=False, repr=False)
    # Lazy-init internal state (not constructor params)
    _agent: Agent[DepsT, ResultT] | None = field(default=None, init=False, repr=False)
    #: False once a gateway rejected the prompt-cache field; the agent is then
    #: rebuilt without it and runs with caching off for its lifetime.
    _prompt_cache: bool = field(default=True, init=False, repr=False)
    _template_loader: TemplateLoader = field(init=False, repr=False)
    _current_source_id: str = field(default="", init=False, repr=False)
    _delta_retry_state: dict[int, JSONValue] = field(default_factory=dict, init=False, repr=False)
    _last_debug_file: Path | None = field(default=None, init=False, repr=False)
    _rlm_usage_limits: UsageLimits | None = field(default=None, init=False, repr=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Copy a declared spec's prompt wiring onto the class.

        A subclass that declares ``spec`` no longer repeats the template
        names and flags as class attributes; the spec stays the single
        definition. Subclasses without a spec keep their class attributes.
        """
        super().__init_subclass__(**kwargs)
        spec = cls.__dict__.get("spec")
        if spec is None:
            return
        cls.template_name = spec.template_name
        cls.system_template_name = spec.system_template_name
        cls.template_prefix = spec.template_prefix
        cls.supports_delta_retry = spec.supports_delta_retry
        # Assign through ``staticmethod``: a plain function on the class would
        # bind as a method and receive ``self`` as the model string.
        builder = spec.model_builder
        cls.model_builder = cast("Callable[[str], Model] | None", staticmethod(builder) if builder is not None else None)
        install_hooks(cls, spec)

    def __post_init__(self) -> None:
        # Resolve result_type / deps_type from the generic type params.
        self._resolve_type_params()
        self._template_loader = get_template_loader()
        # Normalize RLM: a disabled RlmConfig must not be truthy — prompt
        # rendering checks ``if self.rlm:`` to swap the context for the
        # RLM placeholder, and a Pydantic model is always truthy.
        if isinstance(self.rlm, RlmConfig) and not self.rlm.enable:
            self.rlm = None
        self._apply_config_defaults()

    @property
    def stage(self) -> str:
        """Pipeline stage label recorded in the LLM debug dump.

        Derived from the agent class name (e.g. ``ExtractionAgent`` ->
        ``extraction``), so no per-subclass attribute is needed.
        """
        return self.__class__.__name__.removesuffix("Agent").lower()

    def render_user_prompt(self, deps: DepsT) -> str:
        """Render the user-prompt template with the agent's deps.

        Only variables declared in the template's frontmatter are passed,
        avoiding warnings about unused variables.

        When RLM is enabled, the deps attribute named by ``rlm_context_field``
        is replaced with a placeholder in the rendered prompt — the full
        context stays in the deps (available to the REPL ``context`` variable)
        but is not inlined, so the model actually explores it via code.
        """
        template_file = f"{self.template_prefix}{self.template_name}"
        meta = self._template_loader.get_meta(template_file)
        declared = meta.declared_variable_names()
        vars_dict = self._deps_to_template_vars(deps)
        filtered = {k: v for k, v in vars_dict.items() if k in declared}
        if self.rlm and self.rlm_context_field in filtered:
            filtered[self.rlm_context_field] = RLM_CONTEXT_PLACEHOLDER
        return self._template_loader.render(template_file, **filtered)

    def render_system_prompt(self, deps: DepsT) -> str:
        """Render the system-prompt template, appending any skill instructions.

        System templates are static instructions — deps variables are not passed
        since no system template uses them. Returns an empty string when
        ``system_template_name`` is not set.
        """
        if not self.system_template_name:
            return ""
        prompt = self._template_loader.render(
            f"{self.template_prefix}{self.system_template_name}",
        )
        if self.skills:
            skill_blocks = []
            for name, instructions in self.skills.items():
                skill_blocks.append(f"## Skill: {name}\n\n{instructions}")
            prompt += "\n\n" + "\n\n---\n\n".join(skill_blocks)
        return prompt

    def run(
        self,
        deps: DepsT,
        *,
        message_history: list[ModelMessage] | None = None,
        user_prompt: str | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> ResultT:
        """Single-turn agent invocation. Returns the typed result.

        Args:
            deps: Typed context dataclass for this run.
            message_history: Optional prior conversation messages to
                continue from (used by incremental extraction to thread
                context across chunks of the same source).
            user_prompt: Overrides the deps-rendered prompt. Only useful
                together with ``message_history`` (a resume pass continues
                an interrupted conversation with a short continuation
                prompt instead of re-stating the full task).
            conversation_id: Groups the run with related runs (step
                persistence + conversation search scope). Inherited from
                ``message_history`` when omitted there.
            run_id: Explicit identity for this run. Step persistence keys
                its snapshots by it, and it is single-shot per store —
                a re-used id raises. Omitted: a fresh UUID7.

        Raises:
            LLMOutputError: if the LLM call fails or returns
                structurally invalid output.
        """
        if self._agent is None:
            self._agent = self._build_agent()
        self._current_source_id = deps.source_id
        if user_prompt is None:
            user_prompt = self.render_user_prompt(deps)
        system_prompt = self.render_system_prompt(deps)
        try:
            result = self._invoke(user_prompt, deps, system_prompt, message_history, conversation_id=conversation_id, run_id=run_id)
        except Exception as exc:
            raise self._failure(exc, deps) from exc
        return result.output

    def run_with_messages(
        self,
        deps: DepsT,
        *,
        message_history: list[ModelMessage] | None = None,
        user_prompt: str | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> tuple[ResultT, list[ModelMessage]]:
        """Single-turn agent invocation that also returns the new messages.

        Same as :meth:`run` but returns ``(output, new_messages)`` so callers
        can thread the conversation history into the next call (used by the
        incremental extraction strategy). The ``user_prompt`` /
        ``conversation_id`` / ``run_id`` kwargs forward to :meth:`run`.
        """
        if self._agent is None:
            self._agent = self._build_agent()
        self._current_source_id = deps.source_id
        if user_prompt is None:
            user_prompt = self.render_user_prompt(deps)
        system_prompt = self.render_system_prompt(deps)
        try:
            result = self._invoke(user_prompt, deps, system_prompt, message_history, conversation_id=conversation_id, run_id=run_id)
        except Exception as exc:
            raise self._failure(exc, deps) from exc
        return result.output, result.new_messages()

    @property
    def usage(self) -> UsageTracker | None:
        """The usage tracker, if one was provided."""
        return self.usage_tracker

    @property
    def current_source_id(self) -> str:
        """The source_id from the most recent ``run()`` call."""
        return self._current_source_id

    def tool_retry_budget(self) -> int:
        """The tool-call retry budget of this agent.

        ``retries`` is either a plain count (tool *and* output validation) or
        the split form — the tool half is the one that governs a tool that
        keeps raising ``ModelRetry``.
        """
        retries = self.retries
        if isinstance(retries, int):
            return retries
        return int(retries.get("tools", _DEFAULT_RETRIES))

    @staticmethod
    def validate_output(ctx: RunContext[DepsT], output: ResultT) -> ResultT:
        """Semantic output validator. Override in subclasses to add validation.

        Default: pass-through (no validation). Subclasses that override this
        method can raise ``ModelRetry`` to trigger a retry, or modify and
        return the output. It is a static method so an override never has to
        touch the instance — the base passes the result through unchanged.
        """
        return output

    def _failure(self, exc: Exception, deps: DepsT) -> LLMOutputError:
        """Delegate to :func:`kompend.agents.base.retries.failure`."""
        return failure(self, exc, deps)

    def _invoke(
        self,
        user_prompt: str,
        deps: DepsT,
        system_prompt: str,
        message_history: list[ModelMessage] | None,
        *,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> Any:
        """Delegate to :func:`kompend.agents.base.retries.invoke_agent`."""
        return invoke_agent(
            self,
            user_prompt,
            deps,
            system_prompt,
            message_history,
            conversation_id=conversation_id,
            run_id=run_id,
        )

    def _transient_retries(self) -> int:
        """Delegate to :func:`kompend.agents.base.retries.transient_retries`."""
        return transient_retries(self)

    def _transient_backoff(self) -> float:
        """Delegate to :func:`kompend.agents.base.retries.transient_backoff`."""
        return transient_backoff(self)

    def _apply_config_defaults(self) -> None:
        """Take every limit the construction site did not set from the config.

        A site may hand over the ``llm``/``budget`` config groups instead
        of spelling out timeout, retries, output ceiling, reasoning
        effort, context manager, fallback chain and delta-retry switch.
        Explicit constructor values always win — the config only fills
        the gaps, so every agent reacts to the same config.
        """
        llm = self.llm
        if llm is not None:
            if self.timeout == UNSET_TIMEOUT:
                self.timeout = llm.timeout
            if self.retries == UNSET_RETRIES:
                if llm.output_retries_enabled:
                    # Tool failures and schema-validation failures are
                    # different failure modes with different remedies, so
                    # they get their own budgets.
                    self.retries = AgentRetries(tools=llm.tool_retries, output=llm.output_retries)
                else:
                    self.retries = 0
            if self.reasoning_effort is None:
                self.reasoning_effort = llm.reasoning_effort
            if self.context_manager is None:
                self.context_manager = llm.context_manager
            if self.context_window is None and llm.context_window:
                self.context_window = llm.context_window
            if self.fallback_models is None and llm.fallback_chain:
                self.fallback_models = list(llm.fallback_chain)
            if self.delta_retry_enabled is None:
                self.delta_retry_enabled = llm.delta_retry_enabled
            self.data_format = llm.data_format
        if self.timeout == UNSET_TIMEOUT:
            self.timeout = _DEFAULT_TIMEOUT_SECONDS
        if self.retries == UNSET_RETRIES:
            self.retries = _DEFAULT_RETRIES
        if self.context_manager is None:
            self.context_manager = False
        if self.delta_retry_enabled is None:
            self.delta_retry_enabled = True

    def _usage_limits_for(self, deps: DepsT) -> UsageLimits | None:
        """Delegate to :func:`kompend.agents.base.retries.usage_limits_for`."""
        return usage_limits_for(self, deps)

    def _session_limits(
        self,
        *,
        units: int,
        per_unit: int,
        headroom_frac: float = 1.1,
        session_overhead: int = 2,
    ) -> UsageLimits | None:
        """Delegate to :func:`kompend.agents.base.retries.session_limits`."""
        return session_limits(
            self,
            units=units,
            per_unit=per_unit,
            headroom_frac=headroom_frac,
            session_overhead=session_overhead,
        )

    def _retry_requests(self) -> int:
        """Delegate to :func:`kompend.agents.base.retries.retry_requests`."""
        return retry_requests(self)

    def _resolve_type_params(self) -> None:
        """Resolve ``result_type`` and ``deps_type`` from generic type params.

        Walks the MRO to find ``KompendAgentBase[X, Y]`` and extracts X and Y.
        Falls back to ``str`` / ``PipelineAgentDeps`` if resolution fails
        (e.g. the subclass uses ``Any`` or doesn't parameterize).

        A class that carries an :class:`AgentSpec` takes its types from the
        spec instead — the spec is the single definition, so there is no
        second place for the result or deps type to drift.
        """
        if self.spec is not None:
            self.result_type = self.spec.result_type
            self.deps_type = self.spec.deps_type
            return
        for base in getattr(type(self), "__orig_bases__", ()):
            args = typing.get_args(base)
            if len(args) >= 2:
                result_arg, deps_arg = args[0], args[1]
                self.result_type = _coerce_type(result_arg, str)
                self.deps_type = cast("type[AgentDeps]", _coerce_type(deps_arg, PipelineAgentDeps))
                return
        self.result_type = str
        self.deps_type = PipelineAgentDeps

    def _build_agent(self) -> Agent[DepsT, ResultT]:
        """Construct the wrapped PydanticAI agent. Called lazily on first run().

        Uses :func:`kompend.llm.models.build_model` to construct a native
        pydantic-ai Model from the configured model string, wraps it in
        :class:`~kompend.agents.kompend_model.KompendModel` for usage tracking and
        debug saving, and in :class:`~pydantic_ai.models.fallback.FallbackModel`
        when fallback models are configured.
        """
        # One-shot OTel instrumentation of pydantic-ai agents. No-op unless
        # KOMP_OTEL_ENABLED is set; guarded internally so it only runs once.
        configure_observability()

        model_settings: ModelSettings = {"timeout": self.timeout}
        if self.temperature is not None:
            model_settings["temperature"] = self.temperature

        output_type: OutputSpec[Any] = self.result_type
        if self.use_native_output:
            output_type = NativeOutput(
                self.result_type,
                name=self.stage,
                description=f"Structured output for {self.stage}",
            )

        # Build the top-level model once and reuse it for both the Agent and
        # the summarization capability (same failover chain).
        top_model = self.model_builder(self.model_string) if self.model_builder else self._build_top_level_model()
        # Cache the conversation prefix. The agentic loops re-send it on every
        # pass, so this is the difference between N prefills and N cache reads.
        # ``cache`` is pydantic-ai's unified setting: it translates to the
        # provider's own mechanism (Anthropic ``cache_control`` breakpoints,
        # OpenAI cache options/key, OpenRouter pass-through) and is a no-op
        # where the provider caches implicitly or not at all.
        if self._prompt_cache and (self.llm.prompt_cache if self.llm else True):
            model_settings = {**model_settings, "cache": True}
        settings: AgentModelSettings[DepsT] = model_settings
        if requires_output_ceiling(top_model):
            settings = self._ceiling_model_settings(model_settings)
        capability = self._resolve_context_capability(top_model)
        capabilities: list[AbstractCapability[DepsT]] = []
        if capability is not None:
            capabilities.append(capability)
            if self.usage_tracker is not None:
                # The harness reports token usage through an event; keep feeding
                # the tracker so context usage stays visible in the run summary.
                # The gauge measures against the same (possibly capped) window
                # the compaction capability triggers on - reporting the model's
                # raw window would understate how full the context really is.
                capabilities.append(
                    ReportContextUsage(
                        on_usage=self._on_usage_update,
                        context_window=self.context_window,
                    )
                )
        if self.delta_retry_enabled and type(self).supports_delta_retry:
            capabilities.append(self._make_delta_retry_hook())
        rlm_capability = self._resolve_rlm_capability(top_model)
        if rlm_capability is not None:
            capabilities.append(rlm_capability)
            self._rlm_usage_limits = UsageLimits(
                tool_calls_limit=rlm_capability.max_tool_calls,
                # pydantic-ai's default request_limit (50) binds before the
                # tool-call budget: each tool call costs >=1 model request, and
                # retries can double a turn. Derive from max_tool_calls.
                request_limit=rlm_capability.max_tool_calls * 2 + 5,
            )
        capabilities.extend(self.extra_capabilities)
        agent = cast(
            "Agent[DepsT, ResultT]",
            Agent(
                top_model,
                deps_type=cast("type[DepsT]", self.deps_type),
                output_type=output_type,
                model_settings=settings,
                retries=self.retries,
                tools=self.tools,
                capabilities=capabilities or None,
                # Descriptive agent name for observability (gen_ai.agent.name):
                # kebab-cased subclass name, e.g. AgenticGenerationAgent ->
                # "agentic-generation-agent".
                name=_agent_name(type(self).__name__),
            ),
        )
        if type(self).validate_output is not KompendAgentBase.validate_output:
            agent.output_validator(self.validate_output)
        return agent

    def _ceiling_model_settings(self, base: ModelSettings) -> AgentModelSettings[DepsT]:
        """Delegate to :func:`kompend.agents.base.models.ceiling_model_settings`."""
        return ceiling_model_settings(self, base)

    def _output_ceiling(self, messages: Sequence[ModelMessage]) -> int | None:
        """Delegate to :func:`kompend.agents.base.models.output_ceiling`."""
        return output_ceiling(self, messages)

    def _estimate_prompt_tokens(self, messages: Sequence[ModelMessage]) -> int | None:
        """Delegate to :func:`kompend.agents.base.models.estimate_prompt_tokens`."""
        return estimate_prompt_tokens(self, messages)

    def _model_max_output(self) -> int | None:
        """Delegate to :func:`kompend.agents.base.models.model_max_output`."""
        return model_max_output(self)

    def _build_top_level_model(self) -> Model:
        """Delegate to :func:`kompend.agents.base.models.build_top_level_model`."""
        return build_top_level_model(self)

    def _make_delta_retry_hook(self) -> Hooks:
        """Delegate to :func:`kompend.agents.base.delta_retry.make_delta_retry_hook`."""
        return make_delta_retry_hook(self)

    def _resolve_context_capability(self, model: Model) -> AbstractCapability[DepsT] | None:
        """Delegate to :func:`kompend.agents.base.context.resolve_context_capability`."""
        return resolve_context_capability(self, model)

    def _resolve_rlm_capability(self, model: Model) -> RLMCapability | None:
        """Delegate to :func:`kompend.agents.base.context.resolve_rlm_capability`."""
        return resolve_rlm_capability(self, model)

    # ── compaction callbacks ─────────────────────────────

    def _on_usage_update(self, usage: ContextUsage) -> None:
        """Delegate to :func:`kompend.agents.base.context.on_usage_update`."""
        on_usage_update(self, usage)

    @staticmethod
    def _deps_to_template_vars(deps: DepsT) -> dict[str, object]:
        """Convert a Deps dataclass instance to a dict for Jinja2 rendering."""
        if dataclasses.is_dataclass(deps) and not isinstance(deps, type):
            return {f.name: getattr(deps, f.name) for f in dataclasses.fields(deps) if not f.name.startswith("_")}
        return {}


# ── Agent naming ───────────────────────────────────────────────────────


def _agent_name(class_name: str) -> str:
    """Derive a descriptive observability name from the agent class name.

    CamelCase -> kebab-case: ``AgenticGenerationAgent`` ->
    ``agentic-generation-agent``. Feeds pydantic-ai's ``Agent(name=...)`` so
    Logfire's ``gen_ai.agent.name`` attribute identifies each agent.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "-", class_name).lower()
