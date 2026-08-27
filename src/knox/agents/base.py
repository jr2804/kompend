"""KnoxAgent base class — KNOX-side wrapper around pydantic_ai.Agent.

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

- **Model bridge** — creates a :class:`KnoxLiteLLMModel`
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
import json
import typing
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_ai import Agent, ModelRequestContext, NativeOutput, RunContext, Tool, UsageLimits
from pydantic_ai.agent.abstract import AgentRetries
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.output import OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai_summarization import (
    ContextManagerCapability,
    ContextSize,
    SlidingWindowCapability,
)

from knox.agents.instrument import configure_observability
from knox.agents.knox_model import KnoxModel
from knox.agents.rlm import (
    _DEFAULT_NODE_BUDGET,
    RLMCapability,
)
from knox.agents.types import PipelineAgentDeps
from knox.core.config._configs import RlmConfig
from knox.core.templates import TemplateLoader, get_template_loader
from knox.core.utils.exceptions import LLMOutputError
from knox.llm.format import salvage_json
from knox.llm.models import build_model
from knox.llm.usage import UsageTracker
from knox.logging import get_logger
from knox.types import JSONDict, JSONValue

#: Per-stage timeout floor. Agents in KNOX are synchronous single-turn;
#: PydanticAI's run_sync is fast but provider-side latency can spike.
_DEFAULT_TIMEOUT_SECONDS: float = 120.0

#: Default retries when no config value is provided. Matches the config
#: default in ``compile.llm.output_retries``.
_DEFAULT_RETRIES: int = 2

#: ContextManagerCapability presets: name -> (compress_threshold, keep).
#: ``keep`` uses the ``ContextSize`` shape (``("messages", N)``).
_CONTEXT_PRESETS: dict[str, tuple[float, ContextSize]] = {
    "default": (0.9, ("messages", 0)),
    "aggressive": (0.7, ("messages", 5)),
}

#: SlidingWindowCapability default trigger/keep (zero-cost, no LLM call).
_SLIDING_TRIGGER: ContextSize = ("messages", 100)
_SLIDING_KEEP: ContextSize = ("messages", 50)

#: Placeholder replacing the inlined context in prompts when RLM is
#: enabled — the context stays available in the deps (REPL ``context``
#: variable) but is not packed into the prompt.
RLM_CONTEXT_PLACEHOLDER: str = (
    "(The full context is not inlined in this prompt. It is pre-loaded as the `context` variable of the execute_code REPL — explore it there.)"
)

logger = get_logger(__name__)

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


@dataclass
class KnoxAgentBase[ResultT, DepsT: AgentDeps](ABC):
    """KNOX-side wrapper around a PydanticAI agent.

    Subclasses MUST set:

    - ``template_name`` — the filename under ``templates/<prefix>/``
      where ``prefix`` is set by ``template_prefix`` (default ``compilation/``).
    - ``system_template_name`` — the filename of the system prompt.
    - Generic type params ``[ResultT, DepsT]`` — ``result_type`` and
      ``deps_type`` are derived automatically from these.

    Subclasses MUST implement :meth:`_build_agent`.
    """

    model_string: str
    timeout: float = _DEFAULT_TIMEOUT_SECONDS
    retries: int | AgentRetries = _DEFAULT_RETRIES
    max_tokens: int | None = None
    skills: dict[str, str] = field(default_factory=dict)
    usage_tracker: UsageTracker | None = None
    fallback_models: list[str] | None = None
    reasoning_effort: str | None = None
    use_native_output: bool = False
    context_manager: str | bool = False
    context_window: int | None = None
    delta_retry_enabled: bool = True
    tools: list[Tool] = field(default_factory=list)
    extra_capabilities: list[AbstractCapability[DepsT]] = field(default_factory=list)
    temperature: float | None = None
    use_delta_retry: bool = False
    #: RLM configuration from the ``compile.llm.rlm`` group
    #: (:class:`~knox.core.config._configs.RlmConfig`); ``None`` disables.
    rlm: RlmConfig | None = None
    #: Deps attribute holding the document context for the RLM REPL.
    rlm_context_field: str = "text"

    # Class-level template config (overridden by subclasses, NOT dataclass fields)
    template_name: ClassVar[str] = ""
    system_template_name: ClassVar[str] = ""
    #: Subdirectory under ``templates/`` where this agent's templates live.
    #: Override in subclasses whose templates are not under ``compilation/``
    #: (e.g. ``"search/"``, ``"capture/"``, ``"organize/"``).
    template_prefix: ClassVar[str] = "compilation/"

    # Runtime-resolved type params (set in __post_init__, not constructor params)
    result_type: type = field(default=str, init=False, repr=False)
    deps_type: type[AgentDeps] = field(default=PipelineAgentDeps, init=False, repr=False)
    # Lazy-init internal state (not constructor params)
    _agent: Agent[DepsT, ResultT] | None = field(default=None, init=False, repr=False)
    _template_loader: TemplateLoader = field(init=False, repr=False)
    _current_source_id: str = field(default="", init=False, repr=False)
    _delta_retry_state: dict[int, JSONValue] = field(default_factory=dict, init=False, repr=False)
    _last_debug_file: Path | None = field(default=None, init=False, repr=False)
    _rlm_usage_limits: UsageLimits | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Resolve result_type / deps_type from the generic type params.
        self._resolve_type_params()
        self._template_loader = get_template_loader()
        # Normalize RLM: a disabled RlmConfig must not be truthy — prompt
        # rendering checks ``if self.rlm:`` to swap the context for the
        # RLM placeholder, and a Pydantic model is always truthy.
        if isinstance(self.rlm, RlmConfig) and not self.rlm.enable:
            self.rlm = None

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

    def validate_output(self, ctx: RunContext[DepsT], output: ResultT) -> ResultT:  # noqa: PLR6301
        """Semantic output validator. Override in subclasses to add validation.

        Default: pass-through (no validation). Subclasses that override this
        method can raise ``ModelRetry`` to trigger a retry, or modify and
        return the output.
        """
        return output

    def run(
        self,
        deps: DepsT,
        *,
        message_history: list[ModelMessage] | None = None,
    ) -> ResultT:
        """Single-turn agent invocation. Returns the typed result.

        Args:
            deps: Typed context dataclass for this run.
            message_history: Optional prior conversation messages to
                continue from (used by incremental extraction to thread
                context across chunks of the same source).

        Raises:
            LLMOutputError: if the LLM call fails or returns
                structurally invalid output.
        """
        if self._agent is None:
            self._agent = self._build_agent()
        self._current_source_id = deps.source_id
        user_prompt = self.render_user_prompt(deps)
        system_prompt = self.render_system_prompt(deps)
        try:
            result = self._agent.run_sync(
                user_prompt,
                deps=deps,
                instructions=system_prompt,
                message_history=message_history,
                usage_limits=self._rlm_usage_limits,
            )
        except Exception as exc:
            debug_hint = f" (debug: {self._last_debug_file})" if self._last_debug_file else ""
            logger.warning(
                "Agent '%s' failed on '%s': %s%s",
                type(self).__name__,
                deps.source_id,
                exc,
                debug_hint,
            )
            raise LLMOutputError(f"Agent failed for model '{self.model_string}': {exc}{debug_hint}") from exc
        return result.output

    def run_with_messages(
        self,
        deps: DepsT,
        *,
        message_history: list[ModelMessage] | None = None,
    ) -> tuple[ResultT, list[ModelMessage]]:
        """Single-turn agent invocation that also returns the new messages.

        Same as :meth:`run` but returns ``(output, new_messages)`` so callers
        can thread the conversation history into the next call (used by the
        incremental extraction strategy).
        """
        if self._agent is None:
            self._agent = self._build_agent()
        self._current_source_id = deps.source_id
        user_prompt = self.render_user_prompt(deps)
        system_prompt = self.render_system_prompt(deps)
        try:
            result = self._agent.run_sync(
                user_prompt,
                deps=deps,
                instructions=system_prompt,
                message_history=message_history,
                usage_limits=self._rlm_usage_limits,
            )
        except Exception as exc:
            debug_hint = f" (debug: {self._last_debug_file})" if self._last_debug_file else ""
            logger.warning(
                "Agent '%s' failed on '%s': %s%s",
                type(self).__name__,
                deps.source_id,
                exc,
                debug_hint,
            )
            raise LLMOutputError(f"Agent failed for model '{self.model_string}': {exc}{debug_hint}") from exc
        return result.output, result.new_messages()

    @property
    def usage(self) -> UsageTracker | None:
        """The usage tracker, if one was provided."""
        return self.usage_tracker

    @property
    def current_source_id(self) -> str:
        """The source_id from the most recent ``run()`` call."""
        return self._current_source_id

    def _resolve_type_params(self) -> None:
        """Resolve ``result_type`` and ``deps_type`` from generic type params.

        Walks the MRO to find ``KnoxAgentBase[X, Y]`` and extracts X and Y.
        Falls back to ``str`` / ``PipelineAgentDeps`` if resolution fails
        (e.g. the subclass uses ``Any`` or doesn't parameterize).
        """
        for base in getattr(type(self), "__orig_bases__", ()):
            args = typing.get_args(base)
            if len(args) >= 2:
                result_arg, deps_arg = args[0], args[1]
                self.result_type = self._coerce_type(result_arg, str)
                self.deps_type = cast("type[AgentDeps]", self._coerce_type(deps_arg, PipelineAgentDeps))
                return
        self.result_type = str
        self.deps_type = PipelineAgentDeps

    def _build_agent(self) -> Agent[DepsT, ResultT]:
        """Construct the wrapped PydanticAI agent. Called lazily on first run().

        Uses :func:`knox.llm.models.build_model` to construct a native
        pydantic-ai Model from the configured model string, wraps it in
        :class:`~knox.agents.knox_model.KnoxModel` for usage tracking and
        debug saving, and in :class:`~pydantic_ai.models.fallback.FallbackModel`
        when fallback models are configured.
        """
        # One-shot OTel instrumentation of pydantic-ai agents. No-op unless
        # KNOX_OTEL_ENABLED is set; guarded internally so it only runs once.
        configure_observability()

        model_settings: ModelSettings = {"timeout": self.timeout}
        if self.temperature is not None:
            model_settings["temperature"] = self.temperature
        if self.max_tokens is not None:
            model_settings["max_tokens"] = self.max_tokens

        output_type: OutputSpec[Any] = self.result_type
        if self.use_native_output:
            output_type = NativeOutput(
                self.result_type,
                name=self.stage,
                description=f"Structured output for {self.stage}",
            )

        # Build the top-level model once and reuse it for both the Agent and
        # the summarization capability (same failover chain).
        top_model = self._build_top_level_model()
        capability = self._resolve_context_capability(top_model)
        capabilities: list[AbstractCapability[DepsT]] = []
        if capability is not None:
            capabilities.append(capability)
        if self.delta_retry_enabled and self.use_delta_retry:
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
                model_settings=model_settings,
                retries=self.retries,
                capabilities=capabilities or None,
            ),
        )
        if type(self).validate_output is not KnoxAgentBase.validate_output:
            agent.output_validator(self.validate_output)
        return agent

    def _build_top_level_model(self) -> Model:
        """Build the model (with failover + KNOX wrapper) for the Agent."""
        primary = KnoxModel(
            build_model(self.model_string),
            usage_tracker=self.usage_tracker,
            agent=self,
            reasoning_effort=self.reasoning_effort,
        )
        if not self.fallback_models:
            return primary

        return FallbackModel(
            primary,
            *(
                KnoxModel(
                    build_model(m),
                    usage_tracker=self.usage_tracker,
                    agent=self,
                    reasoning_effort=self.reasoning_effort,
                )
                for m in self.fallback_models
            ),
        )

    def _make_delta_retry_hook(self) -> Hooks:
        """Create an ``after_model_request`` hook for delta-retry.

        When the model returns partially valid structured output, this hook
        extracts the valid parts and raises ``ModelRetry`` with a message
        that tells the model exactly which parts need fixing and what the
        expected structure looks like.  On retry, valid parts from previous
        attempts are accumulated and merged.

        Only fires for agents with ``use_delta_retry = True`` and a Pydantic
        ``result_type`` (structured output).
        """
        state = self._delta_retry_state
        hooks = Hooks()

        @hooks.on.after_model_request
        async def _delta_retry_hook(
            ctx: RunContext[DepsT],
            *,
            request_context: ModelRequestContext,
            response: ModelResponse,
        ) -> ModelResponse:
            # Only handle structured output (TextPart with JSON)
            schema = self.result_type
            if not isinstance(schema, type) or not issubclass(schema, BaseModel):
                return response  # text output — nothing to salvage
            text = _get_response_text(response)
            if not text:
                return response

            # Try to parse with schema-aware salvage
            parsed = salvage_json(text, schema=schema)
            if parsed is None:
                return response  # Can't parse — let PydanticAI handle it

            # Try to validate against the schema
            try:
                validated = schema.model_validate(parsed)
            except Exception:
                # Validation failed — build a delta message
                step = ctx.run_step
                delta_msg = _build_delta_message(parsed, schema)
                if delta_msg:
                    # Store the partial data for potential merge on retry
                    state[step] = parsed
                    raise ModelRetry(delta_msg)
                return response

            # Fully valid — replace response text with corrected JSON so
            # PydanticAI validates the fixed version, not the original broken one.
            # salvage_json may have restructured fields, added defaults, or coerced
            # types — without this, PydanticAI re-validates the original text.
            corrected_json = validated.model_dump_json()
            if corrected_json != text:
                _set_response_text(response, corrected_json)

            # Merge with any accumulated state from previous retries
            step = ctx.run_step
            if step in state:
                merged = _merge_valid_data(state[step], validated.model_dump(mode="json"), schema)
                if merged is not None:
                    _set_response_text(response, json.dumps(merged, ensure_ascii=False, default=str))
                del state[step]
            return response

        return hooks

    def _resolve_context_capability(self, model: Model) -> AbstractCapability[DepsT] | None:
        """Build the context-management capability from the configured preset.

        Returns ``None`` when context management is disabled, in which case the
        pydantic-ai Agent is constructed without ``capabilities``.

        Args:
            model: The top-level (failover) model — reused as the
                summarization model so compressions run on the same chain.
        """
        setting = self.context_manager
        if setting is False or setting is None:
            return None
        if isinstance(setting, str):
            normalized = setting.strip().lower()
            if normalized in {"", "off", "false", "none", "no"}:
                return None
            preset = normalized
        else:
            preset = "default"  # ``True`` -> default preset

        if preset == "sliding":
            return SlidingWindowCapability(
                trigger=_SLIDING_TRIGGER,
                keep=_SLIDING_KEEP,
            )
        try:
            threshold, keep = _CONTEXT_PRESETS[preset]
        except KeyError:
            logger.warning(
                "Unknown context_manager preset %r — falling back to 'default'",
                setting,
            )
            threshold, keep = _CONTEXT_PRESETS["default"]
        return ContextManagerCapability(
            max_tokens=self.context_window,
            compress_threshold=threshold,
            keep=keep,
            summarization_model=model,
            on_usage_update=self._on_usage_update,
            on_before_compress=self._on_before_compress,
            on_after_compress=self._on_after_compress,
        )

    def _resolve_rlm_capability(self, model: Model) -> RLMCapability | None:
        """Build the RLM capability from the ``compile.llm.rlm`` group.

        Returns ``None`` when RLM is disabled (``rlm`` unset or
        ``enable = false``). The sub-model defaults to the agent's top-level
        model (same failover chain); ``sub_model`` resolves via
        ``build_model`` and is wrapped in :class:`KnoxModel` for usage
        tracking.

        Args:
            model: The top-level (failover) model — reused as the
                ``llm_query`` sub-model when ``sub_model`` is unset.
        """
        rlm = self.rlm
        if not isinstance(rlm, RlmConfig) or not rlm.enable:
            return None
        max_tool_calls = rlm.max_tool_calls
        code_timeout = rlm.code_timeout
        sub_model_str = rlm.sub_model
        depth = rlm.max_depth
        node_budget = rlm.node_budget
        if sub_model_str:
            sub_model: Model = KnoxModel(
                build_model(sub_model_str),
                usage_tracker=self.usage_tracker,
            )
        else:
            sub_model = model
        return RLMCapability(
            context_field=self.rlm_context_field,
            code_timeout=code_timeout,
            sub_model=sub_model,
            max_tool_calls=max_tool_calls,
            main_model=model,
            depth=depth,
            node_budget=node_budget or _DEFAULT_NODE_BUDGET,
        )

    # ── ContextManagerCapability callbacks ─────────────────────────────

    def _on_usage_update(self, pct: float, current: int, max_tokens: int) -> None:
        """Token-usage snapshot from the capability → UsageTracker."""
        if self.usage_tracker is not None:
            self.usage_tracker.record_context_usage(current, max_tokens)

    @staticmethod
    def _coerce_type(arg: type[Any], fallback: type) -> type:
        """Resolve a TypeVar arg to a usable runtime type.

        Handles plain types (``str``), generic aliases (``dict[str, Any]``),
        and TypeVars (falls back to *fallback*).
        """
        # Plain class — use directly
        if isinstance(arg, type):
            return arg
        # Generic alias like dict[str, Any] — get_origin returns dict
        origin = typing.get_origin(arg)
        if origin is not None and isinstance(origin, type):
            return origin
        return fallback

    @staticmethod
    def _on_before_compress(messages: list[ModelMessage], cutoff_index: int) -> None:
        """Fires between plan and execute — log the impending compression."""
        logger.info(
            "Context compression starting: %d messages, cutoff at index %d",
            len(messages),
            cutoff_index,
        )

    @staticmethod
    def _on_after_compress(messages: list[ModelMessage], summarized: bool, summary: str | None) -> None:
        """Fires after the summary LLM — log the outcome. Returns no reinject."""
        if summarized:
            logger.info("Context compression produced %d-char summary", len(summary or ""))
        else:
            logger.warning("Context compression skipped (history preserved)")

    @staticmethod
    def _deps_to_template_vars(deps: DepsT) -> dict[str, object]:
        """Convert a Deps dataclass instance to a dict for Jinja2 rendering."""
        if dataclasses.is_dataclass(deps) and not isinstance(deps, type):
            return {f.name: getattr(deps, f.name) for f in dataclasses.fields(deps) if not f.name.startswith("_")}
        return {}


# ── Delta-retry helpers ────────────────────────────────────────────────


def _get_response_text(response: ModelResponse) -> str | None:
    """Extract the text content from a ModelResponse."""
    for part in response.parts:
        if isinstance(part, TextPart) and part.content:
            return part.content
    return None


def _set_response_text(response: ModelResponse, data: str) -> None:
    """Replace the text content of a ModelResponse with JSON text."""
    for part in response.parts:
        if isinstance(part, TextPart):
            part.content = data
            return
    response.parts = [*response.parts, TextPart(content=data)]


def _build_delta_message(parsed: JSONValue, schema: type[BaseModel]) -> str | None:
    """Build a delta-retry message describing what needs fixing.

    Examines the parsed (but invalid) output against the schema and
    produces a message that tells the model exactly which parts are
    broken and what the expected structure looks like.
    """
    issues: list[str] = []

    # Check if the model returned a list when the schema expects an object.
    if isinstance(parsed, list):
        expected = _format_schema_fields(schema)
        issues.append(f"- You returned a JSON array, but the expected output is a single JSON object with fields {expected}. Return one object, not a list.")

    if isinstance(parsed, dict):
        # Check each field against the schema
        for field_name, field_info in schema.model_fields.items():
            _check_field(parsed, field_name, field_info, issues)

    if not issues:
        return None

    issue_text = "\n".join(issues)
    return (
        "[CORRECTION REQUEST — This is NOT a new task. Your previous output had "
        "the following structural issues. Fix them and return the COMPLETE corrected "
        "output, keeping all valid parts unchanged.]\n\n"
        f"{issue_text}\n\n"
        "Return the COMPLETE corrected output as a single valid JSON object matching "
        "the expected schema exactly. Do not add explanations, markdown formatting, "
        "or extra fields. If this keeps failing, your model may not support structured "
        "JSON output — consider disabling delta-retry in the config "
        "(`compile.llm.delta_retry_enabled = false`)."
    )


def _check_field(
    parsed: dict[str, JSONValue],
    field_name: str,
    field_info: FieldInfo,
    issues: list[str],
) -> None:
    """Compare one parsed output field against its schema declaration."""
    if field_name not in parsed:
        issues.append(f"- missing field '{field_name}'")
        return
    value = parsed[field_name]
    inner_type = _get_list_inner_type(field_info.annotation)
    if not (inner_type and isinstance(value, list)):
        return
    expected = _format_schema_fields(inner_type)
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            issues.append(f"- {field_name}[{i}]: expected an object with fields {expected}, got {type(item).__name__} ({json.dumps(str(item))})")


def _get_list_inner_type(annotation: Any) -> type[BaseModel] | None:
    """Extract the inner pydantic-model type from a ``list[X]`` annotation."""
    origin = typing.get_origin(annotation)
    if origin is not list:
        return None
    args = typing.get_args(annotation)
    inner = args[0] if args else None
    if isinstance(inner, type) and issubclass(inner, BaseModel):
        return inner
    return None


def _format_schema_fields(model: object) -> str:
    """Format the field names of a Pydantic model for display."""
    if isinstance(model, type) and issubclass(model, BaseModel):
        return "{" + ", ".join(model.model_fields) + "}"
    return "{...}"


def _merge_valid_data(previous: JSONValue, current: JSONDict, schema: type[BaseModel]) -> JSONDict | None:
    """Merge valid data from a previous retry attempt with the current one.

    When the model returns fewer items on retry (e.g., only the corrected
    pages), merge them with the previously valid items by replacing the
    tail of the previous list with the new items.
    """
    if not isinstance(previous, dict):
        return None

    merged = dict(current)
    for field_name in schema.model_fields:
        if field_name not in previous or field_name not in current:
            continue
        prev_val = previous[field_name]
        curr_val = current[field_name]
        if isinstance(prev_val, list) and isinstance(curr_val, list) and len(curr_val) < len(prev_val):
            # New response has fewer items — replace the tail of previous
            merged[field_name] = prev_val[: len(prev_val) - len(curr_val)] + curr_val

    return merged
