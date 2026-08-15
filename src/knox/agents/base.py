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

import json
import typing
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar, cast, runtime_checkable

from pydantic_ai import Agent, NativeOutput, RunContext, Tool, UsageLimits
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai_summarization import (
    ContextManagerCapability,
    ContextSize,
    SlidingWindowCapability,
)

from knox.agents.deps import PipelineAgentDeps
from knox.agents.knox_model import KnoxModel
from knox.agents.rlm import RLMCapability, resolve_rlm_preset
from knox.core.templates import TemplateLoader, get_template_loader
from knox.core.utils.exceptions import LLMOutputError
from knox.llm.format import salvage_json
from knox.llm.models import build_model
from knox.llm.usage import UsageTracker
from knox.logging import get_logger
from knox.types import AgentResult

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
    retries: int = _DEFAULT_RETRIES
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
    temperature: float | None = None
    use_delta_retry: bool = False
    #: RLM preset (``False`` off, ``"default"``/``"aggressive"``/``True``)
    #: enabling code-exploration tools on this agent.
    rlm: str | bool = False
    #: Optional sub-model string for ``llm_query()`` inside the RLM REPL;
    #: defaults to the agent's main model when unset.
    rlm_sub_model: str | None = None
    #: Deps attribute holding the document context for the RLM REPL.
    rlm_context_field: str = "text"
    #: Remaining recursion depth for ``rlm_query`` inside the REPL
    #: (0 = non-recursive, only ``llm_query``). Mirrors ``rlm_max_depth``
    #: config at the wiring sites.
    rlm_depth: int = 0

    # Class-level template config (overridden by subclasses, NOT dataclass fields)
    template_name: ClassVar[str] = ""
    system_template_name: ClassVar[str] = ""
    #: Subdirectory under ``templates/`` where this agent's templates live.
    #: Override in subclasses whose templates are not under ``compilation/``
    #: (e.g. ``"search/"``, ``"capture/"``, ``"organize/"``).
    template_prefix: ClassVar[str] = "compilation/"

    # Runtime-resolved type params (set in __post_init__, not constructor params)
    result_type: type = field(default=str, init=False, repr=False)
    deps_type: type[AgentDeps] = field(default=PipelineAgentDeps, init=False, repr=False)  # type: ignore[assignment]

    # Lazy-init internal state (not constructor params)
    _agent: Agent[DepsT, ResultT] | None = field(default=None, init=False, repr=False)
    _template_loader: TemplateLoader = field(init=False, repr=False)
    _current_source_id: str = field(default="", init=False, repr=False)
    _delta_retry_state: dict[int, AgentResult] = field(default_factory=dict, init=False, repr=False)
    _last_debug_file: Path | None = field(default=None, init=False, repr=False)
    _rlm_usage_limits: UsageLimits | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Resolve result_type / deps_type from the generic type params.
        self._resolve_type_params()
        self._template_loader = get_template_loader()

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
                self.deps_type = self._coerce_type(deps_arg, PipelineAgentDeps)
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
        model_settings: ModelSettings = {"timeout": self.timeout}
        if self.temperature is not None:
            model_settings["temperature"] = self.temperature
        if self.max_tokens is not None:
            model_settings["max_tokens"] = self.max_tokens

        output_type: type | NativeOutput = self.result_type
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
            )
        agent = Agent(
            top_model,
            deps_type=self.deps_type,
            output_type=output_type,
            model_settings=model_settings,
            retries=self.retries,
            capabilities=capabilities or None,
        )
        if type(self).validate_output is not KnoxAgentBase.validate_output:
            agent.output_validator(self.validate_output)
        return cast("Agent[DepsT, ResultT]", agent)

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
        schema = self.result_type
        state = self._delta_retry_state

        hooks = Hooks()

        @hooks.on.after_model_request
        async def _delta_retry_hook(
            ctx: RunContext[DepsT],
            *,
            request_context: RunContext[DepsT],
            response: ModelResponse,
        ) -> ModelResponse:
            # Only handle structured output (TextPart with JSON)
            text = _get_response_text(response)
            if not text:
                return response

            # Try to parse with schema-aware salvage
            parsed = salvage_json(text, schema=schema)
            if parsed is None:
                return response  # Can't parse — let PydanticAI handle it

            # Try to validate against the schema
            try:
                validated = schema.model_validate(parsed)  # ty: ignore[unresolved-attribute]
            except Exception:
                # Validation failed — build a delta message
                step = ctx.run_step if hasattr(ctx, "run_step") else 0
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
            step = ctx.run_step if hasattr(ctx, "run_step") else 0
            if step in state:
                merged = _merge_valid_data(state[step], validated, schema)
                if merged is not None:
                    _set_response_text(response, merged)
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
        """Build the RLM capability from the configured preset.

        Returns ``None`` when RLM is disabled. The sub-model defaults to the
        agent's top-level model (same failover chain); an explicit
        ``rlm_sub_model`` resolves via ``build_model`` and is wrapped in
        :class:`KnoxModel` for usage tracking.

        Args:
            model: The top-level (failover) model — reused as the
                ``llm_query`` sub-model when ``rlm_sub_model`` is unset.
        """
        if self.rlm is False or self.rlm is None:
            return None
        try:
            max_tool_calls, code_timeout = resolve_rlm_preset(self.rlm)
        except ValueError:
            return None
        if self.rlm_sub_model:
            sub_model: Model = KnoxModel(
                build_model(self.rlm_sub_model),
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
            depth=self.rlm_depth,
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
    def _deps_to_template_vars(deps: DepsT) -> dict[str, Any]:
        """Convert a Deps instance to a dict for Jinja2 rendering."""
        if hasattr(deps, "__dict__"):
            return {k: v for k, v in deps.__dict__.items() if not k.startswith("_")}
        return {}


# ── Delta-retry helpers ────────────────────────────────────────────────


def _get_response_text(response: ModelResponse) -> str | None:
    """Extract the text content from a ModelResponse."""
    for part in response.parts:
        if isinstance(part, TextPart) and part.content:
            return part.content
    return None


def _set_response_text(response: ModelResponse, data: ModelResponse) -> None:
    """Replace the text content of a ModelResponse with serialized data."""
    text = json.dumps(data, ensure_ascii=False, default=str)
    for part in response.parts:
        if isinstance(part, TextPart):
            part.content = text
            return
    response.parts.append(TextPart(content=text))  # ty: ignore[unresolved-attribute]


def _build_delta_message(parsed: ModelResponse, schema: type[Any]) -> str | None:
    """Build a delta-retry message describing what needs fixing.

    Examines the parsed (but invalid) output against the schema and
    produces a message that tells the model exactly which parts are
    broken and what the expected structure looks like.
    """
    issues: list[str] = []

    # Check if the model returned a list when the schema expects an object.
    if isinstance(parsed, list) and hasattr(schema, "model_fields"):
        expected = _format_schema_fields(schema)
        issues.append(f"- You returned a JSON array, but the expected output is a single JSON object with fields {expected}. Return one object, not a list.")

    if isinstance(parsed, dict):
        # Check each field against the schema
        for field_name, field_info in schema.model_fields.items():
            if field_name not in parsed:
                issues.append(f"- missing field '{field_name}'")
                continue
            value = parsed[field_name]
            # Check if the value is a list with type errors
            if hasattr(field_info, "annotation"):
                inner_type = _get_list_inner_type(field_info.annotation)
                if inner_type and isinstance(value, list):
                    for i, item in enumerate(value):
                        if not isinstance(item, dict):
                            expected = _format_schema_fields(inner_type)
                            issues.append(
                                f"- {field_name}[{i}]: expected an object with fields {expected}, got {type(item).__name__} ({json.dumps(str(item))})"
                            )

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


def _get_list_inner_type(annotation: Any) -> type[Any] | None:
    """Extract the inner type from a ``list[X]`` annotation."""
    origin = typing.get_origin(annotation)
    if origin is not list:
        return None
    args = typing.get_args(annotation)
    return args[0] if args else None


def _format_schema_fields(model: type[Any]) -> str:
    """Format the field names of a Pydantic model for display."""
    if hasattr(model, "model_fields"):
        return "{" + ", ".join(model.model_fields.keys()) + "}"
    return "{...}"


def _merge_valid_data(previous: AgentResult, current: AgentResult, schema: type[Any]) -> AgentResult | None:
    """Merge valid data from a previous retry attempt with the current one.

    When the model returns fewer items on retry (e.g., only the corrected
    pages), merge them with the previously valid items by replacing the
    tail of the previous list with the new items.
    """
    if not isinstance(previous, dict) or not isinstance(current, dict):
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
