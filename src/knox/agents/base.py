"""KnoxAgent base class — KNOX-side wrapper around pydantic_ai.Agent.

Every concrete agent (extraction, generation, query, wicer) inherits
from this base and supplies:

- ``template_name`` — the Jinja2 template under
  ``templates/<dir>/<name>.md.j2``
- ``system_template_name`` — the system prompt template name
- ``result_type`` — the Pydantic model (or ``str``) the LLM must return
- ``deps_type`` — the typed context dataclass

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
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from pydantic_ai import Agent, NativeOutput
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart
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
from knox.core.templates import get_template_loader
from knox.core.utils.exceptions import LLMOutputError
from knox.llm.format import salvage_json
from knox.llm.models import build_model
from knox.llm.usage import UsageTracker
from knox.logging import get_logger
from knox.types import AgentResult

logger = get_logger(__name__)

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

    - ``template_name`` — the filename under ``templates/<prefix>/``
      where ``prefix`` is set by ``template_prefix`` (default ``compilation/``).
    - ``system_template_name`` — the filename of the system prompt.
    - ``result_type`` — a Pydantic model class (or ``str`` for free-text
      output) that the LLM output will be validated against.

    Subclasses MUST implement :meth:`_build_agent`.
    """

    template_name: str
    system_template_name: str
    #: Subdirectory under ``templates/`` where this agent's templates live.
    #: Override in subclasses whose templates are not under ``compilation/``
    #: (e.g. ``"search/"``, ``"capture/"``, ``"organize/"``).
    template_prefix: str = "compilation/"
    result_type: type
    deps_type: type[AgentDeps] = PipelineAgentDeps
    temperature: float | None = None

    #: Optional output validator function. Subclasses can set this to a
    #: callable that receives ``(ctx, output)`` and returns the (possibly
    #: modified) output, raising ``ModelRetry`` to trigger a retry with
    #: actionable feedback.  The callable signature matches
    #: ``@agent.output_validator`` — it can be sync or async, with or
    #: without ``RunContext``.
    output_validator_func: Any | None = None

    @property
    def stage(self) -> str:
        """Pipeline stage label recorded in the LLM debug dump.

        Derived from the agent class name (e.g. ``ExtractionAgent`` ->
        ``extraction``), so no per-subclass attribute is needed.
        """
        return self.__class__.__name__.removesuffix("Agent").lower()

    #: Enable delta-retry: when the model returns partially valid structured
    #: output, extract the valid parts and ask the model to fix only the
    #: broken ones, instead of regenerating everything from scratch.
    use_delta_retry: bool = False

    def __init__(
        self,
        model_string: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        retries: int = _DEFAULT_RETRIES,
        max_tokens: int | None = None,
        skills: dict[str, str] | None = None,
        usage_tracker: UsageTracker | None = None,
        fallback_models: list[str] | None = None,
        reasoning_effort: str | None = None,
        use_native_output: bool = False,
        context_manager: str | bool = False,
        context_window: int | None = None,
    ) -> None:
        self._model_string = model_string
        self._timeout = timeout
        self._retries = retries
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._usage_tracker = usage_tracker
        self._fallback_models = fallback_models
        self._use_native_output = use_native_output
        self._context_manager = context_manager
        self._context_window = context_window
        # User-defined skills (skill_name -> instructions, injected into system prompt)
        self._skills: dict[str, str] = skills or {}
        # The PydanticAI Agent is constructed lazily (on first run() call)
        # so tests can construct agents without a live provider.
        self._agent: Agent[DepsT, ResultT] | None = None
        # The template loader is available for render_system_prompt
        # and render_user_prompt, which are called at run() time.
        self._template_loader = get_template_loader()
        # Current source_id for debug logging (set per run() call)
        self._current_source_id: str = ""
        #: Accumulated valid data across delta-retry attempts, keyed by run_step.
        self._delta_retry_state: dict[int, Any] = {}
        #: Path to the last debug file written by KnoxModel (for error messages).
        self._last_debug_path: Path | None = None

    def render_user_prompt(self, deps: DepsT) -> str:
        """Render the user-prompt template with the agent's deps.

        Only variables declared in the template's frontmatter are passed,
        avoiding warnings about unused variables.
        """
        template_path = f"{self.template_prefix}{self.template_name}"
        meta = self._template_loader.get_meta(template_path)
        declared = meta.declared_variable_names()
        vars_dict = self._deps_to_template_vars(deps)
        filtered = {k: v for k, v in vars_dict.items() if k in declared}
        return self._template_loader.render(template_path, **filtered)

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
        if self._skills:
            skill_blocks = []
            for name, instructions in self._skills.items():
                skill_blocks.append(f"## Skill: {name}\n\n{instructions}")
            prompt += "\n\n" + "\n\n---\n\n".join(skill_blocks)
        return prompt

    def run(
        self,
        deps: DepsT,
        *,
        message_history: list[Any] | None = None,
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
            )
        except Exception as exc:
            debug_hint = f" (debug: {self._last_debug_path})" if self._last_debug_path else ""
            logger.warning(
                "Agent '%s' failed on '%s': %s%s",
                type(self).__name__,
                deps.source_id,
                exc,
                debug_hint,
            )
            raise LLMOutputError(f"Agent failed for model '{self._model_string}': {exc}{debug_hint}") from exc
        return result.output

    def run_with_messages(
        self,
        deps: DepsT,
        *,
        message_history: list[Any] | None = None,
    ) -> tuple[ResultT, list[Any]]:
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
            )
        except Exception as exc:
            debug_hint = f" (debug: {self._last_debug_path})" if self._last_debug_path else ""
            logger.warning(
                "Agent '%s' failed on '%s': %s%s",
                type(self).__name__,
                deps.source_id,
                exc,
                debug_hint,
            )
            raise LLMOutputError(f"Agent failed for model '{self._model_string}': {exc}{debug_hint}") from exc
        return result.output, result.new_messages()

    @property
    def model_string(self) -> str:
        """The KNOX-format model string (``provider/model``)."""
        return self._model_string

    @property
    def usage(self) -> UsageTracker | None:
        """The usage tracker, if one was provided."""
        return self._usage_tracker

    @property
    def current_source_id(self) -> str:
        """The source_id from the most recent ``run()`` call."""
        return self._current_source_id

    def _build_agent(self) -> Agent[DepsT, ResultT]:
        """Construct the wrapped PydanticAI agent. Called lazily on first run().

        Uses :func:`knox.llm.models.build_model` to construct a native
        pydantic-ai Model from the configured model string, wraps it in
        :class:`~knox.agents.knox_model.KnoxModel` for usage tracking and
        debug saving, and in :class:`~pydantic_ai.models.fallback.FallbackModel`
        when fallback models are configured.
        """
        model_settings: ModelSettings = {"timeout": self._timeout}
        if self.temperature is not None:
            model_settings["temperature"] = self.temperature
        if self._max_tokens is not None:
            model_settings["max_tokens"] = self._max_tokens

        output_type: type | NativeOutput = self.result_type
        if self._use_native_output:
            output_type = NativeOutput(
                self.result_type,
                name=self.stage,
                description=f"Structured output for {self.stage}",
            )

        # Build the top-level model once and reuse it for both the Agent and
        # the summarization capability (same failover chain).
        top_model = self._build_top_level_model()
        capability = self._resolve_context_capability(top_model)
        capabilities: list[AbstractCapability[Any]] = []
        if capability is not None:
            capabilities.append(capability)
        if self.use_delta_retry:
            capabilities.append(self._make_delta_retry_hook())
        agent = Agent(
            top_model,
            deps_type=self.deps_type,
            output_type=output_type,
            model_settings=model_settings,
            retries=self._retries,
            capabilities=capabilities or None,
        )
        if self.output_validator_func is not None:
            agent.output_validator(self.output_validator_func)
        return cast("Agent[DepsT, ResultT]", agent)

    def _build_top_level_model(self) -> Model:
        """Build the model (with failover + KNOX wrapper) for the Agent."""
        primary = KnoxModel(
            build_model(self._model_string),
            usage_tracker=self._usage_tracker,
            agent=self,
            reasoning_effort=self._reasoning_effort,
        )
        if not self._fallback_models:
            return primary

        return FallbackModel(
            primary,
            *(
                KnoxModel(
                    build_model(m),
                    usage_tracker=self._usage_tracker,
                    agent=self,
                    reasoning_effort=self._reasoning_effort,
                )
                for m in self._fallback_models
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
            ctx: Any,
            *,
            request_context: Any,
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

    def _resolve_context_capability(self, model: Model) -> AbstractCapability[Any] | None:
        """Build the context-management capability from the configured preset.

        Returns ``None`` when context management is disabled, in which case the
        pydantic-ai Agent is constructed without ``capabilities``.

        Args:
            model: The top-level (failover) model — reused as the
                summarization model so compressions run on the same chain.
        """
        setting = self._context_manager
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
            max_tokens=self._context_window,
            compress_threshold=threshold,
            keep=keep,
            summarization_model=model,
            on_usage_update=self._on_usage_update,
            on_before_compress=self._on_before_compress,
            on_after_compress=self._on_after_compress,
        )

    # ── ContextManagerCapability callbacks ─────────────────────────────

    def _on_usage_update(self, pct: float, current: int, max_tokens: int) -> None:
        """Token-usage snapshot from the capability → UsageTracker."""
        if self._usage_tracker is not None:
            self._usage_tracker.record_context_usage(current, max_tokens)

    @staticmethod
    def _on_before_compress(messages: list[Any], cutoff_index: int) -> None:
        """Fires between plan and execute — log the impending compression."""
        logger.info(
            "Context compression starting: %d messages, cutoff at index %d",
            len(messages),
            cutoff_index,
        )

    @staticmethod
    def _on_after_compress(messages: list[Any], summarized: bool, summary: str | None) -> None:
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


def _set_response_text(response: ModelResponse, data: Any) -> None:
    """Replace the text content of a ModelResponse with serialized data."""
    text = json.dumps(data, ensure_ascii=False, default=str)
    for part in response.parts:
        if isinstance(part, TextPart):
            part.content = text
            return
    response.parts.append(TextPart(content=text))  # ty: ignore[unresolved-attribute]


def _build_delta_message(parsed: Any, schema: type[Any]) -> str | None:
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
        "The following parts of your output need correction:\n"
        f"{issue_text}\n\n"
        "Return the COMPLETE corrected output. Keep all valid parts unchanged "
        "and fix only the listed issues."
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
