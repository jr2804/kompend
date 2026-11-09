"""KompendModel — thin wrapper adding kompend concerns to any native pydantic-ai Model.

Wraps a native model returned by :func:`kompend.llm.models.build_model` and adds:

- **Usage tracking** — token counts from each response feed the pipeline's
  :class:`~kompend.llm.usage.UsageTracker` so callers (WiCER, lint) can report
  aggregate usage.
- **Debug saving** — when ``KOMP_DEBUG_LLM_DIR`` is set, every LLM call is
  dumped to disk with the message list and response.
- **Thinking** — the configured ``reasoning_effort`` member is mapped to
  pydantic-ai's unified ``thinking`` ``model_settings`` key and forwarded
  to the delegate. See
  https://pydantic.dev/docs/ai/advanced-features/thinking/.

Salvage: small local models frequently produce tool-call arguments with
invalid JSON (truncated output, bad escapes, stray control characters).
We intercept ``ToolCallPart.args`` in :meth:`KompendModel.request` and run
:func:`json_repair.loads` on the raw string before pydantic-ai's strict
validator sees it.  This eliminates ~67% of generation retries that
were caused by repairable JSON syntax errors.

Failover is handled by wrapping multiple :class:`KompendModel` instances in
:class:`pydantic_ai.models.fallback.FallbackModel` at the
:class:`~kompend.agents.base.KompendAgentBase` construction site.
"""

from __future__ import annotations

from typing import Any, Final, cast

from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings, ThinkingLevel

from kompend.agents.failure import EmptyAnswerError
from kompend.core.config import ReasoningEffort
from kompend.llm.debug import _DebugSource
from kompend.llm.postprocess import default_pipeline
from kompend.llm.usage import UsageTracker
from kompend.logging import get_logger

# kompend config ``reasoning_effort`` → pydantic-ai unified ``thinking``.
# ``none`` disables thinking entirely (``False``); the five named levels map
# to the corresponding ``ThinkingLevel`` literals. Keys are the
# :class:`~kompend.core.config.ReasoningEffort` members validated at the
# config schema, so a typo'd value never reaches this lookup. ``DEFAULT`` is
# deliberately absent: it means "leave the provider's default alone" and is
# returned early, never looked up.
_EFFORT_TO_THINKING: dict[ReasoningEffort, ThinkingLevel] = {
    ReasoningEffort.NONE: False,
    ReasoningEffort.MINIMAL: "minimal",
    ReasoningEffort.LOW: "low",
    ReasoningEffort.MEDIUM: "medium",
    ReasoningEffort.HIGH: "high",
    ReasoningEffort.XHIGH: "xhigh",
}


__all__ = ["KompendModel"]


#: pydantic-ai's output mode for a plain-text request. Every other mode
#: (``tool`` = prompted structured output, ``native`` = provider-native
#: structured output, ``auto`` = the first request of either) is asking for a
#: typed object, so an empty answer there is a failure rather than the caller's
#: business. A text request is the one case where "nothing" is an answer the
#: caller interprets itself (a summariser's empty summary, a tool-using agent's
#: plain reply).
_TEXT_OUTPUT_MODE: Final[str] = "text"

#: How often a blank structured-output answer is resampled before it escalates.
#: Measured on a local 4-bit model (ollama, 2026-11-09): one extraction prompt
#: answered nothing in five of six identical calls and answered cleanly the next
#: time, so a blank answer is sampling noise rather than a request the model
#: cannot satisfy. One resample recovers it; a second blank escalates.
_EMPTY_ANSWER_RESAMPLES: Final[int] = 1

logger = get_logger(__name__)


class KompendModel(Model):
    """Wraps a native pydantic-ai Model with kompend-specific concerns."""

    def __init__(
        self,
        delegate: Model,
        *,
        usage_tracker: UsageTracker | None = None,
        agent: _DebugSource | None = None,
        reasoning_effort: ReasoningEffort | str | None = None,
    ) -> None:
        self._delegate = delegate
        self._usage_tracker = usage_tracker
        self._agent = agent
        # Raw strings (tests, direct construction) normalise here — once — so
        # the lookup below only ever sees a validated member. An unrecognised
        # value raises instead of being dropped with a warning: the caller
        # asked for a reasoning level that does not exist.
        self._reasoning_effort = None if reasoning_effort is None else ReasoningEffort(reasoning_effort)

        # Build the schema resolver closure for the post-processing pipeline.
        # Lazily resolves the pydantic-ai agent's result type so ToolCallRepair
        # and ThinkingPartRepair can guide JSON repair with the schema.

        def _schema_resolver() -> type[BaseModel] | None:
            if not self._agent:
                return None
            ot = getattr(self._agent, "output_type", None) or getattr(self._agent, "result_type", None)
            if ot is None or ot is str:
                return None
            return ot

        self._schema_resolver = _schema_resolver

    # ── Model ABC interface ─────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._delegate.model_name

    @property
    def system(self) -> str:
        return self._delegate.system

    @property
    def delegate(self) -> Model:
        """The wrapped native model (for inspection/testing)."""
        return self._delegate

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        merged_settings = self._merge_reasoning(model_settings)
        # History compression is handled at the pydantic-ai Agent layer by
        # ``SummarizingCompaction`` (attached in KompendAgentBase._build_agent),
        # which summarises when the conversation nears the context budget. Do
        # not drop ``ModelResponse`` messages here: that loses the model's own
        # output and defeats pydantic-ai's retry feedback loop.
        response = await self._delegate.request(messages, merged_settings, model_request_parameters)
        empty_resamples = _EMPTY_ANSWER_RESAMPLES
        while empty_resamples and _is_empty_answer(response, model_request_parameters):
            empty_resamples -= 1
            logger.warning(
                "%s answered nothing to a structured request; resampling (%d left)",
                self.model_name,
                empty_resamples,
            )
            response = await self._delegate.request(messages, merged_settings, model_request_parameters)
        _fail_on_empty_answer(response, model_request_parameters)

        pipeline = default_pipeline(
            schema_resolver=self._schema_resolver,
            agent=self._agent,
            model_name=self.model_name,
            usage_tracker=self._usage_tracker,
            messages=messages,
        )
        pipeline(response)
        return response

    def _merge_reasoning(self, model_settings: ModelSettings | None) -> ModelSettings | None:
        """Map the configured ``reasoning_effort`` to the unified ``thinking`` setting.

        pydantic-ai's unified ``thinking`` key covers every provider's
        reasoning/thinking knob. ``none`` → ``False`` (disabled); every
        other level passes through as-is. ``None`` (unset) and ``default``
        leave ``model_settings`` untouched so the provider's own default
        applies. See https://pydantic.dev/docs/ai/advanced-features/thinking/.
        """
        if self._reasoning_effort is None or self._reasoning_effort is ReasoningEffort.DEFAULT:
            return model_settings
        # Members other than DEFAULT all carry a mapping; a KeyError here is
        # impossible by construction and would be a vocabulary drift bug.
        thinking = _EFFORT_TO_THINKING[self._reasoning_effort]
        settings = cast("dict[str, Any]", dict(model_settings or {}))
        settings.setdefault("thinking", thinking)
        # pydantic-ai drops the unified ``thinking`` key when the model profile
        # lacks ``supports_thinking`` (true for most Ollama HF models), so the
        # value never reaches ``model_request_parameters.thinking`` and
        # ``_translate_thinking`` returns OMIT — the API call omits
        # ``reasoning_effort`` entirely.
        #
        # Fix: set ``openai_reasoning_effort`` directly in model_settings.
        # ``_translate_thinking`` checks this key *before* mrp.thinking, so it
        # bypasses the profile gate. Ollama's /v1/chat/completions endpoint
        # respects ``reasoning_effort: 'none'`` and disables thinking.
        if thinking is False:
            settings["openai_reasoning_effort"] = ReasoningEffort.NONE.value
        return cast("ModelSettings", settings)


def _fail_on_empty_answer(response: ModelResponse, request_parameters: ModelRequestParameters) -> None:
    """Raise :class:`EmptyAnswerError` when a structured-output request answered with nothing.

    The check sits in the model wrapper because this is the one place every
    request in kompend passes through. It fires after the resample in
    :meth:`KompendModel.request` and before pydantic-ai's output retry loop, so
    an empty answer costs at most two requests instead of
    ``compile.llm.output_retries`` / ``agentic_output_retries`` of them, and the
    stage receives :class:`EmptyAnswerError` instead of the opaque
    ``Exceeded maximum output retries (4)``. The retry feedback for an empty
    answer is "please return text or include your response in a tool call" — a
    model that just returned nothing does not fail because that feedback was
    missing, so the retries are pure budget burn. A plain resample does help
    (see :data:`_EMPTY_ANSWER_RESAMPLES`), which is why it happens here.

    Args:
        response: The delegate's response, after any JSON salvage.
        request_parameters: What the caller asked for.

    Raises:
        EmptyAnswerError: The request wanted a typed object and got nothing.
    """
    if not _is_empty_answer(response, request_parameters):
        return
    names = ", ".join(tool.name for tool in request_parameters.output_tools)
    expected = f" instead of calling {names}" if names else ""
    repeated = " again" if _EMPTY_ANSWER_RESAMPLES else ""
    raise EmptyAnswerError(
        f"The model returned an empty answer{expected}{repeated}. "
        "Retrying with feedback cannot change a blank answer, so the output retry budget was not spent."
    )


def _is_empty_answer(response: ModelResponse, request_parameters: ModelRequestParameters) -> bool:
    """True when a structured-output request came back with nothing usable.

    A response that made a tool call is never empty: it is the tool that has no
    answer yet. A response with non-blank text is never empty either — in
    prompted mode that text may be the answer itself when the caller allows it.
    A text request has no empty case: "nothing" is an answer the caller
    interprets itself (a summariser's empty summary, a tool-using agent's plain
    reply).

    Args:
        response: The delegate's response, after any JSON salvage.
        request_parameters: What the caller asked for; only its output mode
            decides whether an empty answer counts.

    Returns:
        True when the caller wanted a typed object and got nothing.
    """
    if request_parameters.output_mode == _TEXT_OUTPUT_MODE:
        return False
    if any(isinstance(part, ToolCallPart) for part in response.parts):
        return False
    return not any(isinstance(part, TextPart) and part.content.strip() for part in response.parts)
