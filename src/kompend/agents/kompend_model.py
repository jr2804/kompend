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

from typing import cast

from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.settings import ModelSettings, ThinkingLevel

from kompend.core.config import ReasoningEffort
from kompend.llm.debug import _DebugSource
from kompend.llm.postprocess import default_pipeline
from kompend.llm.usage import UsageTracker

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
        # which summarises when the conversation nears the context budget.
        # Previously this method dropped all ModelResponse messages on retry
        # (``_compact_messages``); that crude workaround lost the model's own
        # output, defeating pydantic-ai's retry feedback loop. The capability
        # is strictly better and preserves history on compression failure.
        response = await self._delegate.request(messages, merged_settings, model_request_parameters)

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
        settings = cast("OpenAIChatModelSettings", dict(model_settings or {}))
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
        return settings
