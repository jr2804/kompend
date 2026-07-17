"""KnoxModel — thin wrapper adding KNOX concerns to any native pydantic-ai Model.

Wraps a native model returned by :func:`knox.llm.models.build_model` and adds:

- **Usage tracking** — token counts from each response feed the pipeline's
  :class:`~knox.llm.usage.UsageTracker` so callers (WiCER, lint) can report
  aggregate usage.
- **Debug saving** — when ``KNOX_DEBUG_LLM_DIR`` is set, every LLM call is
  dumped to disk with the message list and response.
- **Thinking** — the configured ``reasoning_effort`` string is mapped to
  pydantic-ai's unified ``thinking`` ``model_settings`` key and forwarded
  to the delegate. See
  https://pydantic.dev/docs/ai/advanced-features/thinking/.

Salvage and retry enrichment are intentionally NOT reimplemented here:
native models use ToolOutput mode for structured output, which validates
tool-call arguments directly rather than parsing JSON text, so there is no
malformed-JSON text to repair. For models that support NativeOutput
(schema-constrained decoding), output is valid by construction.

Failover is handled by wrapping multiple :class:`KnoxModel` instances in
:class:`pydantic_ai.models.fallback.FallbackModel` at the
:class:`~knox.agents.base.KnoxAgentBase` construction site.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings, ThinkingLevel

from knox.agents.debug import save_debug_prompt
from knox.llm.usage import UsageTracker
from knox.logging import get_logger

# KNOX config ``reasoning_effort`` string → pydantic-ai unified ``thinking``.
# ``none`` disables thinking entirely (``False``); the five named levels map
# to the corresponding ``ThinkingLevel`` literals.
_EFFORT_TO_THINKING: dict[str, ThinkingLevel] = {
    "none": False,
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}

logger = get_logger(__name__)


class KnoxModel(Model):
    """Wraps a native pydantic-ai Model with KNOX-specific concerns."""

    def __init__(
        self,
        delegate: Model,
        *,
        usage_tracker: UsageTracker | None = None,
        agent: Any | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._delegate = delegate
        self._usage_tracker = usage_tracker
        self._agent = agent
        self._reasoning_effort = reasoning_effort

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
        # Compact history on retry: drop all ModelResponse (assistant
        # attempts) to prevent the context window from growing on each
        # validation retry. Only ModelRequest (system + user + retry
        # prompts) carry the instructions the model needs — its own
        # previous output is irrelevant and wastes context.
        compacted = _compact_messages(messages)
        if len(compacted) < len(messages):
            logger.info(
                "Compacted message history: %d → %d messages (dropped %d assistant responses)", len(messages), len(compacted), len(messages) - len(compacted)
            )
        response = await self._delegate.request(compacted, merged_settings, model_request_parameters)
        self._track_usage(response)
        self._save_debug(messages, response)
        return response

    # ── KNOX-specific hooks ─────────────────────────────────────────────

    def _merge_reasoning(self, model_settings: ModelSettings | None) -> ModelSettings | None:
        """Map the configured ``reasoning_effort`` to the unified ``thinking`` setting.

        pydantic-ai's unified ``thinking`` key covers every provider's
        reasoning/thinking knob. ``"none"`` → ``False`` (disabled); every
        other level passes through as-is. See
        https://pydantic.dev/docs/ai/advanced-features/thinking/.
        """
        if self._reasoning_effort is None:
            return model_settings
        thinking = _EFFORT_TO_THINKING.get(self._reasoning_effort)
        if thinking is None:
            logger.warning(
                "Unknown reasoning_effort %r — ignoring (use none/minimal/low/medium/high/xhigh)",
                self._reasoning_effort,
            )
            return model_settings
        settings: ModelSettings = dict(model_settings or {})
        settings.setdefault("thinking", thinking)
        return settings

    def _track_usage(self, response: ModelResponse) -> None:
        """Record token usage from the response onto the UsageTracker."""
        if self._usage_tracker is None:
            return
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self._usage_tracker.record(
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=self.model_name,
        )

    def _save_debug(self, messages: list[ModelMessage], response: ModelResponse) -> None:
        """Dump the call to disk when KNOX_DEBUG_LLM_DIR is set."""
        if self._agent is None:
            return
        content = ""
        for part in response.parts:
            if isinstance(part, TextPart):
                content = part.content
                break
        finish_reason = getattr(response, "finish_reason", None) or "stop"
        # save_debug_prompt expects message-dict message dicts; pydantic-ai
        # ModelMessage objects serialize via their to_dict. We pass a minimal
        # payload — the debug file is for manual inspection, not replay.
        save_debug_prompt(
            self._agent,
            self.model_name,
            _messages_to_debug_dicts(messages),
            {"model": self.model_name},
            content,
            str(finish_reason),
        )


def _compact_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop all ModelResponse messages to prevent context window blowup on retry.

    Each retry iteration adds the model's full previous attempt (a
    ModelResponse) and a retry prompt (a ModelRequest). On a small model
    with limited context, a single retry can exceed the budget.

    ModelResponse carries the model's OWN previous output — irrelevant
    for the next generation. Only the ModelRequest instructions (system
    prompt, user prompt, retry instructions) matter.
    """
    return [m for m in messages if isinstance(m, ModelRequest)]


def _messages_to_debug_dicts(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Convert pydantic-ai ModelMessages to debug-friendly dicts."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        kind = type(msg).__name__
        parts_text: list[str] = []
        for part in getattr(msg, "parts", []):
            if isinstance(part, TextPart):
                parts_text.append(part.content)
            else:
                parts_text.append(str(part))
        result.append({"role": kind, "content": "\n".join(parts_text)})
    return result


__all__ = ["KnoxModel"]
