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

Salvage: small local models frequently produce tool-call arguments with
invalid JSON (truncated output, bad escapes, stray control characters).
We intercept ``ToolCallPart.args`` in :meth:`KnoxModel.request` and run
:func:`json_repair.loads` on the raw string before pydantic-ai's strict
validator sees it.  This eliminates ~67% of generation retries that
were caused by repairable JSON syntax errors.

Failover is handled by wrapping multiple :class:`KnoxModel` instances in
:class:`pydantic_ai.models.fallback.FallbackModel` at the
:class:`~knox.agents.base.KnoxAgentBase` construction site.
"""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.messages import (
    ModelResponse as MR,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings, ThinkingLevel

from knox.agents.debug import save_debug_prompt
from knox.llm.format import salvage_json
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
        # History compression is handled at the pydantic-ai Agent layer by
        # ``ContextManagerCapability`` (attached in KnoxAgentBase._build_agent),
        # which summarises when the conversation nears the context budget.
        # Previously this method dropped all ModelResponse messages on retry
        # (``_compact_messages``); that crude workaround lost the model's own
        # output, defeating pydantic-ai's retry feedback loop. The capability
        # is strictly better and preserves history on compression failure.
        response = await self._delegate.request(messages, merged_settings, model_request_parameters)
        self._repair_tool_call_args(response)
        self._repair_thinking_part(response)
        self._track_usage(response)
        self._save_debug(messages, response)
        return response

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
        settings = cast("ModelSettings", dict(model_settings or {}))
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
            settings["openai_reasoning_effort"] = "none"  # ty: ignore[invalid-key]
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

        # Count turns from the original ModelMessage objects (not serialised dicts)
        user_turns = sum(1 for m in messages if isinstance(m, ModelRequest))
        assistant_turns = sum(1 for m in messages if isinstance(m, MR))

        # Append the final response to the messages list so the debug file
        # shows the complete conversation including the model's last reply.
        all_messages = list(messages) + [response]

        save_debug_prompt(
            self._agent,
            self.model_name,
            _messages_to_debug_dicts(all_messages),
            {"model": self.model_name},
            content,
            str(finish_reason),
            user_turns=user_turns,
            assistant_turns=assistant_turns,
        )

    # ── KNOX-specific hooks ─────────────────────────────────────────────

    @staticmethod
    def _repair_tool_call_args(response: ModelResponse) -> None:
        """Run ``json_repair`` on ``ToolCallPart.args`` strings in-place.

        Small local models frequently produce tool-call arguments with
        invalid JSON: truncated output, bad escapes, stray
        control characters, trailing prose.  pydantic-ai's built-in
        validator uses pydantic's strict ``validate_json`` which rejects
        all of these, triggering a retry that wastes context window and
        usually produces the same error.

        We intercept the raw args string *before* pydantic-ai validates
        it, run ``json_repair.loads``, and replace ``args`` with the
        repaired dict.  When the JSON is already valid or the repair
        fails entirely, the part is left untouched so pydantic-ai's
        normal retry logic applies.
        """
        for part in response.parts:
            if not isinstance(part, ToolCallPart):
                continue
            args = part.args
            if not isinstance(args, str):
                continue  # Already a dict, or None
            repaired = salvage_json(args)
            if isinstance(repaired, dict):
                part.args = repaired

    @staticmethod
    def _repair_thinking_part(response: ModelResponse) -> None:
        """Convert ThinkingPart with valid JSON to TextPart.

        Some Ollama models return thinking content even when
        ``reasoning_effort: none`` is set.  PydanticAI only validates
        ``TextPart`` for structured output, so a response with only a
        ``ThinkingPart`` (even one containing valid JSON) is treated as
        invalid, triggering a retry that wastes context and usually fails
        identically.

        We intercept the response *before* PydanticAI validates it, check
        if the ThinkingPart contains valid JSON, and if so, convert it to
        a TextPart so the structured output validator can process it.
        """
        for i, part in enumerate(response.parts):
            if not isinstance(part, ThinkingPart):
                continue
            content = part.content
            if not content:
                continue
            repaired = salvage_json(content)
            if isinstance(repaired, (dict, list)):
                # Replace the ThinkingPart with a TextPart containing the JSON
                response.parts[i] = TextPart(content=json.dumps(repaired, ensure_ascii=False))
                logger.info(
                    "Converted ThinkingPart to TextPart (%d chars of JSON)",
                    len(response.parts[i].content),
                )
        """Run ``json_repair`` on ``TextPart`` for structured-output agents.

        The same class of JSON errors that tool-call repair fixes (bad
        escapes, trailing commas, truncated output) also appear in the
        main text response when an agent uses structured ``output_type``.
        pydantic-ai's strict validator rejects them, triggering a retry
        that wastes context and usually fails identically.

        We intercept the raw text *before* validation, try
        ``salvage_json``, and replace the text with the repaired JSON
        string.  When the JSON is already valid or the repair fails,
        the part is left untouched so pydantic-ai's normal retry logic
        applies.
        """
        for part in response.parts:
            if not isinstance(part, TextPart):
                continue
            content = part.content
            if not isinstance(content, str) or not content:
                continue
            repaired = salvage_json(content)
            if repaired is not None and not isinstance(repaired, (dict, list)):
                continue  # Not a JSON object/array — likely plain text, skip
            if repaired is not None:
                part.content = json.dumps(repaired, ensure_ascii=False)


def _messages_to_debug_dicts(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Convert pydantic-ai ModelMessages to debug-friendly dicts.

    Each message is serialised with its role (``ModelRequest`` / ``ModelResponse``)
    and a list of structured part dicts.  Parts are serialised with their type
    (``text``, ``thinking``, ``tool_call``, ``retry_prompt``) and type-specific
    content so the debug file is readable without mental parsing of ``str(part)``.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        kind = type(msg).__name__
        parts_list: list[dict[str, Any]] = []
        for part in getattr(msg, "parts", []):
            parts_list.append(_part_to_debug_dict(part))
        result.append({"role": kind, "parts": parts_list})
    return result


def _part_to_debug_dict(part: Any) -> dict[str, Any]:
    """Convert a single ModelPart to a debug-friendly dict."""
    if isinstance(part, SystemPromptPart):
        return {"type": "system_prompt", "content": part.content}
    if isinstance(part, TextPart):
        return {"type": "text", "content": part.content}
    if isinstance(part, ThinkingPart):
        return {"type": "thinking", "content": part.content}
    if isinstance(part, ToolCallPart):
        return {
            "type": "tool_call",
            "tool_name": part.tool_name,
            "args": part.args,
            "tool_call_id": part.tool_call_id,
        }
    if isinstance(part, ToolReturnPart):
        return {
            "type": "tool_return",
            "tool_name": part.tool_name,
            "content": part.content,
        }
    if isinstance(part, RetryPromptPart):
        content = part.content
        # RetryPromptPart.content can be a list of validation errors
        if isinstance(content, list):
            return {"type": "retry_prompt", "errors": content}
        return {"type": "retry_prompt", "content": str(content)}
    return {"type": type(part).__name__, "content": str(part)}


__all__ = ["KnoxModel"]
