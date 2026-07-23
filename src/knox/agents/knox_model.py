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
from knox.types import AgentResult

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

        filepath = save_debug_prompt(
            self._agent,
            self.model_name,
            _messages_to_debug_dicts(all_messages),
            {"model": self.model_name},  # nocheck: debug log dict, not wiki metadata
            content,
            str(finish_reason),
            user_turns=user_turns,
            assistant_turns=assistant_turns,
        )
        # Store the debug path on the agent so error messages can reference it.
        if filepath is not None:
            self._agent._last_debug_path = filepath

    def _repair_thinking_part(self, response: ModelResponse) -> None:
        """Convert ThinkingPart/TextPart with valid JSON to ToolCallPart.

        Small Ollama models frequently return structured output as raw text
        (``TextPart``) instead of a tool call (``ToolCallPart``). PydanticAI's
        ``AutoOutputSchema`` has two validation paths:

        - **Tool path:** receives ``ToolCallPart.args`` as a dict, calls
          ``validate_python(dict)`` — always works.
        - **Text path:** receives ``TextPart.content`` as a string, calls
          ``validate_json(string)`` — fails on double-encoded JSON because
          ``validate_json`` parses the outer string literal and gets a Python
          ``str``, not a ``dict``, producing ``Input should be an object``.

        We intercept the response *before* PydanticAI validates it, parse the
        text content via ``salvage_json``, and convert the ``TextPart`` to a
        ``ToolCallPart`` with the parsed dict as ``args``. This forces the tool
        validation path, which always works correctly.

        Only applies to agents with structured output (``output_type`` is not
        ``str``). Free-text agents keep the original ``TextPart``.
        """
        # Check if this agent uses structured output (not free-text str)
        schema = self._get_result_schema()
        is_structured = schema is not None

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

        # If the response already has a ToolCallPart, the model used the tool
        # path correctly — no need to convert TextParts.
        has_tool_call = any(isinstance(p, ToolCallPart) for p in response.parts)

        for i, part in enumerate(response.parts):
            if not isinstance(part, TextPart):
                continue
            content = part.content
            if not isinstance(content, str) or not content:
                continue
            repaired = salvage_json(content)
            if repaired is None:
                continue
            if not isinstance(repaired, (dict, list)):
                continue  # Not a JSON object/array — likely plain text, skip

            if is_structured and not has_tool_call and isinstance(repaired, dict):
                # Convert TextPart → ToolCallPart to force the tool validation
                # path. This avoids the text-path ``validate_json`` failure on
                # double-encoded JSON strings.
                response.parts[i] = ToolCallPart(
                    tool_name="final_result",
                    args=repaired,
                )
                logger.info(
                    "Converted TextPart to ToolCallPart (final_result, %d keys)",
                    len(repaired),
                )
            else:
                # Free-text agent or tool call already present: just repair
                # the JSON text in-place.
                part.content = json.dumps(repaired, ensure_ascii=False)
                logger.info(
                    "Repaired TextPart content via salvage_json (%d chars -> %d chars)",
                    len(content),
                    len(part.content),
                )

    # ── KNOX-specific hooks ─────────────────────────────────────────────

    def _repair_tool_call_args(self, response: ModelResponse) -> None:
        """Run ``json_repair`` schema-guided repair on tool-call args.

        Small local models produce tool-call arguments with invalid JSON
        (truncation, bad escapes, stray control chars) and type mismatches
        (``"related": ""`` instead of ``[]``, garbled string fragments as
        page dicts).  pydantic-ai's built-in validator uses pydantic's strict
        ``validate_json`` which rejects all of these, triggering a retry
        that wastes context window and usually produces worse output.

        We intercept the raw args *before* pydantic-ai validates them.
        Both string args and dict args (already parsed by the provider)
        go through ``salvage_json`` with the agent's result schema for
        json_repair schema-guided repair + type coercion.  When the schema
        is unknown (free-text agents) or repair fails, the part is left
        untouched so pydantic-ai's normal retry logic applies.
        """
        schema = self._get_result_schema()
        for part in response.parts:
            if not isinstance(part, ToolCallPart):
                continue
            args = part.args
            if isinstance(args, str):
                repaired = salvage_json(args, schema=schema)
                if isinstance(repaired, dict):
                    part.args = repaired
            elif isinstance(args, dict) and schema is not None:
                # Dict args (already parsed by provider) — round-trip through
                # json_repair schema mode for type coercion.
                repaired = salvage_json(json.dumps(args, ensure_ascii=False), schema=schema)
                if isinstance(repaired, dict):
                    part.args = repaired

    def _get_result_schema(self) -> AgentResult | None:
        """Return the agent's structured-output schema, or ``None`` for free-text.

        Free-text agents (``output_type is str``) have no schema to
        repair against — their output is plain text.
        """
        if not self._agent:
            return None
        # pydantic-ai 2.x uses ``output_type`` (renamed from ``result_type``
        # in 1.x).  Check both for safety across versions.
        ot = getattr(self._agent, "output_type", None) or getattr(self._agent, "result_type", None)
        if ot is None or ot is str:
            return None
        return ot


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
