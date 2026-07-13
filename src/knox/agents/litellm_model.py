"""Litellm-backed PydanticAI model bridge.

PydanticAI's built-in ``litellm:`` prefix creates an OpenAI client pointed
at a LiteLLM **proxy server** — it does NOT call ``litellm.completion()``
through litellm's Python routing. KNOX's custom providers (opencode-zen,
ollama_cloud, claudinio) depend on litellm's Python monkey-patching, so
they would break with the built-in integration.

This module creates a :class:`~pydantic_ai.models.function.FunctionModel`
backed by ``litellm.completion()``, preserving litellm's full dispatch chain:
custom providers, API key resolution, per-call kwargs, and all monkey-patches.

Usage::

    from knox.agents.litellm_model import make_litellm_model

    agent = Agent(
        make_litellm_model("ollama/gpt-oss:20b"),
        output_type=ExtractionOutput,
    )
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import litellm
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from knox.logging import get_logger

logger = get_logger(__name__)


def _messages_to_litellm(messages: list[ModelMessage]) -> list[dict[str, str]]:
    """Convert PydanticAI messages to litellm chat format.

    Handles all part types PydanticAI uses in the structured-output retry
    loop: ``RetryPromptPart`` (validation errors sent back on schema
    mismatch) and ``ToolReturnPart`` are converted to ``user`` messages so
    the model sees the feedback. Without this, retries are silently
    dropped and the model repeats the same invalid output.
    """
    result: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    result.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    result.append({"role": "user", "content": str(part.content)})
                elif isinstance(part, RetryPromptPart):
                    content = part.content
                    if isinstance(content, list):
                        content = "\n".join(f"- {e.get('msg', e)}" for e in content)
                    result.append(
                        {
                            "role": "user",
                            "content": f"Validation errors — fix and retry:\n{content}",
                        }
                    )
                elif isinstance(part, ToolReturnPart):
                    result.append(
                        {
                            "role": "user",
                            "content": str(part.content),
                        }
                    )
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    result.append({"role": "assistant", "content": part.content})
    return result


def _make_litellm_function(model_string: str) -> Any:
    """Create a FunctionModel-compatible function calling ``litellm.completion()``."""

    def _fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        litellm_messages = _messages_to_litellm(messages)

        # Inject instructions as system message. PydanticAI 2.9 passes the
        # system prompt via AgentInfo.instructions, NOT as a SystemPromptPart
        # in the message list. Without this, the system prompt is silently
        # discarded and the model never sees critical output-format guidance.
        if info.instructions:
            litellm_messages.insert(0, {"role": "system", "content": info.instructions})

        kwargs: dict[str, Any] = {"model": model_string, "messages": litellm_messages}

        # Forward model settings (timeout, max_tokens, temperature)
        if info.model_settings:
            for key in ("timeout", "max_tokens", "temperature"):
                val = info.model_settings.get(key)
                if val is not None:
                    kwargs[key] = val

        # Force JSON output for structured agents (Pydantic model output_type).
        # When allow_text_output is True (QueryAgent), the model returns free text.
        if not info.allow_text_output:
            kwargs["response_format"] = {"type": "json_object"}

        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason

        # Debug: log raw response anomalies for diagnosis
        if not content.strip() and not info.allow_text_output:
            logger.warning(
                "LitellmModel: empty response from '%s' (finish_reason=%s)",
                model_string,
                finish_reason,
            )
        elif not content.strip().startswith("{") and not info.allow_text_output:
            logger.warning(
                "LitellmModel: non-JSON response from '%s' (finish_reason=%s, first 200 chars): %s",
                model_string,
                finish_reason,
                content[:200],
            )
        elif finish_reason == "length" and not info.allow_text_output:
            logger.warning(
                "LitellmModel: truncated response from '%s' (finish_reason=length, %d chars). Increase max_tokens or reduce input size.",
                model_string,
                len(content),
            )

        return ModelResponse(
            parts=[TextPart(content=content)],
            model_name=model_string,
            timestamp=datetime.now(),
        )

    return _fn


def make_litellm_model(model_string: str) -> FunctionModel:
    """Create a PydanticAI :class:`FunctionModel` backed by ``litellm.completion()``.

    This preserves litellm's full provider dispatch chain — custom providers,
    API key resolution, monkey-patches, and per-call kwargs all work unchanged.
    """
    return FunctionModel(
        function=_make_litellm_function(model_string),
        model_name=model_string,
    )
