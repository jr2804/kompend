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
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from knox.logging import get_logger

logger = get_logger(__name__)


def _messages_to_litellm(messages: list[ModelMessage]) -> list[dict[str, str]]:
    """Convert PydanticAI messages to litellm chat format."""
    result: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    result.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    result.append({"role": "user", "content": str(part.content)})
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    result.append({"role": "assistant", "content": part.content})
    return result


def _make_litellm_function(model_string: str) -> Any:
    """Create a FunctionModel-compatible function calling ``litellm.completion()``."""

    def _fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        litellm_messages = _messages_to_litellm(messages)
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
