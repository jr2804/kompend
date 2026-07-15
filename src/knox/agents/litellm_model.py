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

import json
from datetime import datetime
from typing import Any

import litellm
from pydantic import BaseModel
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

from knox.agents.debug import save_debug_prompt
from knox.core.utils.exceptions import shorten_string
from knox.llm.failover import FailoverChain
from knox.llm.format import salvage_json
from knox.llm.usage import UsageTracker
from knox.logging import get_logger

logger = get_logger(__name__)


def _messages_to_litellm(
    messages: list[ModelMessage],
    *,
    schema: type[BaseModel] | None = None,
) -> list[dict[str, Any]]:
    """Convert PydanticAI messages to litellm chat format.

    Handles all part types PydanticAI uses in the structured-output retry
    loop: ``RetryPromptPart`` (validation errors sent back on schema
    mismatch) and ``ToolReturnPart`` are converted to ``user`` messages so
    the model sees the feedback. Without this, retries are silently
    dropped and the model repeats the same invalid output.

    When *schema* is provided, ``RetryPromptPart`` content is enriched with
    the valid portions of the model's previous output (extracted via
    schema-aware ``salvage_json``), so the model only needs to fix the
    broken parts rather than redo everything.
    """
    result: list[dict[str, Any]] = []
    _last_assistant_content: str | None = None
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    result.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    result.append({"role": "user", "content": str(part.content)})
                elif isinstance(part, RetryPromptPart):
                    content = _enrich_retry_content(
                        part.content,
                        _last_assistant_content,
                        schema,
                    )
                    result.append({"role": "user", "content": content})
                elif isinstance(part, ToolReturnPart):
                    result.append({"role": "user", "content": str(part.content)})
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    _last_assistant_content = part.content
                    result.append({"role": "assistant", "content": part.content})
    return result


def _enrich_retry_content(
    retry_content: object,
    last_assistant_content: str | None,
    schema: type[BaseModel] | None,
) -> str:
    """Enrich PydanticAI's retry message with schema-aware feedback.

    PydanticAI's default retry messages are generic (``"Input should be a
    valid string"``). When *schema* and *last_assistant_content* are
    available, this function extracts the valid portions of the model's
    previous output via ``salvage_json`` and includes them in the retry
    message, so the model only needs to fix the broken parts.
    """
    errors = "\n".join(f"- {e.get('msg', e)}" for e in retry_content) if isinstance(retry_content, list) else str(retry_content)

    if schema is not None and last_assistant_content:
        repaired = salvage_json(last_assistant_content, schema=schema)
        if repaired is not None:
            valid_json = json.dumps(repaired, ensure_ascii=False, indent=2)
            if len(valid_json) > 3000:
                valid_json = valid_json[:3000] + "\n  ... (truncated)"
            return (
                f"Validation errors — fix and retry:\n{errors}\n\n"
                f"The valid parts extracted from your previous response:\n"
                f"```json\n{valid_json}\n```\n\n"
                f"Return ONLY a corrected JSON object that fixes the listed "
                f"errors. Keep all valid fields unchanged."
            )

    return f"Validation errors — fix and retry:\n{errors}"


def _make_litellm_function(  # noqa: C901, PLR0915
    model_string: str,
    usage_tracker: UsageTracker | None = None,
    fallback_models: list[str] | None = None,
    agent: Any | None = None,
    reasoning_effort: str | None = None,
    schema: type[BaseModel] | None = None,
) -> Any:
    """Create a FunctionModel-compatible function calling ``litellm.completion()``.

    When *reasoning_effort* is set, it is forwarded to litellm as
    ``reasoning_effort=`` (for cloud providers like OpenAI/Anthropic) and
    mapped to ``think=False`` for Ollama models that support the think toggle.
    """

    def _fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # noqa: C901, PLR0915
        litellm_messages = _messages_to_litellm(messages, schema=schema)

        # Inject instructions as system message. PydanticAI 2.9 passes the
        # system prompt via AgentInfo.instructions, NOT as a SystemPromptPart
        # in the message list. Without this, the system prompt is silently
        # discarded and the model never sees critical output-format guidance.
        if info.instructions:
            litellm_messages.insert(0, {"role": "system", "content": info.instructions})

        # Build the model list for failover: primary + fallbacks
        all_models = [model_string] + (fallback_models or [])

        # Build kwargs once (same for all models in the chain)
        kwargs: dict[str, Any] = {"messages": litellm_messages}

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

        # Apply reasoning_effort: cloud providers get it natively via litellm,
        # Ollama gets think=False for low effort levels.
        if reasoning_effort:
            if model_string.startswith("ollama/"):
                if reasoning_effort in ("none", "minimal", "low"):
                    kwargs["think"] = False
            else:
                kwargs["reasoning_effort"] = reasoning_effort

        # Try models in order with failover on transient errors
        if len(all_models) > 1:
            chain = FailoverChain(all_models)
            try:
                result = chain.run(lambda m: litellm.completion(model=m, **kwargs))
                response = result.value
            except Exception as exc:
                logger.warning(
                    "All models in failover chain failed for '%s': %s",
                    model_string,
                    exc,
                )
                raise
        else:
            kwargs["model"] = model_string
            response = litellm.completion(**kwargs)
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason

        # Track usage if a tracker is provided
        if usage_tracker is not None:
            usage = getattr(response, "usage", None)
            if usage is not None:
                usage_tracker.record(
                    prompt_tokens=getattr(usage, "prompt_tokens", 0),
                    completion_tokens=getattr(usage, "completion_tokens", 0),
                    model=model_string,
                )

        # Schema-aware JSON repair. json_repair's schema_repair_mode='salvage'
        # fixes syntax errors, coerces types, and fills missing fields toward
        # the schema. If the repaired output still fails PydanticAI's validation,
        # the retry loop sends an enriched RetryPromptPart (see _enrich_retry_content)
        # that includes the valid partial data so the model only fixes the broken parts.
        if not info.allow_text_output and content.strip():
            repaired = salvage_json(content, schema=schema)
            if repaired is not None:
                content = json.dumps(repaired, ensure_ascii=False)

        # Save debug prompt when KNOX_DEBUG_LLM_DIR is set
        if agent is not None:
            save_debug_prompt(
                agent,
                model_string,
                litellm_messages,
                kwargs,
                content,
                finish_reason,
            )

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
                shorten_string(content),
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


def make_litellm_model(
    model_string: str,
    usage_tracker: UsageTracker | None = None,
    fallback_models: list[str] | None = None,
    agent: Any | None = None,
    reasoning_effort: str | None = None,
    schema: type[BaseModel] | None = None,
) -> FunctionModel:
    """Create a PydanticAI :class:`FunctionModel` backed by ``litellm.completion()``.

    This preserves litellm's full provider dispatch chain — custom providers,
    API key resolution, monkey-patches, and per-call kwargs all work unchanged.

    Args:
        model_string: Model name in litellm format (``"provider/model"``).
        usage_tracker: Optional :class:`UsageTracker` to record token counts.
        fallback_models: Optional list of fallback models tried on transient errors.
        agent: Optional owning agent instance.  When provided, its ``stage`` and
            ``current_source_id`` attributes are recorded in the debug dump.
        reasoning_effort: Optional reasoning effort (``"none"``, ``"minimal"``, ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``, ``"default"``).
        schema: Optional Pydantic model class for schema-aware JSON repair.
    """
    return FunctionModel(
        function=_make_litellm_function(
            model_string,
            usage_tracker=usage_tracker,
            fallback_models=fallback_models,
            agent=agent,
            reasoning_effort=reasoning_effort,
            schema=schema,
        ),
        model_name=model_string,
    )
