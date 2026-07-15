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
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
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

from knox.core.utils.exceptions import shorten_string
from knox.llm.failover import FailoverChain
from knox.llm.format import salvage_json
from knox.llm.usage import UsageTracker
from knox.logging import get_logger

logger = get_logger(__name__)


def _clean_json_artifacts(obj: Any) -> tuple[Any, list[str]]:
    """Recursively clean JSON artifacts from model output.

    Small quantized models sometimes place field names as bare strings inside
    arrays that should contain objects (e.g. ``[{...page...}, "provenance", "related"]``).
    This removes non-dict items from any list containing at least one dict.

    Returns a tuple of (cleaned_obj, removed_artifacts) where *removed_artifacts*
    is a list of stringified removed items, used for Layer 2 retry feedback.
    """
    removed: list[str] = []

    def _walk(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: _walk(v) for k, v in o.items()}
        if isinstance(o, list):
            cleaned_items = [_walk(item) for item in o]
            has_dicts = any(isinstance(item, dict) for item in cleaned_items)
            if has_dicts:
                result: list[Any] = []
                for item in cleaned_items:
                    if isinstance(item, dict):
                        result.append(item)
                    else:
                        removed.append(json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item)
                return result
            return cleaned_items
        return o

    return _walk(obj), removed


def _save_debug_prompt(
    model_string: str,
    litellm_messages: list[dict[str, Any]],
    kwargs: dict[str, Any],
    response_content: str,
    finish_reason: str | None,
    source_id: str = "",
) -> None:
    """Save the LLM prompt and response to a debug file.

    Triggered by ``KNOX_DEBUG_LLM_DIR`` env var. Each call creates a
    timestamped JSON file with the full message list, model settings,
    and raw response. Useful for manual testing of model output.
    """
    debug_dir = os.environ.get("KNOX_DEBUG_LLM_DIR")
    if not debug_dir:
        return
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_slug = model_string.replace("/", "_").replace(":", "")
    filepath = debug_path / f"llm_{ts}_{model_slug}.json"
    payload = {
        "model": model_string,
        "source_id": source_id,
        "timestamp": datetime.now().isoformat(),
        "messages": litellm_messages,
        "kwargs": {k: v for k, v in kwargs.items() if k not in ("messages", "model")},
        "response": response_content,
        "response_parsed": salvage_json(response_content),
        "finish_reason": finish_reason,
    }
    try:
        filepath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Debug prompt saved to %s", filepath)
    except Exception as exc:
        logger.warning("Failed to save debug prompt: %s", exc)


def _messages_to_litellm(
    messages: list[ModelMessage],
    *,
    removed_artifacts: list[str] | None = None,
) -> list[dict[str, str]]:
    """Convert PydanticAI messages to litellm chat format.

    Handles all part types PydanticAI uses in the structured-output retry
    loop: ``RetryPromptPart`` (validation errors sent back on schema
    mismatch) and ``ToolReturnPart`` are converted to ``user`` messages so
    the model sees the feedback. Without this, retries are silently
    dropped and the model repeats the same invalid output.

    When *removed_artifacts* is non-empty, they are appended to the
    RetryPromptPart message as Layer 2 feedback — telling the model exactly
    what was wrong with its previous response so it can fix it.
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
                    # Layer 2: include removed artifacts in retry feedback
                    if removed_artifacts:
                        artifacts_str = ", ".join(removed_artifacts)
                        content += (
                            f"\n\nInvalid JSON artifacts detected in previous reply: "
                            f"[{artifacts_str}]. These items were placed as bare "
                            f"values inside an object array. If they represent schema "
                            f"fields, include them as proper key-value pairs in the "
                            f"JSON object. If they can be omitted, return empty JSON."
                        )
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


def _make_litellm_function(  # noqa: C901, PLR0915
    model_string: str,
    usage_tracker: UsageTracker | None = None,
    fallback_models: list[str] | None = None,
    source_id_provider: Callable[[], str] | None = None,
    reasoning_effort: str | None = None,
) -> Any:
    """Create a FunctionModel-compatible function calling ``litellm.completion()``.

    When *reasoning_effort* is set, it is forwarded to litellm as
    ``reasoning_effort=`` (for cloud providers like OpenAI/Anthropic) and
    mapped to ``think=False`` for Ollama models that support the think toggle.
    """
    # Layer 2 state: artifacts removed from the previous response.
    # Persists across retry calls within one agent run.
    _removed_artifacts: list[str] = []

    def _fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # noqa: C901, PLR0915
        litellm_messages = _messages_to_litellm(messages, removed_artifacts=_removed_artifacts)

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
                # Ollama: map low effort levels to think=False
                if reasoning_effort in ("none", "minimal", "low"):
                    kwargs["think"] = False
            else:
                # Cloud providers: forward reasoning_effort directly
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

        # Repair malformed JSON for structured output agents. Small quantized
        # models often return truncated or slightly malformed JSON (trailing
        # commas, unclosed brackets, prose wrapping). The old direct-litellm
        # path used salvage_json for this; PydanticAI's built-in validation
        # is strict and rejects anything that isn't perfectly valid JSON.
        #
        # Layer 1: salvage_json repairs syntax errors.
        # Layer 2: _clean_json_artifacts removes non-dict items from
        # dict-containing lists (common artifact: field names as bare
        # strings mixed into object arrays). Removed artifacts are stored
        # in the closure variable for Layer 2 retry feedback.
        _removed_artifacts.clear()
        if not info.allow_text_output and content.strip():
            repaired = salvage_json(content)
            if repaired is not None:
                repaired, new_removed = _clean_json_artifacts(repaired)
                _removed_artifacts.extend(new_removed)
                content = json.dumps(repaired, ensure_ascii=False)
                if new_removed:
                    logger.info(
                        "LitellmModel: cleaned %d JSON artifact(s) from '%s' response: %s",
                        len(new_removed),
                        model_string,
                        ", ".join(new_removed),
                    )

        # Save debug prompt when KNOX_DEBUG_LLM_DIR is set
        _source_id = source_id_provider() if source_id_provider else ""
        _save_debug_prompt(model_string, litellm_messages, kwargs, content, finish_reason, source_id=_source_id)

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
    source_id_provider: Callable[[], str] | None = None,
    reasoning_effort: str | None = None,
) -> FunctionModel:
    """Create a PydanticAI :class:`FunctionModel` backed by ``litellm.completion()``.

    This preserves litellm's full provider dispatch chain — custom providers,
    API key resolution, monkey-patches, and per-call kwargs all work unchanged.

    Args:
        model_string: Model name in litellm format (``"provider/model"``).
        usage_tracker: Optional :class:`UsageTracker` to record token counts.
        fallback_models: Optional list of fallback models tried on transient errors.
        source_id_provider: Optional callable returning the current source_id for debug logging.
        reasoning_effort: Optional reasoning effort (``"none"``, ``"minimal"``, ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``, ``"default"``).
    """
    return FunctionModel(
        function=_make_litellm_function(
            model_string,
            usage_tracker=usage_tracker,
            fallback_models=fallback_models,
            source_id_provider=source_id_provider,
            reasoning_effort=reasoning_effort,
        ),
        model_name=model_string,
    )
