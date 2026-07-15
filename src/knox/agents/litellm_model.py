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
from collections.abc import Callable
from datetime import datetime
from typing import Any

import litellm
from pydantic import BaseModel, ValidationError
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



def _incremental_json_repair(
    content: str,
    schema: type[BaseModel],
    model_string: str,
    litellm_messages: list[dict[str, Any]],
    kwargs: dict[str, Any],
    usage_tracker: UsageTracker | None,
    *,
    max_retries: int = 2,
) -> str:
    """Repair model JSON output incrementally with focused follow-up prompts.

    Instead of asking the model to redo the entire output, this function:
    1. Parses the output with schema-aware salvage_json (fixes syntax, coerces types)
    2. Validates against the schema
    3. If valid → returns the repaired JSON
    4. If invalid → sends a focused follow-up asking for ONLY the broken/missing parts
    5. Merges the model's new response with previously extracted valid data
    6. Repeats until valid or max_retries exhausted

    Returns the best-effort JSON string (may not pass validation if all
    retries are exhausted — PydanticAI will then handle the final failure).
    """
    # Step 1: Initial salvage
    repaired = salvage_json(content, schema=schema)
    if repaired is None:
        # Complete parse failure — fall back to plain syntax repair
        repaired = salvage_json(content)
        if repaired is None:
            return content  # nothing salvageable, let PydanticAI handle it

    for attempt in range(max_retries):
        # Step 2: Validate against schema
        try:
            schema.model_validate(repaired)
            # Valid! Return the repaired JSON
            return json.dumps(repaired, ensure_ascii=False)
        except ValidationError as exc:
            # Step 3: Build focused follow-up prompt
            errors = []
            for err in exc.errors():
                loc = ".".join(str(x) for x in err.get("loc", []))
                msg = err.get("msg", "")
                err_type = err.get("type", "")
                errors.append(f"  - Field '{loc}': {msg} (expected: {err_type})")

            valid_json = json.dumps(repaired, ensure_ascii=False, indent=2)
            # Truncate valid_json if very large (keep context manageable)
            if len(valid_json) > 4000:
                valid_json = valid_json[:4000] + "\n  ... (truncated)"

            follow_up = (
                f"Your previous JSON response was partially valid. The following "
                f"validation errors remain:\n{chr(10).join(errors)}\n\n"
                f"The valid parts extracted so far:\n```json\n{valid_json}\n```\n\n"
                f"Return ONLY a complete, corrected JSON object that fixes the "
                f"listed errors. Keep all valid fields unchanged."
            )

            logger.info(
                "IncrementalJSONRepair: attempt %d/%d for '%s', %d validation error(s)",
                attempt + 1,
                max_retries,
                model_string,
                len(errors),
            )

            # Step 4: Call model with focused follow-up
            retry_messages = litellm_messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": follow_up},
            ]
            retry_kwargs = {**kwargs, "messages": retry_messages}
            try:
                retry_response = litellm.completion(**retry_kwargs)
            except Exception as retry_exc:
                logger.warning(
                    "IncrementalJSONRepair: retry call failed for '%s': %s",
                    model_string,
                    retry_exc,
                )
                break

            retry_content = retry_response.choices[0].message.content or ""

            # Track usage from retry
            if usage_tracker is not None:
                usage = getattr(retry_response, "usage", None)
                if usage is not None:
                    usage_tracker.record(
                        prompt_tokens=getattr(usage, "prompt_tokens", 0),
                        completion_tokens=getattr(usage, "completion_tokens", 0),
                        model=model_string,
                    )

            if not retry_content.strip():
                logger.warning("IncrementalJSONRepair: empty retry response from '%s'", model_string)
                break

            # Step 5: Salvage and merge the new response
            new_repaired = salvage_json(retry_content, schema=schema)
            if new_repaired is not None:
                repaired = _merge_json(repaired, new_repaired)
                content = json.dumps(repaired, ensure_ascii=False)
            else:
                logger.warning(
                    "IncrementalJSONRepair: could not parse retry response from '%s'",
                    model_string,
                )
                break

    # Best effort — return whatever we have, even if not fully valid
    return json.dumps(repaired, ensure_ascii=False)


def _merge_json(base: Any, update: Any) -> Any:
    """Deep-merge two JSON values. Dict keys from *update* override *base*.

    Lists are replaced entirely (no attempt to merge list elements).
    Non-container values from *update* override *base*.
    """
    if isinstance(base, dict) and isinstance(update, dict):
        result = dict(base)
        for key, val in update.items():
            if key in result:
                result[key] = _merge_json(result[key], val)
            else:
                result[key] = val
        return result
    # For lists, scalars, etc. — update wins
    return update if update is not None else base


def _messages_to_litellm(
    messages: list[ModelMessage],
) -> list[dict[str, Any]]:
    """Convert PydanticAI messages to litellm chat format.

    Handles all part types PydanticAI uses in the structured-output retry
    loop: ``RetryPromptPart`` (validation errors sent back on schema
    mismatch) and ``ToolReturnPart`` are converted to ``user`` messages so
    the model sees the feedback. Without this, retries are silently
    dropped and the model repeats the same invalid output.
    """
    result: list[dict[str, Any]] = []
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


def _make_litellm_function(  # noqa: C901, PLR0915
    model_string: str,
    usage_tracker: UsageTracker | None = None,
    fallback_models: list[str] | None = None,
    source_id_provider: Callable[[], str] | None = None,
    reasoning_effort: str | None = None,
    schema: type[BaseModel] | None = None,
    stage: str = "",
    max_retries: int = 2,
) -> Any:
    """Create a FunctionModel-compatible function calling ``litellm.completion()``.

    When *reasoning_effort* is set, it is forwarded to litellm as
    ``reasoning_effort=`` (for cloud providers like OpenAI/Anthropic) and
    mapped to ``think=False`` for Ollama models that support the think toggle.
    """

    def _fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # noqa: C901, PLR0915
        litellm_messages = _messages_to_litellm(messages)

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

        # Schema-aware JSON repair with incremental retry.
        # Instead of PydanticAI's built-in retry (which sends useless
        # validation errors and asks the model to redo everything), we:
        #   1. Parse the output with salvage_json(schema=schema) — fixes
        #      syntax, coerces types, fills missing fields
        #   2. Validate against the schema
        #   3. If valid → done. If not → extract what IS valid, ask the
        #      model for ONLY the broken/missing parts, merge, retry.
        # This keeps context small (only focused follow-up, not full
        # re-submission) and gives the model actionable feedback.
        if not info.allow_text_output and content.strip() and schema is not None:
            content = _incremental_json_repair(
                content,
                schema,
                model_string,
                litellm_messages,
                kwargs,
                usage_tracker,
                max_retries=max_retries,
            )
        elif not info.allow_text_output and content.strip():
            # No schema available — best-effort syntax repair only
            repaired = salvage_json(content)
            if repaired is not None:
                content = json.dumps(repaired, ensure_ascii=False)

        # Save debug prompt when KNOX_DEBUG_LLM_DIR is set
        _source_id = source_id_provider() if source_id_provider else ""
        save_debug_prompt(
            model_string,
            litellm_messages,
            kwargs,
            content,
            finish_reason,
            source_id=_source_id,
            stage=stage,
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
    source_id_provider: Callable[[], str] | None = None,
    reasoning_effort: str | None = None,
    schema: type[BaseModel] | None = None,
    stage: str = "",
    max_retries: int = 2,
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
        schema: Optional Pydantic model class for schema-aware JSON repair.
        stage: Optional pipeline stage label (``compile``, ``wicer``, ``twostage``, …)
            recorded in the LLM debug dump.
        max_retries: Max incremental JSON repair attempts inside ``_fn``.
    """
    return FunctionModel(
        function=_make_litellm_function(
            model_string,
            usage_tracker=usage_tracker,
            fallback_models=fallback_models,
            source_id_provider=source_id_provider,
            reasoning_effort=reasoning_effort,
            schema=schema,
            stage=stage,
            max_retries=max_retries,
        ),
        model_name=model_string,
    )
