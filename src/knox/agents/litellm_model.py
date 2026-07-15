"""KnoxLiteLLMModel — PydanticAI Model backed by litellm via pydantic-ai-litellm.

Thin subclass of :class:`pydantic_ai_litellm.LiteLLMModel` that adds
KNOX-specific features while delegating all message conversion, tool
calling, and streaming to the upstream library.

Added features
--------------
* **Schema-aware JSON repair** — ``salvage_json(content, schema=…)``
  runs on every structured-output response before PydanticAI validates.
* **Retry enrichment** — when PydanticAI sends a ``RetryPromptPart``,
  the valid portions of the model's previous output are extracted and
  included so the model only fixes the broken parts.
* **Failover chain** — on transient errors the primary model is retried,
  then fallback models are tried in order.
* **Debug saving** — when ``KNOX_DEBUG_LLM_DIR`` is set, every LLM
  call is saved to disk with full messages + response.
* **Usage tracking** — token counts are recorded on the agent's
  :class:`~knox.llm.usage.UsageTracker`.
* **Reasoning effort** — ``reasoning_effort`` is forwarded to cloud
  providers and mapped to ``think=False`` for Ollama.
* **JSON mode** — ``response_format={"type": "json_object"}`` is
  added for structured-output agents.

Usage::

    from knox.agents.litellm_model import KnoxLiteLLMModel

    agent = Agent(
        model=KnoxLiteLLMModel("ollama/gpt-oss:20b", output_schema=MyOutput),
        output_type=MyOutput,
    )
"""

from __future__ import annotations

import json
from typing import Any, cast

import litellm
from pydantic import BaseModel
from pydantic_ai import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters, check_allow_model_requests
from pydantic_ai.settings import ModelSettings
from pydantic_ai_litellm import LiteLLMModel, LiteLLMModelSettings

from knox.agents.debug import save_debug_prompt
from knox.core.utils.exceptions import shorten_string
from knox.llm.format import salvage_json
from knox.llm.usage import UsageTracker
from knox.logging import get_logger

logger = get_logger(__name__)

#: Transient error classes that trigger failover to the next model.
_TRANSIENT_ERRORS = (
    litellm.exceptions.Timeout,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ServiceUnavailableError,
)


class KnoxLiteLLMModel(LiteLLMModel):
    """LiteLLM-backed PydanticAI model with KNOX-specific enhancements."""

    def __init__(
        self,
        model_name: str,
        *,
        output_schema: type[BaseModel] | None = None,
        usage_tracker: UsageTracker | None = None,
        fallback_models: list[str] | None = None,
        agent: Any | None = None,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        custom_llm_provider: str | None = None,
        settings: ModelSettings | None = None,
    ) -> None:
        super().__init__(
            model_name,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            settings=settings,
        )
        self._output_schema = output_schema
        self._usage_tracker = usage_tracker
        self._fallback_models = fallback_models or []
        self._agent = agent
        self._reasoning_effort = reasoning_effort

    # ------------------------------------------------------------------
    # request — failover + debug + usage tracking
    # ------------------------------------------------------------------

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()

        all_models = [self._model_name] + self._fallback_models
        last_error: Exception | None = None

        for model in all_models:
            try:
                response = await self._completion_create(
                    messages,
                    False,
                    cast(LiteLLMModelSettings, model_settings or {}),
                    model_request_parameters,
                    model_name=model,
                )
                result = self._process_response(response)

                # Debug save
                if self._agent is not None:
                    litellm_messages = await self._map_messages(messages, model_request_parameters)
                    litellm_messages = self._enrich_retry_messages(litellm_messages)
                    content = ""
                    for part in result.parts:
                        if isinstance(part, TextPart):
                            content = part.content
                            break
                    finish_reason = getattr(response.choices[0], "finish_reason", "stop")
                    save_debug_prompt(
                        self._agent,
                        model,
                        litellm_messages,
                        {"model": model},
                        content,
                        finish_reason,
                    )

                return result

            except _TRANSIENT_ERRORS as exc:
                logger.warning(
                    "KnoxLiteLLMModel: transient error for '%s': %s",
                    model,
                    exc,
                )
                last_error = exc
                continue

        # All models exhausted
        if last_error is not None:
            raise last_error
        raise UnexpectedModelBehavior("All models in failover chain failed")

    # ------------------------------------------------------------------
    # _completion_create — response_format, reasoning_effort, failover
    # ------------------------------------------------------------------

    async def _completion_create(  # noqa: PLR0915
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: LiteLLMModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        model_name: str | None = None,
    ) -> Any:
        """Build litellm kwargs and call ``acompletion``.

        Overrides the parent to add:

        * ``response_format`` for structured-output agents
        * ``reasoning_effort`` / ``think`` mapping
        * Per-model usage tracking from the raw response
        """
        tools = self._get_tools(model_request_parameters)

        tool_choice: str | None = "required" if not model_request_parameters.allow_text_output else "auto"
        if not tools:
            tool_choice = None

        litellm_messages = await self._map_messages(messages, model_request_parameters)
        litellm_messages = self._enrich_retry_messages(litellm_messages)

        effective_model = model_name or self._model_name
        completion_kwargs = self._build_completion_kwargs(
            effective_model,
            litellm_messages,
            stream,
            tools,
            tool_choice,
            model_settings,
            model_request_parameters,
        )

        try:
            response = await litellm.acompletion(**completion_kwargs)
        except Exception as exc:
            if hasattr(exc, "status_code") and isinstance(exc.status_code, int) and exc.status_code >= 400:
                raise ModelHTTPError(
                    status_code=exc.status_code,
                    model_name=effective_model,
                    body=str(exc),
                ) from exc
            raise

        self._track_usage(response, effective_model)
        self._log_response_anomalies(response, effective_model, model_request_parameters)
        return response

    def _build_completion_kwargs(
        self,
        effective_model: str,
        litellm_messages: list[dict[str, Any]],
        stream: bool,
        tools: list[dict[str, Any]],
        tool_choice: str | None,
        model_settings: LiteLLMModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> dict[str, Any]:
        """Build the kwargs dict for ``litellm.acompletion``."""
        kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": litellm_messages,
            "stream": stream,
        }

        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        for key in ("max_tokens", "temperature", "top_p", "timeout", "seed"):
            if val := model_settings.get(key):
                kwargs[key] = val
        if stop_sequences := model_settings.get("stop_sequences"):
            kwargs["stop"] = stop_sequences
        if parallel_tool_calls := model_settings.get("parallel_tool_calls"):
            kwargs["parallel_tool_calls"] = parallel_tool_calls

        # LiteLLM-specific parameters
        if api_key := model_settings.get("litellm_api_key") or self._api_key:
            kwargs["api_key"] = api_key
        if api_base := model_settings.get("litellm_api_base") or self._api_base:
            kwargs["api_base"] = api_base
        if custom_provider := (model_settings.get("litellm_custom_llm_provider") or self._custom_llm_provider):
            kwargs["custom_llm_provider"] = custom_provider
        if metadata := model_settings.get("litellm_metadata"):
            kwargs["metadata"] = metadata
        if extra_headers := model_settings.get("extra_headers"):
            kwargs["extra_headers"] = dict(extra_headers)
        if extra_body := model_settings.get("extra_body"):
            kwargs["extra_body"] = extra_body

        # --- KNOX additions ---
        if not model_request_parameters.allow_text_output:
            kwargs["response_format"] = {"type": "json_object"}

        if self._reasoning_effort:
            if effective_model.startswith("ollama/"):
                if self._reasoning_effort in ("none", "minimal", "low"):
                    kwargs["think"] = False
            else:
                kwargs["reasoning_effort"] = self._reasoning_effort

        return kwargs

    def _track_usage(self, response: Any, effective_model: str) -> None:
        """Record token usage from a raw litellm response."""
        if self._usage_tracker is None:
            return
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._usage_tracker.record(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                model=effective_model,
            )

    def _log_response_anomalies(  # noqa: PLR6301
        self,
        response: Any,
        effective_model: str,
        model_request_parameters: ModelRequestParameters,
    ) -> None:
        """Log warnings for empty, non-JSON, or truncated responses."""
        content = response.choices[0].message.content or ""
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        if not content.strip() and not model_request_parameters.allow_text_output:
            logger.warning(
                "KnoxLiteLLMModel: empty response from '%s' (finish_reason=%s)",
                effective_model,
                finish_reason,
            )
        elif not content.strip().startswith("{") and not model_request_parameters.allow_text_output:
            logger.warning(
                "KnoxLiteLLMModel: non-JSON response from '%s' (finish_reason=%s, first 200 chars): %s",
                effective_model,
                finish_reason,
                shorten_string(content),
            )
        elif finish_reason == "length" and not model_request_parameters.allow_text_output:
            logger.warning(
                "KnoxLiteLLMModel: truncated response from '%s' (finish_reason=length, %d chars). Increase max_tokens or reduce input size.",
                effective_model,
                len(content),
            )

    # ------------------------------------------------------------------
    # _process_response — schema-aware JSON repair
    # ------------------------------------------------------------------

    def _process_response(self, response: Any) -> ModelResponse:
        """Process a non-streamed response with schema-aware JSON repair."""
        result = super()._process_response(response)

        if self._output_schema is None:
            return result

        # Run salvage_json on every text part
        new_parts: list[Any] = []
        for part in result.parts:
            if isinstance(part, TextPart) and part.content.strip():
                repaired = salvage_json(part.content, schema=self._output_schema)
                if repaired is not None:
                    new_parts.append(TextPart(content=json.dumps(repaired, ensure_ascii=False)))
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)

        return ModelResponse(
            parts=new_parts,
            model_name=result.model_name,
            timestamp=result.timestamp,
        )

    # ------------------------------------------------------------------
    # _enrich_retry_messages — inject valid partial JSON into retries
    # ------------------------------------------------------------------

    def _enrich_retry_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Post-process litellm messages to enrich retry feedback.

        Scans for the pattern: assistant (with content) → user (with
        "Validation errors"). When found, extracts the valid portions of
        the assistant's output via ``salvage_json`` and includes them in
        the retry message so the model only fixes the broken parts.

        No-op when ``_output_schema`` is ``None``.
        """
        if self._output_schema is None:
            return messages
        result: list[dict[str, Any]] = []
        last_assistant_content: str | None = None

        for msg in messages:
            role = msg.get("role", "")
            if role == "assistant":
                last_assistant_content = msg.get("content", "")
                result.append(msg)
            elif role == "user" and last_assistant_content and "Validation errors" in str(msg.get("content", "")):
                enriched = self._build_enriched_retry(msg.get("content", ""), last_assistant_content)
                result.append({**msg, "content": enriched})
            else:
                result.append(msg)

        return result

    def _build_enriched_retry(self, original_content: str, assistant_content: str, max_chars_last_msg: int = 3000) -> str:
        """Build an enriched retry message with valid partial JSON."""
        repaired = salvage_json(assistant_content, schema=self._output_schema)
        if repaired is None:
            return original_content

        valid_json = json.dumps(repaired, ensure_ascii=False, indent=2)
        if len(valid_json) > max_chars_last_msg:
            valid_json = valid_json[:max_chars_last_msg] + "\n  ... (truncated)"

        return (
            f"{original_content}\n\n"
            f"The valid parts extracted from your previous response:\n"
            f"```json\n{valid_json}\n```\n\n"
            f"Return ONLY a corrected JSON object that fixes the listed "
            f"errors. Keep all valid fields unchanged."
        )
