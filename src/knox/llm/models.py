"""Build native pydantic-ai Model instances and resolve model metadata.

KNOX uses the ``provider/model`` convention (e.g. ``ollama/qwen3``,
``openai/gpt-5.2``, ``claudinio/claude-4``). This module translates such
a string into a native pydantic-ai :class:`~pydantic_ai.models.Model`.

Resolution strategy (first match wins):

1. **Native pydantic-ai provider** — :func:`pydantic_ai.providers.infer_provider_class`
   is the smart dispatcher: it returns the matching ``Provider`` subclass
   for every provider pydantic-ai ships a class for (ollama, openai,
   anthropic, google, deepseek, groq, mistral, cohere, bedrock, …). We
   construct that provider, then :func:`~pydantic_ai.models.infer_model`
   builds the typed Model subclass (``OllamaModel``, ``AnthropicModel``,
   ``OpenAIResponsesModel``, …). No hard-coded string comparison on our
   side — pydantic-ai's dispatch resolves both the provider class and
   the model class.

2. **models-dev catalog** — for providers pydantic-ai doesn't ship a
   class for (custom OpenAI-compatible mirrors like ``claudinio``,
   ``opencode``), the catalog's ``api`` URL and advertised ``env`` var
   build a generic :class:`~pydantic_ai.models.openai.OpenAIChatModel`
   on an :class:`~pydantic_ai.providers.openai.OpenAIProvider`.

3. Neither → ``ValueError``.

API keys and endpoints are **never** injected via environment mutation.
Defaults are passed as constructor arguments; user overrides are read
from the env vars the provider/catalog advertises.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import litellm
from json_repair import repair_json
from models_dev import Provider, get_model_by_id, get_provider
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage

from knox.logging import get_logger

logger = get_logger(__name__)

# Placeholder satisfies the OpenAI SDK's non-empty api_key requirement for
# local / keyless OpenAI-compatible endpoints (mirrors pydantic-ai's own
# OllamaProvider behaviour).
_NO_KEY_PLACEHOLDER = "api-key-not-set"


# ── public API ─────────────────────────────────────────────────────────


def build_model(model_string: str) -> Model:
    """Build a native pydantic-ai Model from a KNOX ``provider/model`` string."""
    provider_id, model_name = _split(model_string)

    # 1. Native pydantic-ai provider?
    try:
        return _build_native(provider_id, model_name)
    except _NotNativeProvider:
        pass

    # 2. models-dev catalog (OpenAI-compatible custom provider)?
    try:
        return _build_catalog(provider_id, model_name)
    except _NotInCatalog:
        pass

    msg = (
        f"Unknown provider {provider_id!r}: not natively supported by "
        f"pydantic-ai and not in the models-dev catalog. "
        f"Check the provider prefix (before the first '/' or ':')."
    )
    raise ValueError(msg)


def resolve_model_limits(model_id: str) -> tuple[int, int]:
    """Resolve ``(max_input, max_output)`` token limits from the models-dev catalog.

    Raises:
        ValueError: If *model_id* has no ``/`` or ``:`` separator.
        KeyError: If the provider or model is not in the catalog.
        LookupError: If the model entry has no limit data.
    """
    provider_id, model_name = _split(model_id)
    model = get_model_by_id(provider_id, model_name)  # raises KeyError

    limit = model.limit
    if limit is None or limit.context is None:
        raise LookupError(f"No limit data for {model_id!r}")

    return limit.context, limit.output or limit.context


# ── parsing ────────────────────────────────────────────────────────────


def _split(model_string: str) -> tuple[str, str]:
    """Split ``provider/model`` or ``provider:model`` into *(provider, model_name)*.

    The first ``/`` or ``:`` (whichever comes first) is the separator; the
    model name may itself contain slashes or colons.
    """
    slash = model_string.find("/")
    colon = model_string.find(":")
    candidates = [p for p in (slash, colon) if p > 0]
    if not candidates:
        raise ValueError(
            f"Invalid model string {model_string!r}. "
            f"Use 'provider/model' or 'provider:model'."
        )
    pos = min(candidates)
    return model_string[:pos], model_string[pos + 1 :]


# ── path 1: native pydantic-ai provider ────────────────────────────────


class _NotNativeProvider(Exception):
    """Provider is not natively known to pydantic-ai."""


def _build_native(provider_id: str, model_name: str) -> Model:
    """Build via pydantic-ai's native provider/model dispatch.

    Raises :class:`_NotNativeProvider` when pydantic-ai has no class for
    *provider_id* (caller falls through to the catalog path).
    """
    try:
        provider_cls = infer_provider_class(provider_id)
    except ValueError as exc:
        raise _NotNativeProvider from exc

    # Ollama: route through litellm so `think` can disable reasoning.
    # pydantic-ai's OllamaModel uses /v1/chat/completions which ignores the
    # think parameter; litellm calls the native /api/chat where think=False
    # is honoured. litellm owns the transport (endpoint, schema, images, usage).
    if provider_cls is OllamaProvider:
        return _ollama_function_model(model_name)

    provider = provider_cls()
    return infer_model(
        f"{provider_id}:{model_name}",
        provider_factory=lambda _name: provider,
    )


def _ollama_function_model(model_name: str) -> FunctionModel:
    """Wrap litellm's ollama transport in a pydantic-ai FunctionModel.

    Only concern is mapping ``reasoning_effort`` → litellm's ``think`` flag;
    litellm handles everything else (endpoint, auth, structured output,
    images, token usage).
    """

    async def request_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        settings = info.model_settings or {}
        kwargs: dict[str, Any] = {
            "model": f"ollama/{model_name}",
            "messages": _to_litellm_messages(messages, info),
        }
        if settings.get("reasoning_effort") in {"none", "minimal", "low"}:
            kwargs["think"] = False
        if "timeout" in settings:
            kwargs["timeout"] = settings["timeout"]
        if "max_tokens" in settings:
            kwargs["max_tokens"] = settings["max_tokens"]
        if info.output_tools:
            schema = info.output_tools[0].parameters_json_schema
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": info.output_tools[0].name, "schema": schema},
            }

        response = await litellm.acompletion(**kwargs)
        return _from_litellm_response(response, model_name, info.output_tools)

    return FunctionModel(request_fn, model_name=model_name)


# ── pydantic-ai ↔ litellm message bridge ───────────────────────────────


def _to_litellm_messages(
    messages: list[ModelMessage], info: AgentInfo
) -> list[dict[str, Any]]:
    """Flatten pydantic-ai ModelMessages to litellm ``{role, content}`` dicts."""
    out: list[dict[str, Any]] = []
    if info.instructions:
        out.append({"role": "system", "content": info.instructions})
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    out.append(_user_prompt_to_litellm(part))
                elif isinstance(part, RetryPromptPart):
                    feedback = (
                        part.content
                        if isinstance(part.content, str)
                        else str(part.content)
                    )
                    out.append(
                        {"role": "user", "content": f"Previous output invalid: {feedback}"}
                    )
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    out.append({"role": "assistant", "content": part.content})
                elif isinstance(part, ToolCallPart):
                    args = part.args
                    content = args if isinstance(args, str) else _json_dumps(args)
                    out.append({"role": "assistant", "content": content or "{}"})
    return out


def _user_prompt_to_litellm(part: UserPromptPart) -> dict[str, Any]:
    """UserPromptPart content is a str or a text+image sequence (VLM)."""
    content = part.content
    if isinstance(content, str):
        return {"role": "user", "content": content}
    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif isinstance(item, BinaryContent):
            data_url = f"data:{item.media_type};base64,{base64.b64encode(item.data).decode()}"
            blocks.append({"type": "image_url", "image_url": {"url": data_url}})
    return {"role": "user", "content": blocks}


def _from_litellm_response(
    response: Any,
    model_name: str,
    output_tools: list[Any],
) -> ModelResponse:
    """Convert a litellm completion response to a pydantic-ai ModelResponse."""
    message = response.choices[0].message
    content = message.get("content") or ""
    if output_tools:
        content = _repair_json_output(content)
        parts: list[TextPart | ToolCallPart] = [
            ToolCallPart(tool_name=output_tools[0].name, args=content or "{}")
        ]
    else:
        parts = [TextPart(content=content)]
    usage = response.usage
    return ModelResponse(
        parts=parts,
        usage=RequestUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        ),
        model_name=model_name,
        finish_reason="length" if response.choices[0].finish_reason == "length" else "stop",
    )


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj)


def _repair_json_output(raw: str) -> str:
    r"""Repair small-model JSON output.

    Two-step repair: (1) unescape literal ``\n`` / ``\t`` / ``\r``
    sequences that appear between tokens (small models emit the
    escape representation instead of real whitespace), then (2) use
    ``json_repair`` for remaining issues (unescaped control chars in
    strings, missing commas, unescaped quotes, etc.).
    """
    text = raw.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    try:
        return repair_json(text)
    except Exception:
        return raw  # unfixable — let pydantic-ai's retry handle it


# ── path 2: models-dev catalog (OpenAI-compatible custom provider) ─────


class _NotInCatalog(Exception):
    """Provider is not in the models-dev catalog or has no API endpoint."""


def _build_catalog(provider_id: str, model_name: str) -> Model:
    """Build a generic OpenAI-compatible model from models-dev catalog data.

    The catalog's ``api`` field is the OpenAI-compatible base URL; the
    advertised ``env`` var carries the API key. Raises
    :class:`_NotInCatalog` when the provider is absent or has no ``api``.
    """
    try:
        catalog = get_provider(provider_id)
    except KeyError as exc:
        raise _NotInCatalog from exc

    if not catalog.api:
        raise _NotInCatalog(
            f"Provider {provider_id!r} is in the catalog but has no API URL."
        )

    base_url = _normalize_base_url(catalog.api)
    api_key = _catalog_api_key(catalog)
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
    )


def _normalize_base_url(api_url: str) -> str:
    """Ensure the catalog ``api`` URL ends with ``/v1``."""
    base = api_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _catalog_api_key(catalog: Provider) -> str:
    """API key from the first advertised env var that is set, else a placeholder."""
    for env_var in catalog.env:
        value = os.environ.get(env_var)
        if value:
            return value
    return _NO_KEY_PLACEHOLDER


__all__ = ["build_model", "resolve_model_limits"]
