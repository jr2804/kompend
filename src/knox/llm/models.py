"""Build native pydantic-ai Model instances from KNOX model strings.

KNOX uses the ``provider/model`` convention (e.g. ``ollama/qwen3``,
``opencode-zen/gpt-5.4-nano``). This module translates those strings to
native pydantic-ai models (:class:`~pydantic_ai.models.ollama.OllamaModel`,
:class:`~pydantic_ai.models.openai.OpenAIChatModel`, etc.), using
``provider:model`` auto-resolution for standard providers and explicit
construction for KNOX's custom providers (claudinio, ollama-cloud,
opencode-zen, opencode-go).

Model strings always split at the **first** slash — the remainder is the
model name verbatim and may itself contain slashes (e.g.
``ollama/hf.co/qwen3.6`` → provider ``ollama``, model ``hf.co/qwen3.6``).

OpenCode Zen/Go serve each model at exactly one API format (OpenAI Chat,
OpenAI Responses, or Anthropic Messages). The format is not disclosed by
``/v1/models``, so :class:`_OpencodeModel` discovers it by probing each
endpoint with a minimal request on first use, then delegates all
subsequent requests to the discovered native model. Discovery results are
cached per ``(tier, model_name)``.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from knox.llm.http_client import create_retrying_http_client
from knox.logging import get_logger

logger = get_logger(__name__)

# ── Custom-provider endpoint constants ─────────────────────────────────

_CLAUDINIO_BASE_URL = "https://api.claudin.io/v1"
_OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"
_OPENCODE_ZEN_BASE_URL = "https://opencode.ai/zen"
_OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go"


def build_model(model_string: str) -> Model:
    """Build a native pydantic-ai Model from a KNOX ``provider/model`` string.

    The string is split at the first ``/``. Supported custom prefixes
    (``claudinio``, ``ollama-cloud``, ``opencode-zen``, ``opencode-go``)
    construct native models pointed at the provider's endpoint. Any other
    prefix is translated to pydantic-ai's ``provider:model`` format and
    resolved via :func:`pydantic_ai.models.infer_model`.

    Raises:
        ValueError: if the string has no ``/`` separator and isn't a
            ``provider:model`` string, or if pydantic-ai can't resolve it.
    """
    prefix, sep, name = model_string.partition("/")

    # Retrying HTTP client for custom providers.
    _http = create_retrying_http_client()

    # Our custom providers — explicit native construction.
    if prefix == "claudinio" and sep:
        return OpenAIChatModel(
            name,
            provider=OpenAIProvider(
                base_url=_CLAUDINIO_BASE_URL,
                api_key=os.environ.get("CLAUDINIO_API_KEY", ""),
                http_client=_http,
            ),
        )
    if prefix == "ollama-cloud" and sep:
        return OpenAIChatModel(
            name,
            provider=OpenAIProvider(
                base_url=_OLLAMA_CLOUD_BASE_URL,
                api_key=os.environ.get("OLLAMA_API_KEY", ""),
                http_client=_http,
            ),
        )
    if prefix in ("opencode-zen", "opencode-go") and sep:
        return _OpencodeModel(prefix, name)

    # Standard providers — delegate to pydantic-ai's auto-resolution.
    # KNOX uses ``provider/model``; pydantic-ai uses ``provider:model``.
    pydantic_ai_string = f"{prefix}:{name}" if sep and ":" not in model_string else model_string
    try:
        return infer_model(pydantic_ai_string)
    except Exception as exc:
        msg = (
            f"Unknown model string {model_string!r}. Use a supported "
            f"prefix (ollama/, openai/, anthropic/, claudinio/, "
            f"ollama-cloud/, opencode-zen/, opencode-go/) or a pydantic-ai "
            f"'provider:model' string. Original error: {exc}"
        )
        raise ValueError(msg) from exc


# ── OpenCode lazy-discovery model ──────────────────────────────────────

#: Cache: (tier, model_name) → discovered native Model. Populated on first use.
_opencode_cache: dict[tuple[str, str], Model] = {}


class _OpencodeModel(Model):
    """Native model wrapper for OpenCode Zen/Go with lazy format discovery.

    OpenCode serves each model at exactly one endpoint format. The format
    isn't disclosed by ``/v1/models``, so this wrapper probes each candidate
    endpoint (OpenAI Chat Completions, Anthropic Messages, OpenAI Responses)
    with a minimal request on first use, constructs the matching native
    pydantic-ai model, caches it, and delegates all subsequent requests.

    Discovery requires the provider's API key (``OPENCODE_ZEN_API_KEY`` /
    ``OPENCODE_GO_API_KEY`` / ``OPENCODE_API_KEY``). If the key is missing,
    construction still succeeds but the first request fails with a clear
    auth error from the probed endpoint.
    """

    def __init__(self, tier: str, name: str) -> None:
        self._tier = tier
        self._name = name

    # ── Model ABC interface (delegate after discovery) ──────────────────

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def system(self) -> str:
        return self._tier

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        delegate = await self._resolve()
        return await delegate.request(messages, model_settings, model_request_parameters)

    # ── Discovery ───────────────────────────────────────────────────────

    async def _resolve(self) -> Model:
        """Return the cached native model, discovering it if needed."""
        cache_key = (self._tier, self._name)
        if cache_key in _opencode_cache:
            return _opencode_cache[cache_key]

        delegate = await self._discover()
        _opencode_cache[cache_key] = delegate
        return delegate

    async def _discover(self) -> Model:
        """Probe each endpoint format; return the first native model that works."""
        base, api_key = self._endpoint_config()

        # Candidate formats in probe order. Each builder constructs a native
        # model; the probe fn returns True if the endpoint accepts the model.
        candidates: list[tuple[Callable[[], Model], Callable[[str, str], Awaitable[bool]]]] = [
            (
                lambda: OpenAIChatModel(
                    self._name,
                    provider=OpenAIProvider(base_url=f"{base}/v1", api_key=api_key),
                ),
                _probe_openai_chat,
            ),
            (
                lambda: AnthropicModel(
                    self._name,
                    provider=AnthropicProvider(base_url=base, api_key=api_key),
                ),
                _probe_anthropic,
            ),
            (
                lambda: OpenAIResponsesModel(
                    self._name,
                    provider=OpenAIProvider(base_url=f"{base}/v1", api_key=api_key),
                ),
                _probe_openai_responses,
            ),
        ]

        for builder, probe in candidates:
            if await probe(base, api_key, self._name):
                logger.debug(
                    "OpenCode discovery: %s/%s → %s",
                    self._tier,
                    self._name,
                    type(builder()).__name__,
                )
                return builder()

        raise ValueError(f"Could not discover API format for {self._tier}/{self._name} at {base}. Verify the model name and API key.")

    def _endpoint_config(self) -> tuple[str, str]:
        """Return (base_url, api_key) for this tier."""
        if self._tier == "opencode-go":
            base = _OPENCODE_GO_BASE_URL
            api_key = os.environ.get("OPENCODE_GO_API_KEY") or os.environ.get("OPENCODE_API_KEY", "")
        else:
            base = _OPENCODE_ZEN_BASE_URL
            api_key = os.environ.get("OPENCODE_ZEN_API_KEY") or os.environ.get("OPENCODE_API_KEY", "")
        return base, api_key


# ── Endpoint probes ────────────────────────────────────────────────────
#
# Each probe sends a minimal (1-token) request to its endpoint format and
# returns True if the endpoint accepts the model. A 200 means the model is
# served at this format; a 404/400 (model not found / not supported) means
# try the next format. Auth errors (401) propagate so the user gets a clear
# message rather than silently falling through every format.


def _short_probe_body(model: str, *, anthropic: bool = False, responses: bool = False) -> dict[str, Any]:
    """Build a minimal request body for the given format."""
    if anthropic:
        return {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    if responses:
        return {"model": model, "input": "hi", "max_output_tokens": 1}
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}


async def _probe_openai_chat(base: str, api_key: str, model: str) -> bool:
    headers = {"Authorization": f"Bearer {api_key}"}
    body = _short_probe_body(model)
    return await _probe_endpoint(f"{base}/v1/chat/completions", headers, body)


async def _probe_anthropic(base: str, api_key: str, model: str) -> bool:
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    body = _short_probe_body(model, anthropic=True)
    return await _probe_endpoint(f"{base}/v1/messages", headers, body)


async def _probe_openai_responses(base: str, api_key: str, model: str) -> bool:
    headers = {"Authorization": f"Bearer {api_key}"}
    body = _short_probe_body(model, responses=True)
    return await _probe_endpoint(f"{base}/v1/responses", headers, body)


async def _probe_endpoint(url: str, headers: dict[str, str], body: dict[str, Any]) -> bool:
    """POST a minimal request; return True if the endpoint accepts the model.

    Acceptance = HTTP 200. Rejection (wrong endpoint for this model) =
    404 or 400. Auth failure (401/403) propagates as an error so the
    user sees a clear auth message instead of a misleading "format not found".
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=headers, json=body)
    except httpx.HTTPError as exc:
        logger.debug("OpenCode probe network error for %s: %s", url, exc)
        return False

    if resp.status_code == 200:
        return True
    if resp.status_code in (401, 403):
        # Auth problem — let it surface clearly.
        raise PermissionError(f"OpenCode API key rejected (HTTP {resp.status_code}) at {url}: {resp.text[:200]}")
    # 404 / 400 / others → model not served at this endpoint format.
    logger.debug(
        "OpenCode probe %s rejected model (HTTP %s): %s",
        url,
        resp.status_code,
        resp.text[:200],
    )
    return False


__all__ = ["build_model"]
