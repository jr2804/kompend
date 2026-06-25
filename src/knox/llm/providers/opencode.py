"""Custom litellm providers for OpenCode (https://opencode.ai).

Two tiers — **zen** (pay-as-you-go) and **go** (subscription) — with
model discovery from ``/v1/models`` and automatic endpoint routing
(openai-compat vs anthropic).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx
import litellm
from litellm import CustomLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk, ModelResponse

from knox.logging import get_logger

logger = get_logger(__name__)

_OPENCODE_ZEN_BASE_URL = "https://opencode.ai/zen"
_OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go"


class _OpenCodeProvider(CustomLLM):
    """Base class for OpenCode providers (zen / go).

    Handles model discovery from the OpenAI-compatible ``/v1/models``
    endpoint, endpoint routing (openai-compat vs anthropic), and standard
    completion / streaming methods.  Subclasses only need to set
    ``_base_url`` and ``_api_key_envs``.
    """

    _base_url: str = ""
    _api_key_envs: tuple[str, ...] = ()

    # Class-level cache: model_name -> "openai" | "anthropic"
    # Each subclass inherits its own None — first access populates per-subclass.
    _model_cache: dict[str, str] | None = None

    # ── Standard methods (identical for all OpenCode sub-providers) ───

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Any = None,
        litellm_params: Any = None,
        logger_fn: Any = None,
        headers: Any = None,
        timeout: float | httpx.Timeout | None = None,
        client: HTTPHandler | None = None,
    ) -> ModelResponse | litellm.CustomStreamWrapper:
        return litellm.completion(
            messages=messages,
            **self._delegate_kwargs(model, optional_params),
        )

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Any = None,
        litellm_params: Any = None,
        logger_fn: Any = None,
        headers: Any = None,
        timeout: float | httpx.Timeout | None = None,
        client: HTTPHandler | None = None,
    ) -> Iterator[GenericStreamingChunk]:
        return litellm.completion(
            messages=messages,
            stream=True,
            **self._delegate_kwargs(model, optional_params),
        )

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Any = None,
        litellm_params: Any = None,
        logger_fn: Any = None,
        headers: Any = None,
        timeout: float | httpx.Timeout | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> Any | ModelResponse | litellm.CustomStreamWrapper:
        return await litellm.acompletion(
            messages=messages,
            **self._delegate_kwargs(model, optional_params),
        )

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Any = None,
        litellm_params: Any = None,
        logger_fn: Any = None,
        headers: Any = None,
        timeout: float | httpx.Timeout | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> AsyncIterator[GenericStreamingChunk]:  # type: ignore[override]
        async for chunk in await litellm.acompletion(
            messages=messages,
            stream=True,
            **self._delegate_kwargs(model, optional_params),
        ):
            yield chunk

    # ── Auth ──────────────────────────────────────────────────────────

    @classmethod
    def _api_key(cls) -> str:
        for env in cls._api_key_envs:
            key = os.environ.get(env, "")
            if key:
                return key
        return ""

    # ── Model discovery ───────────────────────────────────────────────

    @classmethod
    def _discover_models(cls) -> dict[str, str]:
        """Query ``GET {base}/v1/models`` and return model → endpoint map.

        All models default to the ``openai`` (chat/completions) endpoint.
        Raises ``ValueError`` on auth or network failure.
        """
        api_key = cls._api_key()
        if not api_key:
            raise ValueError(f"No API key for {cls.__name__}. Set {' or '.join(cls._api_key_envs)}.")

        headers = {"Authorization": f"Bearer {api_key}"}
        models: dict[str, str] = {}

        try:
            resp = httpx.get(f"{cls._base_url}/v1/models", headers=headers, timeout=10)
            resp.raise_for_status()
            for m in resp.json().get("data", []):
                mid = m.get("id", "")
                if mid:
                    models[mid] = "openai"
        except Exception as exc:
            raise ValueError(f"Failed to discover models from {cls._base_url}/v1/models: {exc}") from exc

        return models

    @classmethod
    def _resolve_endpoint(cls, model: str) -> str:
        """Return ``"openai"`` or ``"anthropic"`` for *model*."""
        if cls._model_cache is None:
            cls._model_cache = cls._discover_models()
        return cls._model_cache.get(model, "openai")

    # ── kwargs builder ────────────────────────────────────────────────

    @classmethod
    def _delegate_kwargs(cls, model: str, optional_params: dict[str, Any]) -> dict[str, Any]:
        endpoint = cls._resolve_endpoint(model)
        params = {k: v for k, v in optional_params.items() if k != "stream"}
        api_key = cls._api_key()

        if endpoint == "anthropic":
            return {
                "model": f"anthropic/{model}",
                "api_base": f"{cls._base_url}/v1/messages",
                "api_key": api_key,
                **params,
            }
        return {
            "model": f"openai/{model}",
            "api_base": f"{cls._base_url}/v1/chat/completions",
            "api_key": api_key,
            **params,
        }


class _OpenCodeZenProvider(_OpenCodeProvider):
    """Routes ``opencode-zen/<model>`` to the Zen API (pay-as-you-go)."""

    _base_url = _OPENCODE_ZEN_BASE_URL
    _api_key_envs = ("OPENCODE_ZEN_API_KEY", "OPENCODE_API_KEY")


class _OpenCodeGoProvider(_OpenCodeProvider):
    """Routes ``opencode-go/<model>`` to the Go API (subscription)."""

    _base_url = _OPENCODE_GO_BASE_URL
    _api_key_envs = ("OPENCODE_GO_API_KEY", "OPENCODE_API_KEY")


class _OpenCodeDummyProvider(CustomLLM):
    """Placeholder for bare ``opencode/`` prefix — raises a clear error."""

    _ERR = "Invalid provider prefix 'opencode/'. Use 'opencode-go/' (subscription) or 'opencode-zen/' (pay-as-you-go) instead."

    def completion(self, model: str, messages: list, **_kw: Any) -> Any:
        raise ValueError(self._ERR)

    def streaming(self, model: str, messages: list, **_kw: Any) -> Any:
        raise ValueError(self._ERR)

    async def acompletion(self, model: str, messages: list, **_kw: Any) -> Any:
        raise ValueError(self._ERR)

    async def astreaming(self, model: str, messages: list, **_kw: Any) -> Any:
        raise ValueError(self._ERR)
