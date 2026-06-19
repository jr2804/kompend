"""Custom litellm provider registrations for KNOX.

Registers providers that litellm does not ship natively.  This module
is imported at the top of :mod:`knox.llm.client` so the providers are
always available before any completion call is made.

Registered providers
--------------------
``claudin-io``
    Anthropic-compatible API at https://api.claudin.io.  Delegates
    to the ``anthropic`` provider but routes to the claudin.io endpoint.

    Model strings follow Anthropic's naming:
    ``claudin-io/claude-3.5-sonnet``, ``claudin-io/claude-3-haiku``, etc.

    Example::

        default_model = "claudin-io/claude-3.5-sonnet"

    Authentication: set ``CLAUDINIO_API_KEY`` in your environment or
    configure ``compile.llm.api_key``.  The env variable is checked first.

``ollama-cloud``
    Ollama's cloud service (https://ollama.com).  Mirrors the local
    ``ollama`` provider but routes every call to the cloud API using
    the ``OLLAMA_API_KEY`` environment variable for authentication.

    Model strings follow the same convention as the local provider:
    ``ollama-cloud/<model-name>``

    Example::

        default_model = "ollama-cloud/deepseek-v4-flash"

    Authentication: set ``OLLAMA_API_KEY`` in your environment.
    Context window: configure ``context_window_cap`` in the config
    section, or rely on ``fallback_max_input`` (8192) as the safe
    default — the cloud API does not expose ``/api/show``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx
import litellm
from litellm import CustomLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk, ModelResponse

logger = logging.getLogger(__name__)

_CLAUDIN_IO_BASE_URL = "https://api.claudin.io/v1/messages"
_CLAUDIN_IO_API_KEY_ENV = "CLAUDINIO_API_KEY"
_OLLAMA_CLOUD_BASE_URL = "https://ollama.com"
_OLLAMA_CLOUD_API_KEY_ENV = "OLLAMA_API_KEY"


class _ClaudinIOProvider(CustomLLM):
    """Thin wrapper that routes ``claudin-io/<model>`` calls to
    the claudin.io Anthropic-compatible API at https://api.claudin.io.

    litellm receives the model name *without* the provider prefix, so
    ``model`` here is e.g. ``"claude-3.5-sonnet"``.

    The handler delegates to ``litellm.completion`` / ``litellm.acompletion``
    with the ``anthropic/`` prefix and the claudin.io ``api_base`` / ``api_key``,
    preserving all optional parameters (streaming, temperature, etc.).
    """

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
        if headers is None:
            headers = {}
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
        if headers is None:
            headers = {}
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
        if headers is None:
            headers = {}
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
    ) -> AsyncIterator[GenericStreamingChunk]:  # type: ignore[override]  # litellm base declares AsyncIterator but async gen returns AsyncGenerator  # ty:ignore[invalid-method-override]
        if headers is None:
            headers = {}
        async for chunk in await litellm.acompletion(
            messages=messages,
            stream=True,
            **self._delegate_kwargs(model, optional_params),
        ):
            yield chunk

    @staticmethod
    def _api_key() -> str:
        return os.environ.get(_CLAUDIN_IO_API_KEY_ENV, "")

    @staticmethod
    def _delegate_kwargs(model: str, optional_params: dict[str, Any]) -> dict[str, Any]:
        params = {k: v for k, v in optional_params.items() if k != "stream"}
        return {
            "model": f"anthropic/{model}",
            "api_base": _CLAUDIN_IO_BASE_URL,
            "api_key": _ClaudinIOProvider._api_key(),
            **params,
        }


class _OllamaCloudProvider(CustomLLM):
    """Thin wrapper that routes ``ollama-cloud/<model>`` calls to
    Ollama's cloud API at https://ollama.com.

    litellm receives the model name *without* the provider prefix, so
    ``model`` here is e.g. ``"deepseek-v4-flash"``.

    The handler delegates to ``litellm.completion`` / ``litellm.acompletion``
    with the ``ollama_chat/`` prefix and the cloud ``api_base`` / ``api_key``,
    preserving all optional parameters (streaming, temperature, etc.).
    """

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
        if headers is None:
            headers = {}
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
        if headers is None:
            headers = {}
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
        if headers is None:
            headers = {}
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
    ) -> AsyncIterator[GenericStreamingChunk]:  # type: ignore[override]  # litellm base declares AsyncIterator but async gen returns AsyncGenerator  # ty:ignore[invalid-method-override]
        if headers is None:
            headers = {}
        async for chunk in await litellm.acompletion(
            messages=messages,
            stream=True,
            **self._delegate_kwargs(model, optional_params),
        ):
            yield chunk

    @staticmethod
    def _api_key() -> str:
        return os.environ.get(_OLLAMA_CLOUD_API_KEY_ENV, "")

    @staticmethod
    def _delegate_kwargs(model: str, optional_params: dict[str, Any]) -> dict[str, Any]:
        params = {k: v for k, v in optional_params.items() if k != "stream"}
        return {
            "model": f"ollama_chat/{model}",
            "api_base": _OLLAMA_CLOUD_BASE_URL,
            "api_key": _OllamaCloudProvider._api_key(),
            **params,
        }


def _register_custom_providers() -> None:
    """Register all Knox custom providers into litellm.

    Idempotent — safe to call multiple times.
    """
    # Register claudin-io provider
    provider_name = "claudin-io"
    already = [e for e in (litellm.custom_provider_map or []) if e.get("provider") == provider_name]
    if not already:
        handler = _ClaudinIOProvider()
        if litellm.custom_provider_map is None:
            litellm.custom_provider_map = []
        litellm.custom_provider_map.append({"provider": provider_name, "custom_handler": handler})
        logger.debug("Registered custom litellm provider: %s", provider_name)

    # Register ollama-cloud provider
    provider_name = "ollama-cloud"
    already = [e for e in (litellm.custom_provider_map or []) if e.get("provider") == provider_name]
    if already:
        return

    handler = _OllamaCloudProvider()
    if litellm.custom_provider_map is None:
        litellm.custom_provider_map = []
    litellm.custom_provider_map.append({"provider": provider_name, "custom_handler": handler})
    logger.debug("Registered custom litellm provider: %s", provider_name)


# Auto-register on import.
_register_custom_providers()
