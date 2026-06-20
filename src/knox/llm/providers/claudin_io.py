"""Custom litellm provider for claudin.io (OpenAI-compatible API).

``claudin-io/<model>`` routes to ``https://api.claudin.io/v1``
(OpenAI-format requests; the SDK appends ``/chat/completions``).
Model names like ``claudinio`` (balanced) or ``claudius`` (premium).
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

_CLAUDIN_IO_BASE_URL = "https://api.claudin.io/v1"
_CLAUDIN_IO_API_KEY_ENV = "CLAUDINIO_API_KEY"


class _ClaudinIOProvider(CustomLLM):
    """Thin wrapper that routes ``claudin-io/<model>`` calls to
    the claudin.io OpenAI-compatible API at https://api.claudin.io.

    Delegates to ``litellm.completion`` / ``litellm.acompletion``
    with the claudin.io ``api_base`` / ``api_key``. The OpenAI SDK
    appends ``/chat/completions`` to ``api_base`` automatically.
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
            **self._delegate_kwargs(model, optional_params, api_key),
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
            **self._delegate_kwargs(model, optional_params, api_key),
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
            **self._delegate_kwargs(model, optional_params, api_key),
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
        if headers is None:
            headers = {}
        async for chunk in await litellm.acompletion(
            messages=messages,
            stream=True,
            **self._delegate_kwargs(model, optional_params, api_key),
        ):
            yield chunk

    @staticmethod
    def _api_key() -> str:
        return os.environ.get(_CLAUDIN_IO_API_KEY_ENV, "")

    @staticmethod
    def _delegate_kwargs(model: str, optional_params: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
        # If api_key is an unresolved env reference (starts with "$"), ignore it
        # and fall back to the provider's own env var — preventing a literal
        # "$CLAUDINIO_API_KEY" from being sent as the credential.
        if api_key is not None and api_key.startswith("$"):
            api_key = None
        params = {k: v for k, v in optional_params.items() if k != "stream"}
        return {
            "model": f"openai/{model}",
            "api_base": _CLAUDIN_IO_BASE_URL,
            "api_key": api_key or _ClaudinIOProvider._api_key(),
            **params,
        }
