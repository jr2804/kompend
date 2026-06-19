"""Custom litellm provider for claudin.io (Anthropic-compatible API).

``claudin-io/claude-3.5-sonnet`` routes to ``anthropic/claude-3.5-sonnet``
at ``https://api.claudin.io/v1/messages``.
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

_CLAUDIN_IO_BASE_URL = "https://api.claudin.io/v1/messages"
_CLAUDIN_IO_API_KEY_ENV = "CLAUDINIO_API_KEY"


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
    ) -> AsyncIterator[GenericStreamingChunk]:  # type: ignore[override]
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
