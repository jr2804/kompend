"""Shared base for thin litellm providers that wrap an OpenAI-compatible API.

Most custom providers in KNOX follow the same pattern: accept a custom
model prefix (``claudin-io/``, ``ollama-cloud/``, ``opencode-zen/``,
``opencode-go/``), strip the prefix, then delegate to litellm with a
different model prefix, ``api_base``, and ``api_key``.

:class:`_CustomProvider` implements the four ``CustomLLM`` methods
(``completion``, ``streaming``, ``acompletion``, ``astreaming``) in
terms of a single abstract override — :meth:`_delegate_kwargs`.

Subclasses only need to set class-level constants and override
``_delegate_kwargs`` (and optionally ``_has_streaming`` if streaming
must be faked via a non-streaming call).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import litellm
from litellm import CustomLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import GenericStreamingChunk, ModelResponse


class _CustomProvider(CustomLLM):
    """Base for thin litellm custom providers.

    Subclasses override :meth:`_delegate_kwargs` to return the
    ``model`` (with the target litellm prefix), ``api_base``,
    ``api_key``, and any extra kwargs for the downstream call.
    """

    # ponytail: subclasses typically use class-level constants for
    # these — override at the class level, not via __init_subclass__.
    BASE_URL: str = ""
    API_KEY_ENV: str = ""
    MODEL_PREFIX: str = "openai"  # litellm's internal handler prefix

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
        return await litellm.acompletion(
            messages=messages,
            **self._delegate_kwargs(model, optional_params, api_key),
        )

    async def astreaming(  # noqa: ANN202 — noqa: litellm's base signature is typed wrong for async generators
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
    ):
        async for chunk in await litellm.acompletion(
            messages=messages,
            stream=True,
            **self._delegate_kwargs(model, optional_params, api_key),
        ):
            yield chunk

    @classmethod
    def _delegate_kwargs(
        cls,
        model: str,
        optional_params: dict[str, Any],
        api_key: Any = None,
    ) -> dict[str, Any]:
        """Build kwargs for the downstream ``litellm.completion`` call.

        Subclasses MUST override this.  The default raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def _api_key(cls) -> str:
        """Read the API key from the environment variable named by
        :attr:`API_KEY_ENV`.

        Returns ``""`` when the variable is unset.
        """
        return os.environ.get(cls.API_KEY_ENV, "")

    @classmethod
    def _short_model(cls, model: str) -> str:
        """Strip the provider prefix, e.g.
        ``"ollama-cloud/deepseek-v4-flash"`` → ``"deepseek-v4-flash"``.
        """
        return model.split("/", 1)[-1] if "/" in model else model

    @classmethod
    def _resolve_api_key(cls, api_key: Any) -> str:
        """Return the effective API key.

        If *api_key* is a raw ``$ENV_VAR`` reference (from a config
        file where the user wrote ``"$OLLAMA_API_KEY"`` literally),
        ignore it and fall back to :meth:`_api_key`.  When *api_key*
        is ``None`` or empty, also fall back.
        """
        if api_key is not None and (isinstance(api_key, str) and api_key.startswith("$")):
            api_key = None
        return api_key or cls._api_key()

    @staticmethod
    def _response_to_chunk(
        response: ModelResponse,
    ) -> Iterator[GenericStreamingChunk]:
        """Wrap a non-streaming :class:`ModelResponse` as a single terminal
        streaming chunk.
        """
        choices = response.get("choices", [{}])
        choice = choices[0] if choices else {}
        yield GenericStreamingChunk(
            text=choice.get("text", "") or choice.get("message", {}).get("content", ""),
            finish_reason="stop",
            is_finished=True,
            usage=None,
        )
