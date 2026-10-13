"""HTTP retry transport with tenacity for transient LLM API errors.

Creates a retrying ``httpx.AsyncClient`` that wraps requests in tenacity's
exponential-backoff retry loop, catching 429/502/503/504 and timeouts.
Used by :func:`kompend.llm.models.build_model` to inject into native providers.
"""

from __future__ import annotations

import logging

import httpx
import tenacity
from tenacity import (
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# -- retryable status codes ----------------------------------------------------

RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})

logger = logging.getLogger(__name__)
"""HTTP status codes on which we retry the request."""

DEFAULT_MAX_ATTEMPTS = 3
"""Default number of retry attempts (3 = 1 initial + 2 retries)."""

DEFAULT_RETRY_TIMEOUT: float = 60.0
"""Default total timeout in seconds for the retrying transport."""

DEFAULT_RETRY_CONNECT: float = 10.0
"""Default connect timeout in seconds."""


# -- retry predicate -----------------------------------------------------------


def _retryable_error(exc: BaseException) -> bool:
    """Return True when the exception should trigger a retry."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(
        exc,
        (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError),
    )


# -- tenacity retry config -----------------------------------------------------


def _log_retry_wait(retry_state: RetryCallState) -> None:
    """Log the upcoming backoff sleep before a retry (replaces before_sleep_log,
    whose LoggerProtocol stubs don't accept stdlib logging.Logger).
    """
    sleep = retry_state.next_action.sleep if retry_state.next_action else 0.0
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retrying LLM request in %.1fs (attempt %d): %r",
        sleep,
        retry_state.attempt_number,
        exc,
    )


_tenacity_retry = tenacity.AsyncRetrying(
    stop=stop_after_attempt(DEFAULT_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception(_retryable_error),
    before_sleep=_log_retry_wait,
    reraise=True,
)


# -- httpx transport wrapper ---------------------------------------------------


class AsyncTenacityTransport(httpx.AsyncBaseTransport):
    """An ``httpx`` transport that wraps requests in tenacity retry logic.

    Catches 429/502/503/504 responses and timeouts, retrying with
    exponential backoff.  Delegates the actual I/O to a base transport.

    Parameters
    ----------
    base_transport:
        The upstream transport to wrap.  Defaults to
        ``httpx.AsyncHTTPTransport()``.
    max_attempts:
        Override the default retry count (1 initial + retries).
    """

    def __init__(
        self,
        base_transport: httpx.AsyncBaseTransport | None = None,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        if base_transport is None:
            base_transport = httpx.AsyncHTTPTransport()
        self._base = base_transport

        if max_attempts != DEFAULT_MAX_ATTEMPTS:
            self._retry = tenacity.AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                retry=retry_if_exception(_retryable_error),
                before_sleep=_log_retry_wait,
                reraise=True,
            )
        else:
            self._retry = _tenacity_retry

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._retry(self._attempt, request)

    async def aclose(self) -> None:
        await self._base.aclose()

    async def _attempt(self, request: httpx.Request) -> httpx.Response:
        return await self._base.handle_async_request(request)


# -- factory -------------------------------------------------------------------


def create_retrying_http_client(
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout: float = DEFAULT_RETRY_TIMEOUT,
    connect: float = DEFAULT_RETRY_CONNECT,
    headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` that retries transient LLM API errors.

    Each call creates a new client instance.  The caller is responsible for
    calling ``await client.aclose()`` when finished, or using the client as
    an async context manager.
    """
    transport = AsyncTenacityTransport(max_attempts=max_attempts)
    return httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(timeout=timeout, connect=connect),
        headers=headers,
    )
