"""Tests for knox.llm.http_client — AsyncTenacityTransport and factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from knox.llm.http_client import (
    RETRYABLE_STATUS_CODES,
    AsyncTenacityTransport,
    create_retrying_http_client,
)


class TestRetryableStatusCodes:
    """Verify the retryable status code set."""

    def test_contains_expected_codes(self) -> None:
        assert 429 in RETRYABLE_STATUS_CODES
        assert 502 in RETRYABLE_STATUS_CODES
        assert 503 in RETRYABLE_STATUS_CODES
        assert 504 in RETRYABLE_STATUS_CODES

    def test_excludes_non_retryable(self) -> None:
        assert 400 not in RETRYABLE_STATUS_CODES
        assert 401 not in RETRYABLE_STATUS_CODES
        assert 403 not in RETRYABLE_STATUS_CODES
        assert 404 not in RETRYABLE_STATUS_CODES
        assert 500 not in RETRYABLE_STATUS_CODES


class TestAsyncTenacityTransport:
    """Unit tests for the retrying transport."""

    async def test_successful_request_passthrough(self) -> None:
        """A successful response is returned as-is."""
        inner = AsyncMock(spec=httpx.AsyncBaseTransport)
        expected = httpx.Response(200, text="ok")
        inner.handle_async_request.return_value = expected

        transport = AsyncTenacityTransport(inner)
        request = httpx.Request("POST", "https://example.com/v1/chat")
        response = await transport.handle_async_request(request)

        assert response.status_code == 200
        assert response.text == "ok"
        inner.handle_async_request.assert_awaited_once_with(request)

    async def test_retries_on_429(self) -> None:
        """A 429 response triggers a retry."""
        inner = AsyncMock(spec=httpx.AsyncBaseTransport)
        inner.handle_async_request.side_effect = [
            httpx.HTTPStatusError(
                "Too Many Requests",
                request=httpx.Request("POST", "https://example.com"),
                response=httpx.Response(429),
            ),
            httpx.Response(200, text="ok"),
        ]

        transport = AsyncTenacityTransport(inner, max_attempts=2)
        request = httpx.Request("POST", "https://example.com/v1/chat")
        response = await transport.handle_async_request(request)

        assert response.status_code == 200
        assert response.text == "ok"
        assert inner.handle_async_request.await_count == 2

    async def test_retries_on_503(self) -> None:
        """A 503 response triggers a retry."""
        inner = AsyncMock(spec=httpx.AsyncBaseTransport)
        inner.handle_async_request.side_effect = [
            httpx.HTTPStatusError(
                "Service Unavailable",
                request=httpx.Request("POST", "https://example.com"),
                response=httpx.Response(503),
            ),
            httpx.Response(200, text="ok"),
        ]

        transport = AsyncTenacityTransport(inner, max_attempts=2)
        request = httpx.Request("POST", "https://example.com/v1/chat")
        response = await transport.handle_async_request(request)

        assert response.status_code == 200
        assert inner.handle_async_request.await_count == 2

    async def test_retries_on_timeout(self) -> None:
        """A timeout exception triggers a retry."""
        inner = AsyncMock(spec=httpx.AsyncBaseTransport)
        inner.handle_async_request.side_effect = [
            httpx.TimeoutException("timed out"),
            httpx.Response(200, text="ok"),
        ]

        transport = AsyncTenacityTransport(inner, max_attempts=2)
        request = httpx.Request("POST", "https://example.com/v1/chat")
        response = await transport.handle_async_request(request)

        assert response.status_code == 200
        assert inner.handle_async_request.await_count == 2

    async def test_retries_on_connect_error(self) -> None:
        """A connect error triggers a retry."""
        inner = AsyncMock(spec=httpx.AsyncBaseTransport)
        inner.handle_async_request.side_effect = [
            httpx.ConnectError("connection refused"),
            httpx.Response(200, text="ok"),
        ]

        transport = AsyncTenacityTransport(inner, max_attempts=2)
        request = httpx.Request("POST", "https://example.com/v1/chat")
        response = await transport.handle_async_request(request)

        assert response.status_code == 200
        assert inner.handle_async_request.await_count == 2

    async def test_does_not_retry_on_400(self) -> None:
        """A 400 (bad request) is not retried — it propagates."""
        inner = AsyncMock(spec=httpx.AsyncBaseTransport)
        inner.handle_async_request.side_effect = httpx.HTTPStatusError(
            "Bad Request",
            request=httpx.Request("POST", "https://example.com"),
            response=httpx.Response(400),
        )

        transport = AsyncTenacityTransport(inner, max_attempts=2)
        request = httpx.Request("POST", "https://example.com/v1/chat")

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await transport.handle_async_request(request)
        assert exc_info.value.response.status_code == 400
        # Only one attempt — 400 is not retryable.
        assert inner.handle_async_request.await_count == 1

    async def test_does_not_retry_on_401(self) -> None:
        """A 401 (auth error) is not retried."""
        inner = AsyncMock(spec=httpx.AsyncBaseTransport)
        inner.handle_async_request.side_effect = httpx.HTTPStatusError(
            "Unauthorized",
            request=httpx.Request("POST", "https://example.com"),
            response=httpx.Response(401),
        )

        transport = AsyncTenacityTransport(inner, max_attempts=2)
        request = httpx.Request("POST", "https://example.com/v1/chat")

        with pytest.raises(httpx.HTTPStatusError):
            await transport.handle_async_request(request)
        assert inner.handle_async_request.await_count == 1

    async def test_exhausts_retries(self) -> None:
        """After exhausting retries, the last error propagates."""
        inner = AsyncMock(spec=httpx.AsyncBaseTransport)
        inner.handle_async_request.side_effect = httpx.HTTPStatusError(
            "Too Many Requests",
            request=httpx.Request("POST", "https://example.com"),
            response=httpx.Response(429),
        )

        transport = AsyncTenacityTransport(inner, max_attempts=3)
        request = httpx.Request("POST", "https://example.com/v1/chat")

        with pytest.raises(httpx.HTTPStatusError):
            await transport.handle_async_request(request)
        assert inner.handle_async_request.await_count == 3

    async def test_aclose_delegates(self) -> None:
        """aclose() delegates to the base transport."""
        inner = AsyncMock(spec=httpx.AsyncBaseTransport)
        transport = AsyncTenacityTransport(inner)
        await transport.aclose()
        inner.aclose.assert_awaited_once()


class TestCreateRetryingHttpClient:
    """Tests for the factory function."""

    async def test_returns_async_client(self) -> None:
        client = create_retrying_http_client()
        assert isinstance(client, httpx.AsyncClient)
        await client.aclose()

    async def test_has_retrying_transport(self) -> None:
        client = create_retrying_http_client()
        assert isinstance(client._transport, AsyncTenacityTransport)
        await client.aclose()

    async def test_custom_max_attempts(self) -> None:
        client = create_retrying_http_client(max_attempts=5)
        transport = client._transport
        assert isinstance(transport, AsyncTenacityTransport)
        await client.aclose()

    async def test_custom_timeout(self) -> None:
        client = create_retrying_http_client(timeout=120.0, connect=30.0)
        assert client.timeout is not None
        assert client.timeout.connect == 30.0
        await client.aclose()
