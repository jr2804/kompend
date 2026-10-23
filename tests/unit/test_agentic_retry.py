"""Classification of failed LLM calls and the generation retry policy."""

from __future__ import annotations

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError

from kompend.compilation.flows.generation import (
    _UPSTREAM_TRANSIENT_BACKOFF_BASE,
    _UPSTREAM_TRANSIENT_MAX_RETRIES,
)
from kompend.llm.errors import is_transient_error


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504, 529])
def test_retryable_statuses_are_transient(status: int) -> None:
    assert is_transient_error(ModelHTTPError(status, "model", body={})) is True


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_request_errors_are_not_transient(status: int) -> None:
    """A 4xx is a property of the request - a "timed out" wording does not change that."""
    assert is_transient_error(ModelHTTPError(status, "model", body={"message": "Request timed out."})) is False


def test_transport_failures_are_transient() -> None:
    assert is_transient_error(httpx.ReadTimeout("slow")) is True
    assert is_transient_error(httpx.ConnectError("refused")) is True
    assert is_transient_error(ConnectionError("reset")) is True
    assert is_transient_error(TimeoutError()) is True


def test_cause_chain_is_followed() -> None:
    wrapper = RuntimeError("agent failed")
    wrapper.__cause__ = httpx.ReadTimeout("slow")
    assert is_transient_error(wrapper) is True


def test_status_attribute_is_respected() -> None:
    class ProviderError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__("boom")
            self.status_code = status_code

    assert is_transient_error(ProviderError(503)) is True
    assert is_transient_error(ProviderError(422)) is False


def test_plain_errors_are_not_transient() -> None:
    assert is_transient_error(ValueError("Request timed out.")) is False


def test_retry_constants_have_expected_shape() -> None:
    assert _UPSTREAM_TRANSIENT_MAX_RETRIES >= 1
    assert 0.5 <= _UPSTREAM_TRANSIENT_BACKOFF_BASE <= 5.0
    worst_case = sum(_UPSTREAM_TRANSIENT_BACKOFF_BASE * (2 ** (attempt - 1)) for attempt in range(1, _UPSTREAM_TRANSIENT_MAX_RETRIES + 1))
    assert worst_case < 10.0, f"total worst-case backoff too long: {worst_case:.1f}s"
