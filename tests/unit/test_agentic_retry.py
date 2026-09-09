"""Unit tests for the upstream transient-timeout retry heuristic in
``knox.compilation.flows.generation``.

The heuristic catches ``pydantic_ai.exceptions.ModelHTTPError`` whose
``status_code`` is below 500 and whose body contains ``"timed out"`` and
retries the same agentic-generation pass with exponential backoff before
counting it as a failed pass.
"""

from __future__ import annotations

from pydantic_ai.exceptions import ModelHTTPError

from knox.compilation.flows.generation import (
    _UPSTREAM_TRANSIENT_BACKOFF_BASE,
    _UPSTREAM_TRANSIENT_MAX_RETRIES,
    _is_upstream_transient_timeout,
)


def test_upstream_transient_timeout_matches_4xx_with_timed_out_body() -> None:
    """4xx + 'timed out' body → transient, retry-worthy."""
    exc = ModelHTTPError(400, "gemma", body={"message": "Request timed out."})
    assert _is_upstream_transient_timeout(exc) is True


def test_upstream_transient_timeout_matches_lowercase_body() -> None:
    """Match is case-insensitive on the body substring."""
    exc = ModelHTTPError(408, "gemma", body={"message": "request TIMED OUT."})
    assert _is_upstream_transient_timeout(exc) is True


def test_upstream_transient_timeout_rejects_5xx() -> None:
    """5xx is hard-fail territory; do not retry, surface upstream."""
    exc = ModelHTTPError(503, "gemma", body={"message": "Request timed out."})
    assert _is_upstream_transient_timeout(exc) is False


def test_upstream_transient_timeout_rejects_non_timeout_body() -> None:
    """4xx without 'timed out' body is a real client error; do not retry."""
    exc = ModelHTTPError(400, "gemma", body={"message": "Invalid tool_choice"})
    assert _is_upstream_transient_timeout(exc) is False


def test_upstream_transient_timeout_handles_string_body() -> None:
    """Some gateways serialize the body as a bare string."""
    exc = ModelHTTPError(400, "gemma", body="Request timed out.")
    assert _is_upstream_transient_timeout(exc) is True


def test_upstream_transient_timeout_handles_nested_error_key() -> None:
    """Body may use ``error`` instead of ``message``."""
    exc = ModelHTTPError(400, "gemma", body={"error": "Request timed out."})
    assert _is_upstream_transient_timeout(exc) is True


def test_upstream_transient_timeout_rejects_non_model_http_error() -> None:
    """Only ModelHTTPError qualifies; ValueError, RuntimeError, etc. do not."""
    assert _is_upstream_transient_timeout(ValueError("Request timed out.")) is False
    assert _is_upstream_transient_timeout(RuntimeError("boom")) is False


def test_retry_constants_have_expected_shape() -> None:
    """Sanity: backoff base is small enough to keep total wait < 10s."""
    # 1 + 2 + 4 = 7s worst-case at base=1.0, max 3 retries.
    assert _UPSTREAM_TRANSIENT_MAX_RETRIES >= 1
    assert 0.5 <= _UPSTREAM_TRANSIENT_BACKOFF_BASE <= 5.0
    worst_case = sum(_UPSTREAM_TRANSIENT_BACKOFF_BASE * (2 ** (attempt - 1)) for attempt in range(1, _UPSTREAM_TRANSIENT_MAX_RETRIES + 1))
    assert worst_case < 10.0, f"total worst-case backoff too long: {worst_case:.1f}s"
