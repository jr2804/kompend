"""Tests for KNOX LLM error classification."""

from __future__ import annotations

from knox.llm.errors import (
    ConfigurationError,
    LLMError,
    PermanentError,
    TransientError,
    classify_litellm_error,
)


class _FakeExc(Exception):
    """Stand-in for a litellm exception with controllable class name."""


def _make_exc(class_name: str, *, status_code: int | None = None) -> Exception:
    """Build a fresh exception class so __name__ matches what we want."""
    cls = type(class_name, (_FakeExc,), {})
    exc = cls("boom")
    if status_code is not None:
        exc.status_code = status_code  # type: ignore[attr-defined]
    return exc


class TestErrorHierarchy:
    def test_typed_errors_inherit_llmerror(self) -> None:
        assert issubclass(TransientError, LLMError)
        assert issubclass(PermanentError, LLMError)
        assert issubclass(ConfigurationError, LLMError)

    def test_error_carries_model_and_attempts(self) -> None:
        err = TransientError("rate-limited", model="openai/gpt-4o", attempts=3)
        assert err.model == "openai/gpt-4o"
        assert err.attempts == 3
        assert str(err) == "rate-limited"

    def test_cause_preserved(self) -> None:
        original = RuntimeError("boom")
        err = TransientError("wrapped", cause=original)
        assert err.cause is original


class TestClassifyLitellmError:
    def test_rate_limit_is_transient(self) -> None:
        err = classify_litellm_error(_make_exc("RateLimitError"), model="m")
        assert isinstance(err, TransientError)
        assert err.model == "m"

    def test_timeout_is_transient(self) -> None:
        err = classify_litellm_error(_make_exc("Timeout"))
        assert isinstance(err, TransientError)

    def test_connection_error_is_transient(self) -> None:
        err = classify_litellm_error(_make_exc("APIConnectionError"))
        assert isinstance(err, TransientError)

    def test_bad_request_is_permanent(self) -> None:
        err = classify_litellm_error(_make_exc("BadRequestError"))
        assert isinstance(err, PermanentError)

    def test_content_policy_is_permanent(self) -> None:
        err = classify_litellm_error(_make_exc("ContentPolicyViolationError"))
        assert isinstance(err, PermanentError)

    def test_context_window_is_permanent(self) -> None:
        err = classify_litellm_error(_make_exc("ContextWindowExceededError"))
        assert isinstance(err, PermanentError)

    def test_authentication_is_configuration(self) -> None:
        err = classify_litellm_error(_make_exc("AuthenticationError"))
        assert isinstance(err, ConfigurationError)

    def test_permission_denied_is_configuration(self) -> None:
        err = classify_litellm_error(_make_exc("PermissionDeniedError"))
        assert isinstance(err, ConfigurationError)

    def test_not_found_is_configuration(self) -> None:
        err = classify_litellm_error(_make_exc("NotFoundError"))
        assert isinstance(err, ConfigurationError)

    def test_unknown_exception_defaults_to_transient(self) -> None:
        """Unknown exception types are treated as transient so the chain still recovers."""
        err = classify_litellm_error(_make_exc("WeirdNewError"))
        assert isinstance(err, TransientError)

    def test_status_code_429_is_transient(self) -> None:
        err = classify_litellm_error(_make_exc("HTTPError", status_code=429))
        assert isinstance(err, TransientError)

    def test_status_code_5xx_is_transient(self) -> None:
        err = classify_litellm_error(_make_exc("HTTPError", status_code=503))
        assert isinstance(err, TransientError)

    def test_status_code_400_is_permanent(self) -> None:
        err = classify_litellm_error(_make_exc("HTTPError", status_code=400))
        assert isinstance(err, PermanentError)

    def test_status_code_401_is_configuration(self) -> None:
        err = classify_litellm_error(_make_exc("HTTPError", status_code=401))
        assert isinstance(err, ConfigurationError)

    def test_status_code_403_is_configuration(self) -> None:
        err = classify_litellm_error(_make_exc("HTTPError", status_code=403))
        assert isinstance(err, ConfigurationError)

    def test_llmerror_passthrough(self) -> None:
        """If the exception is already typed, return it unchanged."""
        original = PermanentError("already classified", model="m")
        err = classify_litellm_error(original)
        assert err is original
