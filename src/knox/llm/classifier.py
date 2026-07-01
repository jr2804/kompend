"""Litellm exception → KNOX error type mapper."""

from __future__ import annotations

from knox.llm.errors import (
    AmbiguousError,
    ConfigurationError,
    LLMError,
    PermanentError,
    TransientError,
)

# litellm exception __name__ → KNOX error type. litellm exceptions don't
# share a KNOX-like hierarchy, so we map by name.
_NAME_MAP: dict[str, type[LLMError]] = {
    # Transient — retryable
    "RateLimitError": TransientError,
    "Timeout": TransientError,
    "APIConnectionError": TransientError,
    "APIError": TransientError,
    "ServiceUnavailableError": TransientError,
    "InternalServerError": TransientError,
    # Permanent — don't retry
    "ContentPolicyViolationError": PermanentError,
    "ContextWindowExceededError": PermanentError,
    "InvalidRequestError": PermanentError,
    "UnprocessableEntityError": PermanentError,
    # Configuration — user-fixable
    "AuthenticationError": ConfigurationError,
    "PermissionDeniedError": ConfigurationError,
    "NotFoundError": ConfigurationError,
    # Ambiguous — let failover chain decide
    "BadRequestError": AmbiguousError,
}


def classify_litellm_error(exc: BaseException, *, model: str | None = None) -> LLMError:
    """Map a litellm exception onto KNOX's typed hierarchy.

    Falls back to :class:`TransientError` so the failover chain gets a chance.
    If *exc* is already an :class:`LLMError`, it is returned unchanged.
    """
    if isinstance(exc, LLMError):
        return exc

    name = type(exc).__name__
    msg = str(exc) or name

    category = _NAME_MAP.get(name)
    # 'not found' in the message is likely a wrong model name → ConfigurationError
    if category is None and "not found" in msg.lower():
        category = ConfigurationError
    if category is None:
        category = TransientError

    return category(msg, model=model, cause=exc)
