"""Litellm exception → KNOX error type mapper.

Separated from :mod:`knox.llm.errors` so the error type definitions
stay decoupled from the classification logic. This makes it easier to
test the classification in isolation and to maintain the mapping as
litellm evolves.
"""

from __future__ import annotations

from typing import Any

from knox.llm.errors import (
    AmbiguousError,
    ConfigurationError,
    LLMError,
    PermanentError,
    TransientError,
)

# ── litellm exception name sets ──────────────────────────────────────

# Names of litellm exception classes that map to each KNOX error type.
# We match by class ``__name__`` to avoid importing litellm at module
# load time (keeps test imports fast and tolerates litellm version drift).

_AMBIGUOUS_NAMES: frozenset[str] = frozenset(
    {
        "BadRequestError",  # can be permanent (bad prompt) or configuration (auth format)
    }
)

_TRANSIENT_NAMES: frozenset[str] = frozenset(
    {
        "RateLimitError",
        "Timeout",
        "APIConnectionError",
        "APIError",  # generic 5xx-ish
        "ServiceUnavailableError",
        "InternalServerError",
    }
)

_PERMANENT_NAMES: frozenset[str] = frozenset(
    {
        "ContentPolicyViolationError",
        "ContextWindowExceededError",
        "InvalidRequestError",
        "UnprocessableEntityError",
    }
)

_CONFIGURATION_NAMES: frozenset[str] = frozenset(
    {
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
    }
)

# ── HTTP status code map ─────────────────────────────────────────────

_STATUS_ERROR_MAP: dict[int, type[LLMError]] = {
    429: TransientError,
    400: PermanentError,
    422: PermanentError,
    401: ConfigurationError,
    403: ConfigurationError,
    404: ConfigurationError,
}


def _classify_http_status(status: int) -> type[LLMError] | None:
    """Map HTTP status code to LLM error type, or None if unknown."""
    if 500 <= status < 600:
        return TransientError
    return _STATUS_ERROR_MAP.get(status)


# ── classifier ───────────────────────────────────────────────────────


def _make(msg: str, category: type[LLMError], model: str | None, cause: BaseException) -> LLMError:
    return category(msg, model=model, cause=cause)


def classify_litellm_error(exc: BaseException, *, model: str | None = None) -> LLMError:
    """Map a raw litellm exception onto KNOX's typed hierarchy.

    Falls back to :class:`TransientError` for unknown exception types so
    the failover chain still gets a chance to recover. If the original
    exception is already an :class:`LLMError`, it is returned unchanged.

    Exceptions whose type alone is ambiguous (e.g. ``BadRequestError``)
    are returned as :class:`AmbiguousError`, allowing the failover chain
    to inspect metadata before deciding retry vs. fail.
    """
    if isinstance(exc, LLMError):
        return exc

    name = type(exc).__name__
    message = str(exc) or name

    # 'not found' in the message is a configuration error (wrong model name,
    # missing API endpoint) — fail fast rather than retry endlessly.
    if 'not found' in message.lower():
        return _make(message, ConfigurationError, model, exc)

    # Resolve the target category from exception name first, then status code.
    category: type[LLMError]
    if name in _AMBIGUOUS_NAMES:
        category = AmbiguousError
    elif name in _CONFIGURATION_NAMES:
        category = ConfigurationError
    elif name in _PERMANENT_NAMES:
        category = PermanentError
    elif name in _TRANSIENT_NAMES:
        category = TransientError
    else:
        # Fallback: check HTTP status code on generic exceptions.
        status: Any = getattr(exc, "status_code", None)
        if isinstance(status, int):
            mapped = _classify_http_status(status)
            category = mapped if mapped is not None else TransientError
        else:
            category = TransientError

    return _make(message, category, model, exc)
