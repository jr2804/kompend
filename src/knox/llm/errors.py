"""Typed error hierarchy for KNOX LLM calls.

Every business module that talks to an LLM must catch one of these
classes — never bare ``Exception``. The classifier
:func:`classify_litellm_error` maps litellm's exception zoo onto this
hierarchy so call sites stay readable.

Design rationale: bare ``except Exception`` swallows real bugs (bad
prompts, missing API keys, code errors) into the same code path as
transient network failures. With typed errors, the failover chain
retries only what is genuinely retriable.
"""

from __future__ import annotations

from typing import Any


class LLMError(Exception):
    """Base class for all KNOX LLM errors.

    Carries optional ``model`` and ``attempts`` metadata for
    observability so logs and tests can assert on them.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        attempts: int = 0,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.attempts = attempts
        self.cause = cause


class TransientError(LLMError):
    """Recoverable failure — rate limit, timeout, 5xx, connection error.

    The failover chain will retry with exponential backoff and, when
    retries are exhausted, advance to the next model in the chain.
    """


class PermanentError(LLMError):
    """Non-recoverable failure for the current request.

    Examples: content policy refusals, malformed prompts, prompt
    rejected by the provider, unparseable JSON. The failover chain
    surfaces these immediately — retrying the same prompt against
    another model will not help.
    """


class ConfigurationError(LLMError):
    """Fatal configuration problem — fail fast.

    Examples: unknown model name, missing API key, invalid provider
    prefix. The failover chain raises this without trying any model.
    """


# ── litellm exception classifier ────────────────────────────────────────

# Names of litellm exception classes that map to each KNOX error type.
# We match by class ``__name__`` to avoid importing litellm at module
# load time (keeps test imports fast and tolerates litellm version drift).
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
        "BadRequestError",
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
        "BadRequestError",  # ambiguous: also permanent — see classify_litellm_error
    }
)


def classify_litellm_error(exc: BaseException, *, model: str | None = None) -> LLMError:
    """Map a raw litellm exception onto KNOX's typed hierarchy.

    Falls back to :class:`TransientError` for unknown exception types so
    the failover chain still gets a chance to recover. If the original
    exception is already an :class:`LLMError`, it is returned unchanged.
    """
    if isinstance(exc, LLMError):
        return exc

    name = type(exc).__name__
    message = str(exc) or name

    # AuthenticationError and similar provider-config errors must fail fast.
    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return ConfigurationError(message, model=model, cause=exc)
    if name == "NotFoundError":
        # Often "model not found" → configuration issue.
        return ConfigurationError(message, model=model, cause=exc)

    if name in _PERMANENT_NAMES:
        return PermanentError(message, model=model, cause=exc)
    if name in _TRANSIENT_NAMES:
        return TransientError(message, model=model, cause=exc)

    # Heuristic: HTTP status code on the exception (litellm sets it)
    status: Any = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status == 429 or 500 <= status < 600:
            return TransientError(message, model=model, cause=exc)
        if status in {400, 422}:
            return PermanentError(message, model=model, cause=exc)
        if status in {401, 403, 404}:
            return ConfigurationError(message, model=model, cause=exc)

    # Unknown — assume transient so the chain still tries to recover.
    return TransientError(message, model=model, cause=exc)
