"""Typed error hierarchy for KNOX LLM calls.

Every business module that talks to an LLM must catch one of these
classes — never bare ``Exception``. Use :func:`~knox.llm.classifier.classify_litellm_error`
to map raw litellm exceptions onto this hierarchy.

Design rationale: bare ``except Exception`` swallows real bugs (bad
prompts, missing API keys, code errors) into the same code path as
transient network failures. With typed errors, the failover chain
retries only what is genuinely retriable.
"""

from __future__ import annotations


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


class AmbiguousError(LLMError):
    """Context-dependent failure — could be transient or permanent.

    Raised by :func:`~knox.llm.classifier.classify_litellm_error` when
    the exception type alone is insufficient to determine retryability.
    The failover chain inspects available metadata (e.g. HTTP status
    code, error message) to decide whether to retry or abort.

    Example: ``BadRequestError`` can represent a malformed prompt
    (permanent) or an auth format issue (configuration) depending on
    context.
    """
