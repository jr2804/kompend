"""Exception formatting helpers for log/error messages.

Centralised so that formatting stays consistent across the codebase
and so that smarter formatting (e.g. extracting the most relevant line
from a multi-line stacktrace) can be added in one place.
"""

from __future__ import annotations


class LLMOutputError(Exception):
    """Raised when an LLM cannot produce valid structured output.

    The model is considered not suitable for compilation — no fallback,
    no retry.  The caller should abort compilation with a clear message.
    """


class LLMTransientError(LLMOutputError):
    """Raised when an LLM call failed for a reason worth repeating.

    Timeouts, dropped connections and retryable server statuses (see
    :func:`kompend.llm.errors.is_transient_error`) say nothing about the
    model's suitability: the same request may well succeed on the next
    attempt. Callers catch this class before :class:`LLMOutputError` and
    retry with backoff instead of skipping the unit of work.
    """
