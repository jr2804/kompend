"""Transient-retry policy, failure mapping and cache-rejection sniffing.

The transient-retry loop (``_invoke`` and its helpers, relocated here
from the agent class in a later step of the split) retries transient
transport errors with a linear backoff and delegates the empty-answer
fast-fail (``EmptyAnswerError`` raised by
``kompend.agents.kompend_model._fail_on_empty_answer``) to the stage
unmolested — an empty answer must fail fast, a transient one retries;
do not simplify either half. ``_map_agent_error`` turns everything else
into the exception type the stage reports, and ``_is_cache_rejection``
recognises provider schema mismatches that no retry can fix.
"""

from __future__ import annotations

from kompend.agents.failure import EmptyAnswerError
from kompend.core.utils.exceptions import (
    LLMContextOverflowError,
    LLMOutputError,
    LLMTransientError,
)
from kompend.llm.errors import is_transient_error
from kompend.llm.overflow import is_context_overflow, stated_context_size
from kompend.llm.usage import UsageTracker

#: Transient retries when no config is handed over (``compile.llm`` unset).
_DEFAULT_TRANSIENT_RETRIES: int = 2
_DEFAULT_TRANSIENT_BACKOFF_SECONDS: float = 5.0


def _is_cache_rejection(exc: BaseException) -> bool:
    """True for provider rejections that no retry can fix.

    These are schema mismatches — an unknown field, an extra input —
    reported by the provider as a 4xx cache write rejection; retrying
    would only burn budget.
    """
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status in {400, 404, 422} and ("cache" in text or "unknown field" in text or "extra inputs" in text)


def _context_usage_str(tracker: UsageTracker | None) -> str:
    """Format the last-known context-window utilization for failure logs.

    Returns an empty string when no context snapshot was recorded (capability
    disabled or no request reached ``before_model_request``).
    """
    if tracker is None or tracker.context_budget <= 0:
        return ""
    pct = 100 * tracker.context_tokens / tracker.context_budget
    return f" (context: {tracker.context_tokens}/{tracker.context_budget} tokens, {pct:.0f}%)"


def _map_agent_error(exc: Exception, *, model_string: str, debug_hint: str) -> LLMOutputError:
    """Map a provider failure to the exception the pipeline understands.

    Four outcomes:

    - an empty answer (:class:`EmptyAnswerError`, raised by the model layer
      before the output-retry budget is spent) stays an empty answer: the
      stage and the classifier both need to see *why* the unit has no
      result, not a generic "agent failed";
    - a context-window overflow becomes :class:`LLMContextOverflowError`
      carrying the limit the provider named, so the stage can lower the
      effective window and re-plan instead of failing the run;
    - a transient failure becomes :class:`LLMTransientError` (retryable);
    - everything else becomes :class:`LLMOutputError`.
    """
    message = f"Agent failed for model '{model_string}': {exc}{debug_hint}"
    if isinstance(exc, EmptyAnswerError):
        return EmptyAnswerError(message)
    if is_context_overflow(exc):
        return LLMContextOverflowError(message, stated_limit=stated_context_size(exc))
    if is_transient_error(exc):
        return LLMTransientError(message)
    return LLMOutputError(message)
