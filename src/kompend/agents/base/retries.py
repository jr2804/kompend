"""Transient-retry policy, failure mapping and cache-rejection sniffing.

The transient-retry loop (``invoke_agent`` and its helpers) retries transient
transport errors with a linear backoff and delegates the empty-answer
fast-fail (``EmptyAnswerError`` raised by
``kompend.agents.kompend_model._fail_on_empty_answer``) to the stage
unmolested — an empty answer must fail fast, a transient one retries;
do not simplify either half. ``_map_agent_error`` turns everything else
into the exception type the stage reports, and ``_is_cache_rejection``
recognises provider schema mismatches that no retry can fix.

The agent is typed as :class:`~kompend.agents.base.protocols.RetryHost`
rather than :class:`KompendAgentBase`: ``agent.py`` imports this module,
so naming the class at runtime would close the cycle.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from pydantic_ai import UsageLimits
from pydantic_ai.messages import ModelMessage

from kompend.agents.base.deps import DepsT
from kompend.agents.base.protocols import RetryHost
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

logger = logging.getLogger(__name__)


def failure(agent: RetryHost, exc: Exception, deps: DepsT) -> LLMOutputError:
    """Log a failed run once and return the exception the stage must see.

    Both entry points (``run`` and ``run_with_messages``) fail the same
    way, so the log line and the mapping live here instead of in two
    identical ``except`` blocks that could drift apart.
    """
    debug_hint = f" (debug: {agent._last_debug_file})" if agent._last_debug_file else ""
    ctx_info = _context_usage_str(agent.usage_tracker)
    logger.warning(
        "Agent '%s' failed on '%s': %s%s%s",
        type(agent).__name__,
        deps.source_id,
        exc,
        debug_hint,
        ctx_info,
    )
    return _map_agent_error(exc, model_string=agent.model_string, debug_hint=debug_hint)


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


def invoke_agent(
    agent: RetryHost,
    user_prompt: str,
    deps: DepsT,
    system_prompt: str,
    message_history: list[ModelMessage] | None,
    *,
    conversation_id: str | None = None,
    run_id: str | None = None,
) -> Any:
    """Run the agent: one transient-retry policy, one cache-rejection retry.

    The retry policy is the *only* place a request is repeated. A stage that
    needs a different number of tries changes ``compile.llm.transient_retries``
    instead of growing its own backoff loop.

    A gateway that does not know the provider's cache field answers 400 with
    the field name in the body; caching is an optimisation, so that failure
    disables caching for this agent and retries immediately.
    """
    if agent._agent is None:
        agent._agent = agent._build_agent()

    transient_left = agent._transient_retries()
    attempt = 0
    while True:
        try:
            return agent._agent.run_sync(
                user_prompt,
                deps=deps,
                instructions=system_prompt,
                message_history=message_history,
                usage_limits=agent._usage_limits_for(deps),
                conversation_id=conversation_id,
                run_id=run_id,
            )
        except Exception as exc:
            if agent._prompt_cache and _is_cache_rejection(exc):
                logger.warning(
                    "Prompt caching rejected by '%s' (%s) - retrying without it; set compile.llm.prompt_cache = false to skip the probe",
                    agent.model_string,
                    exc,
                )
                agent._prompt_cache = False
                agent._agent = agent._build_agent()
                continue
            if transient_left <= 0 or not is_transient_error(exc):
                raise
            attempt += 1
            transient_left -= 1
            delay = agent._transient_backoff() * 2 ** (attempt - 1)
            logger.warning(
                "Transient failure from '%s' (attempt %d, %d retry/retries left, waiting %.1fs): %s",
                agent.model_string,
                attempt,
                transient_left,
                delay,
                exc,
            )
            time.sleep(delay)


def _is_cache_rejection(exc: BaseException) -> bool:
    """True for provider rejections that no retry can fix.

    These are schema mismatches — an unknown field, an extra input —
    reported by the provider as a 4xx cache write rejection; retrying
    would only burn budget.
    """
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status in {400, 404, 422} and ("cache" in text or "unknown field" in text or "extra inputs" in text)


def transient_retries(agent: RetryHost) -> int:
    """Configured whole-request retries after a transient failure."""
    return agent.llm.transient_retries if agent.llm is not None else _DEFAULT_TRANSIENT_RETRIES


def transient_backoff(agent: RetryHost) -> float:
    """Configured first backoff (seconds) for a transient retry."""
    return agent.llm.transient_backoff if agent.llm is not None else _DEFAULT_TRANSIENT_BACKOFF_SECONDS


def usage_limits_for(agent: RetryHost, deps: DepsT) -> UsageLimits | None:
    """Per-run UsageLimits; default is the RLM preset limits (or None).

    pydantic-ai's implicit default caps a run at 50 requests, which is
    tighter than multi-turn tool agents need - overriding agents derive
    their honest budget from deps via :func:`session_limits`.
    """
    return agent._rlm_usage_limits


def session_limits(
    agent: RetryHost,
    *,
    units: int,
    per_unit: int,
    headroom_frac: float = 1.1,
    session_overhead: int = 2,
) -> UsageLimits | None:
    """Request budget for a session that works through ``units`` items.

    One formula for every agentic session: the planned turns
    (``units`` x ``per_unit``) plus the fixed session overhead - the
    "no more items" probe and the closing turn - plus the retry
    capacity :func:`retry_requests` derives from the configured
    budgets, scaled by ``headroom_frac``.

    The unit count and its per-unit cost stay in the subclass: only the
    agent knows what one work item costs. Everything that is the same
    for all sessions lives here.
    """
    if agent._rlm_usage_limits is not None:
        return agent._rlm_usage_limits
    planned = max(units, 0) * per_unit + session_overhead + retry_requests(agent)
    return UsageLimits(request_limit=math.ceil(planned * headroom_frac))


def retry_requests(agent: RetryHost) -> int:
    """Requests the configured retry budgets may add to one run.

    A retried tool call and a rejected structured output each cost another
    request, so a session budget that counts only planned turns aborts as
    soon as the model needs its first retry - a one-group distill session
    with four output retries configured died on ``request_limit of 6``
    while still holding real work. The sum of both budgets is the upper
    bound a single session may legitimately need.
    """
    retries = agent.retries
    if isinstance(retries, int):
        # A bare int sets both categories (pydantic-ai ``AgentRetries``
        # shorthand).
        return max(retries, 0) * 2
    return max(int(retries.get("tools", 0)), 0) + max(int(retries.get("output", 0)), 0)
