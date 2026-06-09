"""Failover chain for LLM calls.

A :class:`FailoverChain` is an ordered list of models tried in
sequence. For each model it applies retry-with-backoff on transient
errors; once retries are exhausted it advances to the next model.
``PermanentError`` and ``ConfigurationError`` short-circuit the chain.
``AmbiguousError`` is resolved by inspecting the wrapping context.

The chain is callable as ``chain.run(call)`` where ``call`` accepts a
single argument (the model name) and returns the litellm response.
The chain returns the first successful result alongside diagnostics
(``model_used``, ``attempts``, ``total_latency_s``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from knox.llm.classifier import classify_litellm_error
from knox.llm.errors import (
    AmbiguousError,
    ConfigurationError,
    LLMError,
    PermanentError,
    TransientError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class CallDiagnostics:
    """Per-call observability returned by :class:`FailoverChain.run`.

    ``model_used`` is the model that produced the successful result.
    ``attempts`` is the total number of provider invocations across
    all models in the chain (transient retries inflate this).
    ``total_latency_s`` is wall-clock time including backoff sleeps.
    """

    model_used: str
    attempts: int
    total_latency_s: float


@dataclass(frozen=True)
class CallResult[T]:
    """Result wrapper combining the call's return value and diagnostics."""

    value: T
    diagnostics: CallDiagnostics


def _resolve_ambiguous(err: AmbiguousError) -> type[LLMError]:
    """Inspect ``AmbiguousError`` metadata to decide final category.

    Checks the original exception for an HTTP ``status_code`` attribute
    and maps it via :func:`_STATUS_ERROR_MAP` semantics. Falls back to
    :class:`TransientError` (safe default — lets the retry logic
    attempt recovery).
    """
    cause = err.cause
    if cause is not None:
        status: Any = getattr(cause, "status_code", None)
        if isinstance(status, int):
            if 500 <= status < 600:
                return TransientError
            _status_map: dict[int, type[LLMError]] = {
                429: TransientError,
                400: PermanentError,
                422: PermanentError,
                401: ConfigurationError,
                403: ConfigurationError,
                404: ConfigurationError,
            }
            mapped = _status_map.get(status)
            if mapped is not None:
                return mapped
    return TransientError


class FailoverChain:
    """Try a list of LLM models in priority order with retry + cascade.

    Parameters
    ----------
    models:
        Ordered list of model names. The first model is the primary;
        each subsequent model is tried only if the previous one
        exhausted its retries with transient errors.
    max_retries_per_model:
        Maximum transient-error retries before advancing to the next
        model. ``2`` means up to three attempts per model (1 + 2).
    initial_backoff_s:
        First backoff delay. Doubles with each retry (1s → 2s → 4s).
    max_backoff_s:
        Cap on the backoff delay so we never sleep absurdly long.
    sleep:
        Injectable sleep function for tests.
    """

    def __init__(
        self,
        models: list[str],
        *,
        max_retries_per_model: int = 2,
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not models:
            msg = "FailoverChain requires at least one model"
            raise ConfigurationError(msg)
        self.models = list(models)
        self.max_retries_per_model = max_retries_per_model
        self.initial_backoff_s = initial_backoff_s
        self.max_backoff_s = max_backoff_s
        self._sleep = sleep

    def run(self, call: Callable[[str], T]) -> CallResult[T]:
        """Execute ``call(model)`` against each model until one succeeds.

        Raises the most relevant typed error if every model fails:
        :class:`ConfigurationError` if the chain itself is unusable,
        otherwise the last :class:`PermanentError` or, if all errors
        were transient, a :class:`TransientError` aggregating the chain.

        :class:`AmbiguousError` is resolved at runtime by inspecting
        the original exception metadata (e.g. HTTP status code) before
        deciding retry vs. advance vs. abort.
        """
        attempts = 0
        start = time.monotonic()
        last_error: LLMError | None = None

        for model in self.models:
            for retry in range(self.max_retries_per_model + 1):
                attempts += 1
                try:
                    value = call(model)
                except Exception as exc:  # noqa: BLE001 — classified below
                    err = classify_litellm_error(exc, model=model)

                    # Configuration errors are fatal: do not retry, do not advance.
                    if isinstance(err, ConfigurationError):
                        err.attempts = attempts
                        raise err from exc

                    # Ambiguous: inspect metadata to decide category.
                    if isinstance(err, AmbiguousError):
                        resolved = _resolve_ambiguous(err)
                        if resolved is ConfigurationError:
                            err.__class__ = ConfigurationError
                            err.attempts = attempts
                            raise err from exc
                        if resolved is PermanentError:
                            logger.warning(
                                "LLM ambiguous-resolved-permanent on %s (attempt %d): %s",
                                model,
                                attempts,
                                err,
                            )
                            last_error = err
                            break  # next model
                        # resolved == TransientError: fall through to retry logic.
                        logger.info(
                            "LLM ambiguous-resolved-transient on %s (attempt %d): %s",
                            model,
                            attempts,
                            err,
                        )
                        last_error = err
                        if retry < self.max_retries_per_model:
                            delay = min(
                                self.initial_backoff_s * (2**retry),
                                self.max_backoff_s,
                            )
                            logger.info(
                                "LLM retry %d/%d on %s after %.1fs: %s",
                                retry + 1,
                                self.max_retries_per_model,
                                model,
                                delay,
                                err,
                            )
                            self._sleep(delay)
                            continue
                        break  # next model

                    # Permanent errors: advance to next model immediately.
                    if isinstance(err, PermanentError):
                        logger.warning(
                            "LLM permanent error on %s (attempt %d): %s",
                            model,
                            attempts,
                            err,
                        )
                        last_error = err
                        break  # next model

                    # Transient: retry with backoff, then advance.
                    last_error = err
                    if retry < self.max_retries_per_model:
                        delay = min(
                            self.initial_backoff_s * (2**retry),
                            self.max_backoff_s,
                        )
                        logger.info(
                            "LLM transient error on %s (attempt %d), retrying in %.1fs: %s",
                            model,
                            attempts,
                            delay,
                            err,
                        )
                        self._sleep(delay)
                        continue
                    logger.warning(
                        "LLM transient retries exhausted on %s after %d attempts: %s",
                        model,
                        attempts,
                        err,
                    )
                    break  # next model
                else:
                    diagnostics = CallDiagnostics(
                        model_used=model,
                        attempts=attempts,
                        total_latency_s=time.monotonic() - start,
                    )
                    return CallResult(value=value, diagnostics=diagnostics)

        # All models exhausted.
        if last_error is None:
            msg = "FailoverChain produced no result and no error"
            raise TransientError(msg, attempts=attempts)
        last_error.attempts = attempts
        raise last_error


def call_with_failover[T](
    chain: FailoverChain,
    call: Callable[[str], T],
) -> tuple[T, CallDiagnostics]:
    """Convenience helper that unwraps :class:`CallResult` for callers
    that prefer a tuple return signature.
    """
    result = chain.run(call)
    return result.value, result.diagnostics
