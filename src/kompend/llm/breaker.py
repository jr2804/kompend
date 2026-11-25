"""Per-model circuit breaker: a quota-dead model is not asked again this run.

A provider allowance that dies mid-run (weekly usage limit, out of credits)
stays dead for the rest of the run, but the pipeline cannot tell that from a
per-minute throttle by status alone — both surface as 429. Without a breaker
every remaining unit burns its full retry budget against the dead model (the
pyvas run behind knox-dnw3.6 spent five hours doing exactly that), and every
failure looks like a fresh transient in the logs.

The breaker sits in the request path (:func:`kompend.agents.base.retries.invoke_agent`,
the only place a request is repeated) and trips per model, per run:

- **quota refusal** (:func:`kompend.llm.errors.is_quota_refusal`) trips
  unconditionally — the allowance will not restore inside the run;
- **a run of identical failures** trips once it outlasts one full retry
  cycle (``compile.llm.transient_retries``): the first ``budget`` failures
  still ride the loop's own retries exactly as before, and any success or
  any *different* failure resets the run. A single blip can never condemn a
  model — a *run* of failures means at least two in a row.

A tripped model refuses locally: the request is never built, nothing is
sent, nothing sleeps. The refusal raises :class:`BreakerOpenError`, which
is already an :class:`~kompend.core.utils.exceptions.LLMOutputError` and
never classifies transient (see :func:`kompend.llm.errors.is_transient_error`).

Run scoping: one :class:`ModelBreaker` per :class:`UsageTracker` (the
per-run object every agent already shares), held in a registry keyed by
``id(tracker)`` with a weakref identity check — ids are recycled, so a hit
must still belong to the live tracker or it is dropped. A call without a
tracker has no run to scope to and skips the breaker.
"""

from __future__ import annotations

import weakref
from typing import Final

from kompend.core.utils.exceptions import LLMOutputError
from kompend.llm.usage import UsageTracker
from kompend.logging import get_logger

#: Floor for the trip count: "a run of identical failures" means at least
#: two in a row, so a lone blip can never condemn a model even when the
#: configured retry budget is 0.
_MIN_TRIP_RUN: Final[int] = 1

logger = get_logger(__name__)


_breakers: dict[int, tuple[weakref.ReferenceType[UsageTracker], ModelBreaker]] = {}


class BreakerOpenError(LLMOutputError):
    """A request refused locally because its model tripped the run's breaker.

    The message names the model and the trip cause. It carries no provider
    status and no transient markers, and :func:`kompend.llm.errors.is_transient_error`
    answers ``False`` for it unconditionally (via the marker below), so no
    caller retries a refusal — retrying would just refuse again.
    """

    _kompend_breaker_open = True


class ModelBreaker:
    """Per-run trip state for every model the run touches.

    One instance per run (see :func:`breaker_for_tracker`). The tracker and
    the agents are single-threaded within a run, so no locking: the breaker
    only ever sees one call at a time.
    """

    def __init__(self) -> None:
        self._tripped: dict[str, str] = {}
        self._runs: dict[str, tuple[str, int]] = {}

    def is_open(self, model: str) -> bool:
        """Whether *model* tripped and must be refused without sending."""
        return model in self._tripped

    def check(self, model: str) -> None:
        """Raise :class:`BreakerOpenError` when *model* tripped this run."""
        cause = self._tripped.get(model)
        if cause is not None:
            raise BreakerOpenError(f"Model '{model}' is circuit-broken for this run ({cause}): refusing to send")

    def refusal(self, model: str) -> BreakerOpenError:
        """The refusal for a *model* :meth:`is_open` already reported."""
        cause = self._tripped.get(model, "tripped earlier this run")
        return BreakerOpenError(f"Model '{model}' is circuit-broken for this run ({cause}): refusing to send")

    def record_success(self, model: str) -> None:
        """A success resets the model's consecutive-failure run."""
        self._runs.pop(model, None)

    def trip(self, model: str, cause: str, *, stage: str) -> None:
        """Trip *model* unconditionally (quota death). Logs the one WARNING.

        A second trip for the same model is a no-op: the refusal path, not
        another warning, is what later calls see.
        """
        if model in self._tripped:
            return
        self._tripped[model] = cause
        self._runs.pop(model, None)
        logger.warning(
            "Model '%s' is quota-dead for the rest of this run (%s); further requests are refused locally (tripped by stage '%s').",
            model,
            cause,
            stage,
        )

    def record_failure(self, model: str, exc: BaseException, budget: int) -> bool:
        """Record one failed attempt; return True when it trips the model.

        Identical failures (same exception type and status) extend the run;
        any different failure restarts it at one, and :meth:`record_success`
        clears it. The run trips the model once it is longer than one full
        retry cycle (``budget``) — see :data:`_MIN_TRIP_RUN` for the floor.
        Quota refusals never reach here (they trip in :meth:`trip`); a
        failure recorded after the trip is ignored.
        """
        if model in self._tripped:
            return True
        signature = _failure_signature(exc)
        previous, length = self._runs.get(model, (None, 0))
        length = length + 1 if signature == previous else 1
        self._runs[model] = (signature, length)
        if length > max(budget, _MIN_TRIP_RUN):
            cause = f"{length} identical failures in a row ({signature})"
            self._tripped[model] = cause
            self._runs.pop(model, None)
            logger.warning(
                "Model '%s' failed %s; tripping it for the rest of this run.",
                model,
                cause,
            )
            return True
        return False


def _failure_signature(exc: BaseException) -> str:
    """What makes two failures 'identical' for the consecutive-failure run."""
    status: int | None = None
    current: BaseException | None = exc
    depth = 0
    while current is not None and depth < 6:
        found = getattr(current, "status_code", None)
        if isinstance(found, int):
            status = found
            break
        current = current.__cause__ or current.__context__
        depth += 1
    return f"{type(exc).__name__}:{status}"


def breaker_for_tracker(tracker: UsageTracker | None) -> ModelBreaker | None:
    """The run's breaker for *tracker*, or ``None`` when there is no run.

    Registry entries are keyed by ``id(tracker)`` with a weakref identity
    check: ids are recycled once a tracker dies, so a hit whose weakref is
    dead or points at a different object is dropped and replaced.
    """
    if tracker is None:
        return None
    entry = _breakers.get(id(tracker))
    if entry is not None:
        ref, breaker = entry
        if ref() is tracker:
            return breaker
    breaker = ModelBreaker()
    try:
        ref = weakref.ref(tracker)
    except TypeError:  # pragma: no cover - UsageTracker is weakrefable
        return breaker
    _breakers[id(tracker)] = (ref, breaker)
    _purge_dead_breakers()
    return breaker


def _purge_dead_breakers() -> None:
    """Drop registry entries whose tracker died (bounded id-keyed cache)."""
    dead = [key for key, (ref, _) in _breakers.items() if ref() is None]
    for key in dead:
        del _breakers[key]
