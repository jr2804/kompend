"""One failure vocabulary for every LLM-backed unit of work.

This module is the one place that answers two questions:

- **What kind of failure is this?** :func:`classify_agent_failure` maps the
  exception pydantic-ai raised onto :class:`AgentFailureKind`.
- **Is repeating it worth anything?** :func:`is_retry_pointless` says no for
  the kinds where the model already proved it cannot answer the question
  that was asked (it returned nothing, or it returned something the schema
  rejects). Re-asking it four times only burns the request budget and
  hides the unit that has no result.

Stages keep their own best-effort policy (``.agents/CODE.md`` rule 3): they
call :func:`record_unit_failure` per failed unit, which logs the classified
failure and returns the exception - so the stage appends its SKIP result in
one line - and :meth:`UnitFailureLog.check` raises once the batch is more
failure than result, using the thresholds
``compile.llm.batch_failure_threshold_*`` already define for extraction.

The exception types live here rather than in :mod:`kompend.core.utils.exceptions`
because the classifier needs pydantic-ai's exception classes, and the core
utility module must stay importable without the LLM stack.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from pydantic_ai.exceptions import ModelRetry, ToolRetryError, UnexpectedModelBehavior

from kompend.core.utils.exceptions import LLMOutputError, LLMTransientError
from kompend.llm.errors import is_transient_error
from kompend.llm.overflow import is_context_overflow
from kompend.logging import get_logger

#: A ``ValidationError`` link carries the failure detail under ``.errors()``.
_ERRORS_ATTRIBUTE: Final[str] = "errors"

#: ``pydantic`` marks "the payload was not JSON at all" with this error type.
_JSON_INVALID: Final[str] = "json_invalid"

#: ``json`` reports an empty document — or an empty string — with this text.
#: A *malformed* document reports a different message ("expected ident",
#: "Expecting property name..."), which is a real schema problem and must not
#: be confused with an empty answer.
_EMPTY_JSON_MARKER: Final[str] = "EOF while parsing"

#: pydantic's error entry keys.
_ERROR_INPUT: Final[str] = "input"
_ERROR_MESSAGE: Final[str] = "msg"
_ERROR_TYPE: Final[str] = "type"

#: pydantic-ai's own words for "you returned neither text nor a tool call".
_EMPTY_ANSWER_MARKER: Final[str] = "Please return text or include your response in a tool call"

logger = get_logger(__name__)


class AgentFailureKind(StrEnum):
    """Why an LLM-backed unit of work produced no result.

    The members are the *remedies*, not the call sites: two stages that fail
    the same way get the same policy without either of them deciding it.
    """

    #: The model returned nothing (empty text, or an empty structured payload
    #: that a validator rejected). Repeating the identical request cannot fix
    #: this: the retry feedback says "return text", which the model already
    #: ignored.
    EMPTY_ANSWER = "empty_answer"
    #: The model answered, but not in the shape the schema demands (wrong
    #: types, missing fields, invalid JSON). The retry feedback *is* useful
    #: here, so the agent's own output budget handles it.
    SCHEMA_MISMATCH = "schema_mismatch"
    #: The request did not fit the window. The caller re-plans with a smaller
    #: effective window; repeating it unchanged cannot succeed.
    CONTEXT_OVERFLOW = "context_overflow"
    #: Transport failure — timeout, dropped connection, 429/5xx. The only kind
    #: worth repeating verbatim.
    TRANSIENT = "transient"
    #: Anything else. Treated like a semantic failure: recorded, not retried.
    OTHER = "other"


@dataclass(frozen=True)
class UnitFailure:
    """One failed unit of work, as recorded by a stage.

    Attributes:
        unit: Stable identity of the unit (page id, source id, query).
        kind: The classified failure.
        error: The exception, kept for the log line and re-raise.
        attempts: How many requests the agent made before giving up.
    """

    unit: str
    kind: AgentFailureKind
    error: BaseException
    attempts: int = 1

    def describe(self) -> str:
        """One-line human description for a log message or SKIP result."""
        return f"{self.unit}: {self.kind.value} after {self.attempts} request(s) - {self.error}"


class EmptyAnswerError(LLMOutputError):
    """Raised when the model answered with nothing at all.

    Subclasses :class:`~kompend.core.utils.exceptions.LLMOutputError` so every
    existing ``except LLMOutputError`` site keeps working, but it is raised
    from the model layer *before* the output-retry budget is spent: the model
    already proved it had nothing to say, and re-asking it the identical
    question cannot change that. The stage decides what a missing unit means;
    what it must not do is burn four requests and then report "exceeded
    maximum output retries", which names neither the unit nor the cause.
    """


def is_retry_pointless(kind: AgentFailureKind) -> bool:
    """True when re-sending the identical request cannot change the outcome.

    Empty answers and a request that did not fit the window are the two the
    model has already answered: it produced nothing, or the provider refused
    the size. Schema mismatches, transport failures and unknown errors keep
    their retry budget, because the retry feedback or a second attempt can
    genuinely change the answer.
    """
    return kind in _POINTLESS_TO_RETRY


def record_unit_failure(
    *,
    stage: str,
    unit: str,
    exc: BaseException,
    attempts: int = 1,
) -> UnitFailure:
    """Classify a failed unit, log it at ``WARNING``, and return the record.

    The stage then adds its own SKIP marker to the result list it returns -
    this helper never decides what a "result" looks like. Logging happens
    here so every stage names the same facts (which unit, what kind, how many
    requests) instead of each one inventing a message.
    """
    kind = classify_agent_failure(exc)
    failure = UnitFailure(unit=unit, kind=kind, error=exc, attempts=attempts)
    logger.warning("%s: %s", stage, failure.describe())
    return failure


def classify_agent_failure(exc: BaseException) -> AgentFailureKind:
    """Classify *exc* into the remedy it needs.

    The classifier walks the ``raise ... from ...`` chain, because pydantic-ai
    wraps the interesting part: a structured-output mismatch surfaces as
    ``UnexpectedModelBehavior('Exceeded maximum output retries (N)')`` whose
    cause is a validator error, and that cause is the only thing that says
    whether the model returned nothing or returned the wrong shape.

    Empty answers are recognised in two ways, because pydantic-ai reports
    them differently depending on whether structured output went through a
    tool call or through inline JSON:

    - no validator error at all, only a ``ToolRetryError`` — the tool-less
      agent answered with neither text nor a call;
    - a validator error whose only entry is ``json_invalid`` with an *empty*
      input (``json.loads('')`` says "EOF while parsing a value").

    :class:`EmptyAnswerError` is matched on the whole chain, not just the
    head: :class:`~kompend.agents.base.KompendAgentBase` re-wraps every
    failure as ``LLMOutputError`` with the model name in the message, so by
    the time a stage catches it the model-layer class is the ``__cause__``.

    A malformed-but-present document also reports ``json_invalid``, so the
    marker check matters: dropping it would classify a genuine schema
    mismatch as an empty answer and skip a retry that would have helped.
    """
    if isinstance(exc, EmptyAnswerError):
        return AgentFailureKind.EMPTY_ANSWER
    if isinstance(exc, LLMTransientError) or is_transient_error(exc):
        return AgentFailureKind.TRANSIENT
    chain = list(_failure_cause_chain(exc))
    if any(isinstance(link, EmptyAnswerError) for link in chain):
        return AgentFailureKind.EMPTY_ANSWER
    if any(_names_context_overflow(link) for link in chain):
        return AgentFailureKind.CONTEXT_OVERFLOW
    errors = _validation_errors(chain)
    if errors:
        return AgentFailureKind.EMPTY_ANSWER if _is_empty_payload(errors) else AgentFailureKind.SCHEMA_MISMATCH
    if any(_names_empty_answer(link) for link in chain):
        return AgentFailureKind.EMPTY_ANSWER
    return AgentFailureKind.OTHER


def _failure_cause_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield *exc* and its causes, without looping.

    ``BaseExceptionGroup`` (what an ``asyncio.TaskGroup`` raises) keeps its
    members in ``.exceptions`` rather than in ``__cause__``, so both are
    walked - a raised error inside a task group would otherwise be invisible.
    """
    seen: set[int] = set()
    depth = 0
    current: BaseException | None = exc
    while current is not None and depth <= _CHAIN_DEPTH and id(current) not in seen:
        seen.add(id(current))
        yield current
        nested = getattr(current, "exceptions", ())
        depth += 1
        if isinstance(nested, tuple | list):
            for member in nested:
                if isinstance(member, BaseException) and id(member) not in seen:
                    yield from _failure_cause_chain(member)
        current = current.__cause__ or current.__context__


def _names_context_overflow(exc: BaseException) -> bool:
    """True when *exc* is (or wraps) a provider context-window rejection."""
    return is_context_overflow(exc)


def _names_empty_answer(exc: BaseException) -> bool:
    """True when *exc* is pydantic-ai reporting "you returned nothing"."""
    if isinstance(exc, ModelRetry):
        return _EMPTY_ANSWER_MARKER in str(exc)
    if isinstance(exc, ToolRetryError):
        return _EMPTY_ANSWER_MARKER in str(exc)
    return isinstance(exc, UnexpectedModelBehavior) and _EMPTY_ANSWER_MARKER in str(exc)


def _validation_errors(chain: Sequence[BaseException]) -> list[dict[str, object]]:
    """Collect the flattened ``ValidationError`` entries along *chain*."""
    entries: list[dict[str, object]] = []
    for link in chain:
        collect = getattr(link, _ERRORS_ATTRIBUTE, None)
        if callable(collect):
            try:
                entries.extend(collect())
            except (TypeError, ValueError):  # pragma: no cover - a non-pydantic .errors()
                continue
    return entries


def _is_empty_payload(errors: Sequence[dict[str, object]]) -> bool:
    """True when the collected validation entries describe an empty answer.

    Two shapes qualify:

    - the payload was not JSON and what was there is whitespace only;
    - the payload was not JSON and the parser stopped at end-of-file, which
      is what parsing an empty string reports.

    Anything else — including a malformed document — is a schema mismatch.
    """
    if not errors:
        return False
    if {entry.get(_ERROR_TYPE) for entry in errors} != {_JSON_INVALID}:
        return False
    return any(_is_blank_entry(entry) for entry in errors)


def _is_blank_entry(entry: dict[str, object]) -> bool:
    """True when one ``json_invalid`` entry points at nothing at all."""
    if _EMPTY_JSON_MARKER in str(entry.get(_ERROR_MESSAGE, "")):
        return True
    raw = entry.get(_ERROR_INPUT)
    return isinstance(raw, str) and not raw.strip()


#: The kinds whose retry budget is pointless — see :func:`is_retry_pointless`.
_POINTLESS_TO_RETRY: Final[frozenset[AgentFailureKind]] = frozenset({AgentFailureKind.EMPTY_ANSWER, AgentFailureKind.CONTEXT_OVERFLOW})

#: How far down a wrapped chain to look before giving up.
_CHAIN_DEPTH: Final[int] = 8


@dataclass
class UnitFailureLog:
    """Accumulates per-unit failures for one batch and decides when to stop.

    A stage that iterates units (pages, sources, claims) records each failure
    here instead of keeping its own counter. Once the batch is more failure
    than result - the same ratio/minimum pair extraction already obeys
    (``compile.llm.batch_failure_threshold_ratio`` and
    ``..._threshold_min``) - :meth:`check` raises, so a broken model stops the
    run instead of producing a report that looks merely "small".

    Below the threshold the batch continues: the SKIP markers the stage
    recorded stay in the result. That is the point — counting only completed
    units lets a swallowed failure raise the health score by shrinking the
    denominator.
    """

    stage: str
    failures: list[UnitFailure] = field(default_factory=list)
    #: How many units the stage tried to process (see :meth:`attempted`).
    total: int = 0
    #: See :data:`kompend.compilation.two_phase.state._FAILURE_THRESHOLD_RATIO`.
    ratio: float = 0.5
    #: See :data:`kompend.compilation.two_phase.state._FAILURE_THRESHOLD_MIN`.
    min_units: int = 5

    def attempted(self, count: int = 1) -> None:
        """Count *count* units the stage tried to process.

        Called once per unit *before* the call, so a failure recorded below
        lands in the same denominator as the successes around it. Without
        that, a stage that only counted failures could never tell "every third
        page failed" from "three pages failed out of ninety".
        """
        self.total += count

    def record(self, unit: str, exc: BaseException, *, attempts: int = 1) -> UnitFailure:
        """Record one failed unit and return its classified record.

        Does not change :attr:`total` — the stage already called
        :meth:`attempted` for this unit, and counting it twice would halve
        the failure ratio.
        """
        failure = record_unit_failure(stage=self.stage, unit=unit, exc=exc, attempts=attempts)
        self.failures.append(failure)
        return failure

    def count(self, kind: AgentFailureKind) -> int:
        """How many recorded failures have *kind*."""
        return sum(1 for failure in self.failures if failure.kind == kind)

    def summary(self) -> str:
        """One line naming every kind that occurred, most frequent first."""
        if not self.failures:
            return f"{self.stage}: no failures"
        counts = {kind: self.count(kind) for kind in AgentFailureKind if self.count(kind)}
        rendered = ", ".join(f"{kind.value}={count}" for kind, count in sorted(counts.items(), key=lambda kv: -kv[1]))
        return f"{self.stage}: {len(self.failures)} failed unit(s) of {self.total} - {rendered}"

    def check(self) -> None:
        """Raise :class:`LLMOutputError` once the batch is mostly failures.

        The message carries the kinds, so the operator sees "the model returned
        nothing" instead of "exceeded maximum output retries".
        """
        if not self.failures:
            return
        if self.total <= 0 or self.total < self.min_units or len(self.failures) / self.total <= self.ratio:
            return
        raise LLMOutputError(
            f"{self.summary()}. More than {self.ratio:.0%} of the batch failed; stopping instead of returning a partial result as if it were complete."
        )
