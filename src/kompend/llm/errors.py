"""Classification of failed LLM calls.

One question decides what a caller may do with a failed model call: was the
failure *transient* — a timeout, a dropped connection, an overloaded or
rate-limited server — or *semantic*, where the model answered but not with
usable output? A transient failure is worth repeating; a semantic one is not,
because the same request will fail the same way.

The classification uses exception types and protocol status codes only. No
message matching: a gateway that words a 400 as "timed out" is still a client
error, and a 504 is still a server-side timeout. When a caller needs a judgement
beyond "was this transport or semantics", it asks the optional decision model
(see :mod:`kompend.llm.decision`).

The one exception is :func:`is_quota_refusal`: a quota death and a
per-minute throttle share the 429 status, so telling "do not send again
this run" from "retry in a minute" needs the body's quota vocabulary.
That matcher lives here, next to the status walk, and nowhere else.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import httpx
import httpx2
from pydantic_ai.exceptions import ModelAPIError

#: HTTP status codes that indicate a retryable server/transport condition.
#: 408/504 are timeouts, 425/429 throttling, 5xx server faults, 529 Anthropic's
#: "overloaded". Every other 4xx is a property of the request itself.
TRANSIENT_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: How far down a ``raise ... from ...`` chain to look for the root cause.
_MAX_CAUSE_DEPTH: Final[int] = 5


#: The status that carries quota refusals. A quota death surfaces as 429
#: (the rate-limit family); nothing else trips a model unconditionally.
_QUOTA_STATUS: Final[int] = 429

#: Lowercase body markers that name a quota/allowance death rather than a
#: per-minute throttle. Deliberately narrower than "rate limit": a throttle
#: clears in seconds and stays retryable, while these name a balance that
#: will not restore inside the run.
QUOTA_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "quota",
        "usage limit",
        "usage credit",
        "out of credit",
        "out of credits",
        "billing",
        "spend limit",
    }
)


def is_transient_error(exc: BaseException) -> bool:
    """True when *exc* — or one of its causes — is a retryable failure.

    Parameters
    ----------
    exc:
        The exception a caller caught, usually the wrapper an agent raised.

    Returns:
    -------
    bool
        ``True`` for transport failures and retryable status codes, ``False``
        for validation, schema and request errors.
    """
    # A breaker refusal is a local verdict, not a provider signal: it must
    # never read as transient even when raised inside a transient handler,
    # whose ``__context__`` would otherwise classify transient and re-drive
    # the dead model. The refusal carries the marker as a class attribute so
    # this module stays type-and-status-only (no import of the breaker).
    if getattr(exc, "_kompend_breaker_open", False):
        return False
    return any(_transient_cause(cause) for cause in _cause_chain(exc))


def _transient_cause(exc: BaseException) -> bool:
    """Classify one link of the cause chain."""
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    if isinstance(exc, httpx.TimeoutException | httpx.NetworkError | httpx.RemoteProtocolError):
        return True
    if isinstance(exc, httpx2.TimeoutException | httpx2.NetworkError | httpx2.RemoteProtocolError):
        return True
    if isinstance(exc, ModelAPIError):
        # A ModelAPIError without a status never reached an HTTP response
        # (connection refused, stream aborted), so it is transport-level.
        status = getattr(exc, "status_code", None)
        return True if status is None else status in TRANSIENT_STATUS_CODES
    # Provider SDK errors that keep their own class (anthropic, openai) expose
    # the HTTP status as an attribute; treat it like a ModelHTTPError.
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in TRANSIENT_STATUS_CODES


def is_quota_refusal(exc: BaseException) -> bool:
    """True when the chain shows a 429 whose body names a quota death.

    Both halves must hold: the status proves the provider refused for
    load/allowance reasons, and the marker proves the refusal is the
    allowance (which will not restore this run) rather than a per-minute
    throttle (which will). A plain 429 — "too many requests", "slow
    down" — stays transient and keeps retrying.
    """
    chain = list(_cause_chain(exc))
    if not any(_link_status(link) == _QUOTA_STATUS for link in chain):
        return False
    text = " ".join(str(link) for link in chain).lower()
    return any(marker in text for marker in QUOTA_BODY_MARKERS)


def quota_cause(exc: BaseException) -> str:
    """Short operator-facing cause for a refusal :func:`is_quota_refusal` found.

    Names the matched quota marker (or the bare status when the body is
    empty) so the trip warning says what died, not just that something did.
    """
    text = " ".join(str(link) for link in _cause_chain(exc)).lower()
    markers = sorted(m for m in QUOTA_BODY_MARKERS if m in text)
    if markers:
        named = ", ".join(repr(m) for m in markers)
        return f"quota refusal (429, body names {named})"
    return "quota refusal (429)"


def _cause_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield *exc* and up to ``_MAX_CAUSE_DEPTH`` causes, without looping."""
    seen: set[int] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and depth <= _MAX_CAUSE_DEPTH and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__
        depth += 1


def _link_status(exc: BaseException) -> int | None:
    """The HTTP status a chain link carries, if any."""
    if isinstance(exc, ModelAPIError):
        status = getattr(exc, "status_code", None)
        return status if isinstance(status, int) else None
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None
