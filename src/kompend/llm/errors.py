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
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import httpx
from pydantic_ai.exceptions import ModelAPIError

#: HTTP status codes that indicate a retryable server/transport condition.
#: 408/504 are timeouts, 425/429 throttling, 5xx server faults, 529 Anthropic's
#: "overloaded". Every other 4xx is a property of the request itself.
TRANSIENT_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: How far down a ``raise ... from ...`` chain to look for the root cause.
_MAX_CAUSE_DEPTH: Final[int] = 5


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
    return any(_transient_cause(cause) for cause in _cause_chain(exc))


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


def _transient_cause(exc: BaseException) -> bool:
    """Classify one link of the cause chain."""
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    if isinstance(exc, httpx.TimeoutException | httpx.NetworkError | httpx.RemoteProtocolError):
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
