"""Classify item-path exceptions into :class:`~kompend.core.types.ReasonCode`.

The item path reports every failure as ``logger.error``/``logger.warning``
with ``extra={"reason_code": ...}``. Emission sites that catch a concrete
exception pass the literal member; sites that catch a broad ``Exception``
call :func:`reason_code_for` instead, so the code still reflects what
actually raised.

Two shapes of failure reach those broad handlers:

* typed exceptions raised in-process (``ParseDeadlineExceeded``,
  ``BinarySourceError``, ``ConnectionError``, …);
* strings flattened by the deadline worker, which reports a child-process
  failure as ``RuntimeError(f"{type(exc).__name__}: {exc}")`` — the type
  survives only as a prefix, matched via :data:`_WORKER_WRAPPED_PREFIXES`.
"""

from __future__ import annotations

from typing import Final

from kompend.core.pipeline.bounded_parse import ParseDeadlineExceeded
from kompend.core.types import ReasonCode
from kompend.preprocessing.host_memory import InsufficientHostMemoryError
from kompend.preprocessing.parsers.base import BinarySourceError

#: Worker-flattened exception prefixes worth a code other than the catch-all,
#: keyed by the code they map to. The prefix is ``f"{type(exc).__name__}: "``
#: — stable per exception class.
_WORKER_WRAPPED_PREFIXES: Final[dict[ReasonCode, str]] = {
    ReasonCode.MISLABELLED_BINARY: "BinarySourceError:",
}


def reason_code_for(exc: BaseException) -> ReasonCode:
    """Map *exc* to the reason code an item-path log/ledger row should carry.

    Unknown failures classify as :attr:`ReasonCode.BACKEND_UNAVAILABLE` — the
    pragmatic bucket for "the parse backend path raised something we did not
    model", with the exception text preserved in the log message itself. If a
    histogram of ``backend_unavailable`` ever shows non-backend pollution,
    split a new member instead of widening this mapping.
    """
    if isinstance(exc, ParseDeadlineExceeded):
        return ReasonCode.TIMEOUT
    if isinstance(exc, BinarySourceError):
        return ReasonCode.MISLABELLED_BINARY
    if isinstance(exc, InsufficientHostMemoryError | ConnectionError):
        return ReasonCode.BACKEND_UNAVAILABLE
    if isinstance(exc, RuntimeError):
        for code, prefix in _WORKER_WRAPPED_PREFIXES.items():
            if str(exc).startswith(prefix):
                return code
    return ReasonCode.BACKEND_UNAVAILABLE
