"""Exception formatting helpers for log/error messages.

Centralised so that truncation lengths stay consistent across the codebase
and so that smarter formatting (e.g. extracting the most relevant line
from a multi-line stacktrace) can be added in one place.
"""

from __future__ import annotations

from typing import Final

#: Maximum number of characters used when embedding an exception's string
#: representation into a log message or user-facing error. Exceptions are
#: often multi-line and verbose; truncating them keeps single-line log
#: lines readable without losing the leading cause. The full stack trace
#: is preserved separately by the logging framework via ``exc_info=True``.
MAX_CHARS_EXCEPTION: Final[int] = 200


def shorten_exception(exc: BaseException, *, limit: int = MAX_CHARS_EXCEPTION) -> str:
    """Return a single-line, length-bounded string for ``exc``.

    Args:
        exc: Any exception instance.
        limit: Maximum character count. Defaults to :data:`MAX_CHARS_EXCEPTION`.

    Returns:
        The exception's ``str(exc)``, collapsed to a single line and
        truncated to ``limit`` characters with an ellipsis if cut.
    """
    text = str(exc).replace("\n", " ").replace("\r", " ").strip()
    if len(text) > limit:
        return f"{text[: limit - 1]}…"
    return text
