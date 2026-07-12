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
    return shorten_string(str(exc), limit=limit)


def shorten_string(text: str, *, limit: int = MAX_CHARS_EXCEPTION) -> str:
    """Return a single-line, length-bounded string for *text*.

    Use this for arbitrary string truncation in log messages, error
    payloads, or display previews where the input is already a string
    (not an exception). Collapses newlines to spaces and truncates
    with an ellipsis when the limit is exceeded.

    Args:
        text: The string to format.
        limit: Maximum character count. Defaults to
            :data:`MAX_CHARS_EXCEPTION`.

    Returns:
        ``text`` collapsed to a single line and truncated to ``limit``
        characters with an ellipsis if cut.
    """
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if len(text) > limit:
        return f"{text[: limit - 1]}…"
    return text


class LLMOutputError(Exception):
    """Raised when an LLM cannot produce valid structured output.

    The model is considered not suitable for compilation — no fallback,
    no retry.  The caller should abort compilation with a clear message.
    """
