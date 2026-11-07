"""How a command reports a usage mistake and an unexpected failure.

:func:`usage_error` ends the run with the conventional exit code; the formatter
keeps a traceback out of the user's face unless ``--verbose`` asked for it.
"""

from __future__ import annotations

import sys
import traceback
from typing import NoReturn

from kompend.logging import get_logger

logger = get_logger(__name__)


def usage_error(message: str) -> NoReturn:
    """Report a usage error and exit 2, matching the parser's own exit code."""
    sys.stderr.write(f"{message}\n")
    raise SystemExit(2)


def format_exception_for_cli(exc: BaseException, *, verbose: bool) -> str:
    """Format a fatal exception for the CLI.

    Default: a single ``error: <Type>: <message>`` line, no traceback. With
    *verbose*, the full traceback. Deliberately does NOT use
    ``traceback.format_exc()`` to build the short form — a compressed trace is
    uglier than the trace it replaces.
    """
    if verbose:
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
    return f"error: {type(exc).__name__}: {exc}"
