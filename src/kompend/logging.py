"""Package-wide Rich console and logging factory.

Mirrors the pattern from tdoc_crawler.logging — application code should
use ``get_logger(__name__)`` instead of ``logging.getLogger(__name__)`` so
that log output is automatically routed through the shared Rich stderr
console.

Console layout is always terminal-width-aware (no fixed width).

Usage::

    from kompend.logging import get_console, get_logger, set_verbosity

    logger = get_logger(__name__)
    logger.info("Starting operation")

    # For direct rich output (stdout):
    console = get_console()
    console.print("[bold]Done![/bold]")
"""

from __future__ import annotations

import functools
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

DEFAULT_LEVEL: int = logging.WARNING

#: How many items a diagnostic log line samples out of a larger collection
#: ("first 5 collisions", "e.g. these aliases"). A log line summarises; it never
#: lists the whole set, and the count is a presentation choice rather than a
#: data limit — the full set is available at the call site.
LOG_SAMPLE_LIMIT: int = 5


@functools.cache
def get_console() -> Console:
    """Cached stdout console.

    Auto-detects terminal width on every render.  Colour is suppressed
    when stdout is not a TTY (CI, pipes).
    """
    return Console(stderr=False, no_color=not sys.stdout.isatty())


def get_logger(name: str) -> logging.Logger:
    """Return a logger routed to the shared Rich console.

    Drop-in replacement for ``logging.getLogger(name)``.  The first call
    installs RichHandler on the root logger; all subsequent calls are free.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
    """
    configure_logging()
    return logging.getLogger(name)


@functools.cache
def configure_logging() -> None:
    """Wire the root logger to RichHandler (idempotent via @functools.cache).

    Called automatically by ``get_logger()``.  Safe to call explicitly at
    CLI entry points to ensure logging is configured before any command runs.
    """
    handler = RichHandler(
        console=get_err_console(),
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
        tracebacks_width=None,
    )
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(DEFAULT_LEVEL)


@functools.cache
def get_err_console() -> Console:
    """Cached stderr console used by log output and progress bars.

    Kept separate from the stdout console so progress and log lines do
    not interleave with program output.
    """
    return Console(stderr=True, no_color=not sys.stderr.isatty())


def set_verbosity(level: int | str) -> None:
    """Set the logging level for all loggers (root and children).

    Args:
        level: Integer level (``logging.DEBUG``, ``logging.INFO``, …) or
               string name (``"debug"``, ``"info"``, ``"warning"``, …).

    Raises:
        ValueError: If *level* is an unrecognised string name.
    """
    if isinstance(level, str):
        level_int = logging._nameToLevel.get(level.upper())
        if level_int is None:
            msg = f"Invalid logging level: {level!r}"
            raise ValueError(msg)
        level = level_int
    logging.root.setLevel(level)
