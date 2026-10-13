"""Process-wide CLI verbosity toggle.

Set once by the CLI callback (``--verbose`` / ``KOMP_VERBOSE``) and read by
recoverable log call sites that attach a traceback only when the user asked for
detail. Lives in ``core`` (not ``cli``) so pipeline modules can read it without
depending on the CLI layer.

The default Python behaviour is unchanged: tracebacks are attached where a log
call passes ``exc_info=is_verbose()``, and fatal crashes are formatted by
:func:`kompend.cli._shared.format_exception_for_cli`.
"""

from __future__ import annotations

_verbose = False


def set_verbose(enabled: bool) -> None:
    """Set the process-wide verbose toggle (called by the CLI callback)."""
    global _verbose  # noqa: PLW0603 — one process-wide switch, set exactly once
    _verbose = bool(enabled)


def is_verbose() -> bool:
    """Whether the user opted into verbose output (full tracebacks)."""
    return _verbose
