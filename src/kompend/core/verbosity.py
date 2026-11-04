"""Process-wide CLI verbosity switch.

Set once by the CLI callback (``--verbose`` / ``KOMP_VERBOSE``) and read by
recoverable log call sites that attach a traceback only when the user asked for
detail. Lives in ``core`` (not ``cli``) so pipeline modules can read it without
depending on the CLI layer.

The value is a :class:`~contextvars.ContextVar`, not a module attribute: the
switch is genuinely process-wide (thirteen modules read it while logging, and a
thread must see what the main thread decided), but context-scoped storage keeps
tests and nested in-process invocations from leaking into one another, and a
child thread or task inherits the value it was started with.

The default Python behaviour is unchanged: tracebacks are attached where a log
call passes ``exc_info=is_verbose()``, and fatal crashes are formatted by
:func:`kompend.cli._shared.format_exception_for_cli`.
"""

from __future__ import annotations

from contextvars import ContextVar

#: Verbose mode for the current context. The CLI sets it once per invocation;
#: ``ContextVar`` semantics make that value visible to every reader below it.
_VERBOSE: ContextVar[bool] = ContextVar("kompend_verbose", default=False)


def set_verbose(enabled: bool) -> None:
    """Set verbose mode for the current context (called by the CLI callback)."""
    _VERBOSE.set(bool(enabled))


def is_verbose() -> bool:
    """Whether the user opted into verbose output (full tracebacks)."""
    return _VERBOSE.get()
