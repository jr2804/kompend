"""Run-level context shared by every command invocation.

The CLI resolves its global options (``--wiki-dir``, ``--config``,
``--no-user-tracking``) once, before a command runs. Commands receive them
through this context instead of threading a framework context object through
forty signatures.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

#: Ambient run context, set by :func:`use_run_context` while a command runs.
_run_context: ContextVar[RunContext | None] = ContextVar("kompend_run_context", default=None)


@dataclass(frozen=True, slots=True)
class RunContext:
    """Global option values resolved for the current invocation."""

    wiki_dir: str | None = None
    config_file: str | None = None
    no_user_tracking: bool = False


def current_run_context() -> RunContext:
    """Return the run context of the invocation in progress."""
    return _run_context.get() or RunContext()


@contextmanager
def use_run_context(context: RunContext) -> Generator[RunContext]:
    """Make ``context`` the ambient run context for the enclosed block."""
    token = _run_context.set(context)
    try:
        yield context
    finally:
        _run_context.reset(token)
