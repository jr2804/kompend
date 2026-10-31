"""In-process CLI test harness.

The CLI runs on Cyclopts, which ships no ``CliRunner``. Invoking the app object
directly would bypass the root meta command, so ``-w``/``--config`` in front of
a sub-command would not bind. :class:`CliRunner` therefore drives ``app.meta``
exactly like ``kompend.cli.app.main`` does, and captures the two streams
separately (several tests assert on stdout alone).

Errors are not formatted the way ``main()`` formats them: a test that expects a
crashed command needs the real exception object, hence ``catch_exceptions``
semantics from Typer's runner — unexpected exceptions land in
``result.exception`` with ``exit_code == 1`` instead of being swallowed.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Any

from cyclopts import App
from cyclopts.exceptions import CycloptsError

#: Exit code Cyclopts uses for usage errors (unknown option, bad value).
_USAGE_ERROR_EXIT_CODE = 2


@dataclass
class CliResult:
    """Outcome of one in-process CLI invocation."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    exception: BaseException | None = None
    returned: Any = field(default=None)

    @property
    def output(self) -> str:
        """Stdout and stderr concatenated, in that order."""
        return self.stdout + self.stderr


class CliRunner:
    """Minimal drop-in for ``typer.testing.CliRunner``."""

    def __init__(self, *_: Any, **__: Any) -> None:
        """Accept and ignore Typer's constructor options (``mix_stderr``, ``charset``)."""

    def invoke(
        self,
        app: App,
        args: list[str] | None = None,
        *,
        catch_exceptions: bool = True,
        **_: Any,
    ) -> CliResult:
        """Run ``args`` against ``app`` and capture what it printed."""
        argv = [str(item) for item in (args or [])]
        out, err = io.StringIO(), io.StringIO()
        result = CliResult()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                result.returned = app.meta(argv, exit_on_error=False, result_action="return_value")
            except SystemExit as exc:
                result.exit_code = exc.code if isinstance(exc.code, int) else 0
            except CycloptsError as exc:
                result.exit_code = _USAGE_ERROR_EXIT_CODE
                result.stderr = f"{exc}\n"
            except BaseException as exc:  # noqa: BLE001 — the runner reports, it does not handle
                if not catch_exceptions:
                    raise
                result.exception = exc
                result.exit_code = 1
        result.stdout = out.getvalue()
        result.stderr += err.getvalue()
        if isinstance(result.returned, int):
            result.exit_code = result.returned
        return result
