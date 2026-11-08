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

import contextlib
import functools
import logging
import sys
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol

from rich.console import Console
from rich.logging import RichHandler

from kompend.core.paths import RUN_LOG_FILE_TEMPLATE
from kompend.types import ConfigValue

DEFAULT_LEVEL: int = logging.WARNING


#: How many items a diagnostic log line samples out of a larger collection
#: ("first 5 collisions", "e.g. these aliases"). A log line summarises; it never
#: lists the whole set, and the count is a presentation choice rather than a
#: data limit — the full set is available at the call site.
LOG_SAMPLE_LIMIT: int = 5

#: Formatted DEBUG records the flight recorder keeps in memory per run. The
#: oldest records are dropped when the ring is full, so an hour-long ingest
#: costs the same memory as a minute-long one.
RUN_LOG_RING_CAPACITY: Final[int] = 2000

#: Records at this level or above are written to the run log immediately;
#: everything below waits in the ring for a failed run to dump it.
RUN_LOG_DIRECT_LEVEL: Final[int] = logging.INFO

#: Third-party loggers whose DEBUG detail belongs in the flight recorder, not
#: the terminal. Their logger level is raised to DEBUG so the records reach the
#: root and the run-log ring; the terminal handler's own level (WARNING by
#: default) keeps them off screen. ``logging.modules`` overlays this map.
THIRD_PARTY_LOG_LEVELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "docling": "DEBUG",
        "transformers": "DEBUG",
        "httpx": "DEBUG",
        "ollama": "DEBUG",
        "RapidOCR": "CRITICAL",
        "rapidocr": "CRITICAL",
        "onnxruntime": "CRITICAL",
        "pdfminer": "WARNING",
        "PIL": "WARNING",
        "LiteLLM": "WARNING",
        "LiteLLM Router": "WARNING",
        "LiteLLM Proxy": "WARNING",
    }
)


@dataclass
class _LoggingState:
    """Mutable module state (no ``global`` statements): the two named handlers.

    ``terminal_handler`` — its level, not the root level, decides what the
    user sees, so the root logger can stay at DEBUG for the flight recorder's
    benefit without changing terminal output. ``run_log_handler`` is the
    flight recorder attached to the current run, if any.
    """

    terminal_handler: RichHandler | None = None
    run_log_handler: RunLogHandler | None = None
    module_levels: dict[str, int] = field(default_factory=dict)


class _ConfigLike(Protocol):
    """The piece of a resolved config :func:`apply_logging_config` needs."""

    def get(self, key: str, default: ConfigValue = None) -> ConfigValue: ...


_state: Final[_LoggingState] = _LoggingState()


class RunLogHandler(logging.Handler):
    """Flight recorder for one CLI run: INFO+ straight to the file, DEBUG held.

    Writes records at :data:`RUN_LOG_DIRECT_LEVEL` (INFO) or above to the run
    log immediately, and keeps the last ``capacity`` *formatted* DEBUG records
    in a ring buffer that is written to the same file only when the run fails
    — a clean run leaves an INFO-only file, a failed run lands the debug ring
    next to it.

    Not a ``logging.handlers.MemoryHandler``: the stdlib handler's flush
    trigger is ``flushLevel`` alone, so its buffer grows past ``capacity``
    unbounded — this ring drops the oldest records and keeps memory flat for
    hour-long runs.

    The handler never raises: a log handler that breaks the command it
    observes is worse than a missing log. An unwritable vault disables the
    handler (every write becomes a no-op); errors during a write are dropped.
    """

    def __init__(
        self,
        path: Path,
        *,
        capacity: int = RUN_LOG_RING_CAPACITY,
        level: int = logging.DEBUG,
    ) -> None:
        """Prepare the run log file at *path* (``a`` mode) and an empty ring."""
        super().__init__(level=level)
        self.path = path
        self._ring: deque[str] = deque(maxlen=max(1, capacity))
        self._file = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")
        except OSError as exc:
            # Never crash the command: record the failure once at debug and
            # become a silent no-op (the ledger module's pattern).
            logging.getLogger(__name__).debug("run log unavailable (%s): %s", path, exc)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        """INFO+ goes to the file now; DEBUG waits in the ring."""
        if self._file is None:
            return
        try:
            line = self.format(record)
            if record.levelno >= RUN_LOG_DIRECT_LEVEL:
                self._file.write(f"{line}\n")
                self._file.flush()
            else:
                self._ring.append(line)
        except Exception:  # noqa: BLE001 - a log write must never break the command
            self.handleError(record)

    def dump_ring(self, *, reason: str) -> None:
        """Write the buffered DEBUG records to the file (a failed run).

        *reason* names why the ring is being dumped, so the section in the
        file is self-explaining (``status=error``, ``status=cancelled``).
        """
        if self._file is None or not self._ring:
            return
        try:
            lock = self.lock
            if lock is not None:
                lock.acquire()
            try:
                self._file.write(f"\n--- debug ring ({len(self._ring)} records) — {reason} ---\n")
                for line in self._ring:
                    self._file.write(f"{line}\n")
                self._file.flush()
            finally:
                if lock is not None:
                    lock.release()
        except Exception:  # noqa: BLE001 - a log write must never break the command
            logging.getLogger(__name__).debug("run-log ring dump failed", exc_info=True)

    def close(self) -> None:
        """Close the run log file (the ring is discarded unless dumped first)."""
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None
        super().close()


@functools.cache
def get_console() -> Console:
    """Cached stdout console.

    Auto-detects terminal width on every render.  Colour is suppressed
    when stdout is not a TTY (CI, pipes).
    """
    return Console(stderr=False, no_color=not sys.stdout.isatty())


def attach_run_log(run_id: str, logs_dir: Path) -> RunLogHandler:
    """Start recording the run to ``logs_dir/run-<run_id>.log``.

    The handler is attached to the root logger and replaces any recorder left
    over from a previous run in the same process. The root logger drops to
    DEBUG so the ring sees everything; the terminal handler's own level keeps
    terminal output unchanged.
    """
    handler = RunLogHandler(logs_dir / RUN_LOG_FILE_TEMPLATE.format(run_id=run_id))
    detach_run_log(failed=False)
    _state.run_log_handler = handler
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG)
    return handler


def detach_run_log(*, failed: bool, reason: str = "") -> None:
    """Detach the recorder; dump the DEBUG ring first when *failed*.

    Never raises — the caller may already be unwinding a crash.
    """
    handler = _state.run_log_handler
    if handler is None:
        return
    _state.run_log_handler = None
    try:
        if failed:
            handler.dump_ring(reason=reason or "run failed")
        logging.root.removeHandler(handler)
        handler.close()
    except Exception:  # noqa: BLE001 - teardown must not break the command
        logging.getLogger(__name__).debug("run log detach failed", exc_info=True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger routed to the shared Rich console.

    Drop-in replacement for ``logging.getLogger(name)``.  The first call
    installs RichHandler on the root logger; all subsequent calls are free.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
    """
    configure_logging()
    return logging.getLogger(name)


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
    # Move the terminal handler's level, not the root: the root stays at
    # DEBUG so the flight recorder's ring keeps filling (see
    # configure_logging).
    if _state.terminal_handler is None:
        configure_logging()
    if _state.terminal_handler is not None:
        _state.terminal_handler.setLevel(level)


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
        level=DEFAULT_LEVEL,
    )
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    # The root level stays at DEBUG: the terminal handler's own level (set
    # above, moved by set_verbosity) is what the user sees, and the flight
    # recorder (attach_run_log) needs DEBUG records to reach the root.
    logging.root.setLevel(logging.DEBUG)
    _state.terminal_handler = handler


@functools.cache
def get_err_console() -> Console:
    """Cached stderr console used by log output and progress bars.

    Kept separate from the stdout console so progress and log lines do
    not interleave with program output.
    """
    return Console(stderr=True, no_color=not sys.stderr.isatty())


def apply_logging_config(config: _ConfigLike | None) -> None:
    """Apply the resolved ``[logging] modules`` mapping (knox-nnm6).

    Reads ``logging.modules`` through the resolved config's dotted lookup and
    re-routes the third-party loggers with the operator's levels on top.
    Safe to call repeatedly; an empty mapping keeps the built-in levels.
    """
    raw = config.get("logging.modules") if config is not None else None
    # ConfigValue is the wide JSON union; the LoggingConfig model guarantees
    # str levels, the isinstance filter just narrows the static type.
    overrides = {str(name): level for name, level in raw.items() if isinstance(level, (int, str))} if isinstance(raw, dict) and raw else None
    route_third_party_loggers(overrides=overrides)


def route_third_party_loggers(overrides: Mapping[str, int | str] | None = None) -> None:
    """Send third-party DEBUG detail to the run log and keep it off the terminal.

    Applies :data:`THIRD_PARTY_LOG_LEVELS` with *overrides* on top (the
    operator's ``logging.modules`` config). Call after
    :func:`configure_logging`; the terminal handler's level still decides what
    the user sees, so ``docling = "DEBUG"`` costs nothing on screen.
    """
    merged: dict[str, int | str] = dict(THIRD_PARTY_LOG_LEVELS)
    if overrides:
        merged.update(overrides)
    set_module_levels(merged)


def set_module_levels(levels: Mapping[str, int | str]) -> None:
    """Set the level of named loggers, taming third-party chatter per module.

    Called after :func:`configure_logging` with the resolved
    ``logging.modules`` mapping. A logger's own level gates both the terminal
    handler and the flight recorder, so ``{"docling": "DEBUG"}`` records its
    detail in ``.kompend/logs/run-<id>.log`` while the terminal handler's
    WARNING threshold keeps it off screen.

    Replacing the mapping resets the loggers the previous call touched, so a
    second in-process invocation does not inherit the first one's overrides.

    Args:
        levels: Logger name → level name (case-insensitive) or integer level.

    Raises:
        ValueError: If a level name is not recognised. Nothing is applied in
            that case, so a typo cannot leave a half-configured run.
    """
    resolved: dict[str, int] = {}
    for name, level in levels.items():
        if isinstance(level, str):
            level_int = logging.getLevelNamesMapping().get(level.upper())
            if level_int is None:
                msg = f"Invalid logging level {level!r} for module {name!r}"
                raise ValueError(msg)
            resolved[name] = level_int
        else:
            resolved[name] = level
    for name in _state.module_levels:
        logging.getLogger(name).setLevel(logging.NOTSET)
    for name, level_int in resolved.items():
        logging.getLogger(name).setLevel(level_int)
    _state.module_levels = resolved
