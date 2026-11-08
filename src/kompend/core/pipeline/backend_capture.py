"""Capture third-party stdout/stderr banners into the run log (bead knox-drz1).

Backends print diagnostics straight to the process streams — docling
conversion banners, transformers weight-loading lines. That output bypasses
the logging system entirely, so no logger routing can record it (knox-ty8f.5
covers only what goes *through* logging), yet it is exactly the evidence a
surprising parse needs beside its ledger row.

This module swaps ``sys.stdout``/``sys.stderr`` for the duration of a parse
call and records what the *parsing thread* prints, forwarding it to the run
log with attribution. Design points, each deliberate:

* **Replace, not tee** — captured banners go to the run log only; the
  terminal is not flooded. Kompend's own consoles hold references to the
  real stream objects, so progress output is unaffected by the swap.
* **Per-thread attribution** — the compile pool parses concurrently, and
  ``sys.stdout`` is process-global. Each capture registers its thread, so a
  banner is charged to the parse that printed it. Writes from unregistered
  threads (a library background thread, unrelated code) pass through to the
  real stream unchanged.
* **Run-log contract intact** — captured lines are INFO records (the run
  log keeps INFO and above always); the DEBUG ring never fills with banner
  noise. A chatty backend is capped per parse with the remainder summarised.
* **No behaviour change to the parse result** — stdin is untouched, and the
  swap only redirects diagnostic output, never parser inputs or outputs.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final, TextIO

from kompend.logging import get_logger

#: Banner lines kept per parse. Progress-bar chatter (``\r`` updates) and
#: model-loading walls stop here; the count is summarised in one record.
DEFAULT_MAX_BANNER_LINES: Final[int] = 100

#: Per-line character ceiling. A banner is diagnostic output, not an LLM
#: payload — a pathological multi-megabyte line must not dominate the run log.
_MAX_LINE_CHARS: Final[int] = 4000

_UNKNOWN_BACKEND: Final[str] = "unknown-backend"

logger = get_logger(__name__)


_install_lock = threading.Lock()


#: Install/restore discipline for the process streams. First capture in
#: installs the writers, last one out restores the originals; concurrent
#: captures (compile pool threads) share one installation. One mutable
#: holder keeps the discipline without module-level rebinding.
@dataclass
class _InstallState:
    depth: int = 0
    stdout: TextIO | None = None
    stderr: TextIO | None = None


def backend_name(backend: object) -> str:
    """Best-effort identity of a routed backend, for banner attribution.

    Backends expose ``backend_identity()`` when their output depends on
    registered engines; a bare class name is the floor. Never raises —
    attribution is diagnostic, not load-bearing.
    """
    hook = getattr(backend, "backend_identity", None)
    if callable(hook):
        try:
            return str(hook()) or type(backend).__name__
        except Exception:  # noqa: BLE001 — attribution must never break a parse
            return type(backend).__name__
    return type(backend).__name__


_install = _InstallState()

#: Per-thread capture stack. The top of the stack is the parse this thread is
#: currently inside; a write is captured only when the writing thread has one.
_tls = threading.local()


@dataclass
class BackendBanners:
    """Collected banner output for one parse, with its attribution.

    Mutated by :func:`capture_backend_output` while the parse runs; read
    afterwards by :func:`log_backend_banners` (same process) or shipped to
    the parent process (deadline worker) for logging there.
    """

    source: str
    backend: str = _UNKNOWN_BACKEND
    max_lines: int = DEFAULT_MAX_BANNER_LINES
    lines: list[str] = field(default_factory=list)
    suppressed: int = 0
    _partial: list[str] = field(default_factory=list)

    def record(self, text: str, *, is_stderr: bool) -> None:  # noqa: ARG002 — merged stream
        r"""Fold one ``write()`` into complete banner lines.

        A line ending in ``\r`` was overwritten on the terminal, so only its
        final segment was ever visible — tqdm-style progress updates collapse
        to the last state instead of one record per tick. ``\r\n`` is a normal
        line terminator and is left alone.
        """
        self._partial.append(text)
        joined = "".join(self._partial).replace("\r\n", "\n")
        *complete, tail = joined.split("\n")
        self._partial = [tail] if tail else []
        for line in complete:
            self._add(_visible_segment(line))

    def finish(self) -> None:
        """Flush a trailing line without a terminator."""
        if self._partial:
            self._add(_visible_segment("".join(self._partial)))
            self._partial = []

    def _add(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        if len(self.lines) >= self.max_lines:
            self.suppressed += 1
            return
        self.lines.append(stripped[:_MAX_LINE_CHARS])


class _CapturingWriter:
    """Stream replacement that records writes from registered threads.

    Everything else — ``isatty``, ``fileno``, ``encoding`` — delegates to the
    wrapped stream so backend capability probes keep their old answers.
    """

    __slots__ = ("_fallback", "_is_stderr")

    def __init__(self, fallback: TextIO, *, is_stderr: bool) -> None:
        self._fallback = fallback
        self._is_stderr = is_stderr

    def write(self, text: str) -> int:
        """Record *text* for the capturing thread; pass strangers through."""
        state = _current_state()
        if state is None:
            return self._fallback.write(text)
        state.record(text, is_stderr=self._is_stderr)
        return len(text)

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        """Flush the fallback; captured output lives in memory, not a pipe."""

    def __getattr__(self, name: str) -> object:
        return getattr(self._fallback, name)


def _visible_segment(line: str) -> str:
    r"""The part of *line* a terminal would have left visible.

    ``\r`` returns the cursor to column zero, so every earlier segment of the
    same line was overwritten by the next one.
    """
    return line.rsplit("\r", 1)[-1]


def _current_state() -> BackendBanners | None:
    stack = getattr(_tls, "stack", None)
    return stack[-1] if stack else None


@contextmanager
def capture_backend_output(banners: BackendBanners) -> Generator[BackendBanners]:
    """Capture what the current thread prints to stdout/stderr into *banners*.

    The process streams are swapped while any capture is active and restored
    when the last one exits. The swap is thread-registered: only this
    thread's writes are recorded; everything else passes through unchanged.
    """
    stack: list[BackendBanners] = getattr(_tls, "stack", None) or []
    stack.append(banners)
    _tls.stack = stack
    with _install_lock:
        if _install.depth == 0:
            _install.stdout = sys.stdout
            _install.stderr = sys.stderr
            sys.stdout = _CapturingWriter(_install.stdout, is_stderr=False)  # type: ignore[assignment]
            sys.stderr = _CapturingWriter(_install.stderr, is_stderr=True)  # type: ignore[assignment]
        _install.depth += 1
    try:
        yield banners
    finally:
        banners.finish()
        stack.pop()
        with _install_lock:
            _install.depth -= 1
            if _install.depth == 0:
                if isinstance(sys.stdout, _CapturingWriter):
                    sys.stdout = _install.stdout  # type: ignore[assignment]
                if isinstance(sys.stderr, _CapturingWriter):
                    sys.stderr = _install.stderr  # type: ignore[assignment]
                _install.stdout = None
                _install.stderr = None


def log_backend_banners(banners: BackendBanners) -> None:
    """Log a finished collector in the process that ran the parse."""
    log_banner_lines(banners.lines, suppressed=banners.suppressed, source=banners.source, backend=banners.backend)


def log_banner_lines(
    lines: Iterable[str],
    *,
    suppressed: int = 0,
    source: str,
    backend: str = _UNKNOWN_BACKEND,
) -> None:
    """Emit captured banner lines as INFO records for the run log.

    INFO (not DEBUG) on purpose: the run log keeps INFO and above always, so
    a banner survives even when the run succeeds — that is the whole point of
    capturing it.
    """
    for line in lines:
        logger.info("backend banner [%s] %s: %s", backend, source, line)
    if suppressed:
        logger.info(
            "backend banner [%s] %s: … and %d more banner line(s) over the per-parse cap",
            backend,
            source,
            suppressed,
        )
