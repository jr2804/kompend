"""Parse under a wall-clock deadline, in a worker process that outlives one file.

A backend can wedge: parsecraft's docling path spins forever on a small
markdown list, holding an ingest at one full core with no progress on disk.
Nothing inside the parser can interrupt that — a thread cannot
kill code that never returns from C or Rust — so a hard bound has to come from
outside the process.

A child per file would be correct and unaffordable: importing the parser stack
takes seconds, so a 300-file corpus would spend most of its time starting
interpreters. The worker here is therefore created once per ingest run and
answers one parse at a time; when a parse blows its budget the worker is killed
(the only reliable way to stop native code), the file is reported as failed, and
the next parse starts a fresh worker.

With no deadline configured (``ingest.parse_timeout_s = 0``) nothing is spawned
and parsing stays in-process, so the default path keeps its exact semantics and
its exact exception behaviour.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
from contextlib import suppress
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from kompend.core.models import ParsedDocument
from kompend.core.pipeline.backend_capture import (
    BackendBanners,
    backend_name,
    capture_backend_output,
    log_backend_banners,
    log_banner_lines,
)

#: Seconds to wait for a killed worker to disappear. A killed process normally
#: exits at once; this only bounds a native library blocking in its finaliser.
_REAP_TIMEOUT_S = 10.0

#: Seconds to wait for a fresh worker to finish importing the parser stack. This
#: is deliberately NOT the per-file budget: start-up imports take seconds, and a
#: budget tuned for parsing must not be spent on interpreter warm-up.
_STARTUP_TIMEOUT_S = 180.0

#: Sentinel the parent sends to make the worker exit instead of parsing.
_STOP = None


class ParseDeadlineExceeded(RuntimeError):
    """A parse did not finish inside its budget; the worker was killed."""

    def __init__(self, source_file: Path, timeout_s: float) -> None:
        super().__init__(
            f"parsing did not finish within {timeout_s:g} s and was killed. "
            f"Raise [ingest] parse_timeout_s (0 disables the bound) or route this file to a "
            f"faster backend (``[routing] preferred_backend`` / ``only_backend`` in parsecraft's config)."
        )
        self.source_file = source_file
        self.timeout_s = timeout_s


class DeadlineWorker:
    """Parse files one at a time under a per-file deadline.

    The worker process starts on the first parse and is reused until it dies or
    the run ends, so a corpus pays one interpreter start-up per run instead of
    one per file. The parser object is sent with each request, which is cheap
    (it is a small descriptor; its imports happen once, inside the worker) and
    keeps the worker stateless across files.
    """

    def __init__(self, timeout_s: float = 0.0) -> None:
        self._timeout_s = timeout_s
        self._process: Any = None
        self._connection: Any = None

    def __enter__(self) -> DeadlineWorker:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def parse(  # noqa: ANN401 — any object with .parse()
        self,
        parser: Any,
        target: Path,
        *,
        source: str | None = None,
        backend: str | None = None,
    ) -> ParsedDocument:
        """Parse *target* with *parser*, killing the parse when the budget expires.

        ``timeout_s <= 0`` parses in this process, keeping the caller's
        exception handling intact. A positive budget parses in the worker;
        :class:`ParseDeadlineExceeded` is raised on expiry and the worker is
        replaced on the next call.

        Either path captures what the backend prints to the process streams
        (bead knox-drz1) and logs it with attribution; *source* and *backend*
        name the file and backend in those records. The worker ships its
        captured lines to this process first — the worker's own loggers have
        no run-log handler.
        """
        if self._timeout_s <= 0:
            banners = BackendBanners(source=source or target.name, backend=backend or backend_name(parser))
            try:
                with capture_backend_output(banners):
                    return parser.parse(target)  # type: ignore[no-any-return]
            finally:
                log_backend_banners(banners)
        self._ensure_worker()
        assert self._connection is not None  # noqa: S101 — set by _ensure_worker
        self._connection.send((parser, str(target)))
        if not self._connection.poll(self._timeout_s):
            self._kill()
            raise ParseDeadlineExceeded(target, self._timeout_s)
        while True:
            try:
                status, payload = self._connection.recv()
            except EOFError:
                self._kill()
                msg = (
                    "the parse worker died without reporting a result — a native library probably aborted "
                    "the process (an OpenBLAS allocation failure does this)"
                )
                raise RuntimeError(msg) from None
            if status != "banners":
                break
            lines, suppressed, worker_backend = payload
            log_banner_lines(
                lines,
                suppressed=suppressed,
                source=source or target.name,
                backend=backend or worker_backend,
            )
        if status == "ok":
            return payload  # type: ignore[no-any-return]
        raise RuntimeError(payload)

    def close(self) -> None:
        """Stop the worker, if one is running."""
        if self._connection is not None:
            with suppress(OSError, ValueError):
                self._connection.send(_STOP)
            with suppress(Exception):
                self._connection.close()
            self._connection = None
        if self._process is not None:
            self._process.join(_REAP_TIMEOUT_S)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(_REAP_TIMEOUT_S)
            self._process = None

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self.close()
        context = mp.get_context("spawn")
        receiver, sender = context.Pipe(duplex=True)
        self._process = context.Process(target=_worker_main, args=(sender,), daemon=True)
        self._process.start()
        sender.close()
        if not receiver.poll(_STARTUP_TIMEOUT_S):
            receiver.close()
            self._kill()
            msg = f"the parse worker did not start within {_STARTUP_TIMEOUT_S:g} s"
            raise RuntimeError(msg)
        ready, _payload = receiver.recv()
        if ready != "ready":
            receiver.close()
            self._kill()
            msg = f"the parse worker reported {ready!r} instead of ready"
            raise RuntimeError(msg)
        self._connection = receiver

    def _kill(self) -> None:
        if self._connection is not None:
            with suppress(Exception):
                self._connection.close()
            self._connection = None
        if self._process is not None:
            self._process.kill()
            self._process.join(_REAP_TIMEOUT_S)
            self._process = None


def _worker_main(connection: Connection) -> None:
    """Warm up the parser stack, then answer parse requests until stopped.

    The import happens here, once, and the parent waits for the ``ready``
    handshake outside any file's budget — otherwise the first file of every run
    would be measured against an interpreter start-up it never caused.
    """
    importlib.import_module("kompend.preprocessing.parsers")
    connection.send(("ready", None))
    while True:
        try:
            request = connection.recv()
        except EOFError:
            return
        if request is _STOP:
            return
        parser, target = request
        banners = BackendBanners(source=Path(target).name, backend=backend_name(parser))
        try:
            with capture_backend_output(banners):
                document = parser.parse(Path(target))
        except BaseException as exc:  # the parent is the only reporter left, so even SystemExit must travel
            connection.send(("banners", (banners.lines, banners.suppressed, banners.backend)))
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        else:
            connection.send(("banners", (banners.lines, banners.suppressed, banners.backend)))
            connection.send(("ok", document))
