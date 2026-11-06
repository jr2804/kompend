"""The launcher's thread-pool cap, its version fast path, its failure hooks and the console-stream guard."""

from __future__ import annotations

import errno
import logging
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import kompend
from kompend.cli import launch
from kompend.cli.runtime import console, faults, thread_pools
from kompend.cli.tree.registry import OnDemandModule
from kompend.core.env import EnvVar

_THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class _CodePage:
    """A ``kernel32`` stand-in recording the code page it was given."""

    def __init__(self, current: int) -> None:
        self.current = current
        self.set_to: int | None = None

    def GetConsoleOutputCP(self) -> int:
        return self.current

    def SetConsoleOutputCP(self, code_page: int) -> int:
        self.set_to = code_page
        self.current = code_page
        return 1


class TestConsoleCodePage:
    def test_an_oem_code_page_is_switched_to_utf8(self) -> None:
        api = _CodePage(850)

        console.ensure_utf8_code_page(api)

        assert api.set_to == 65001

    def test_a_host_without_a_console_is_left_alone(self) -> None:
        api = _CodePage(0)

        console.ensure_utf8_code_page(api)

        assert api.set_to is None

    def test_an_already_utf8_console_is_left_alone(self) -> None:
        api = _CodePage(65001)

        console.ensure_utf8_code_page(api)

        assert api.set_to is None

    def test_a_failing_console_call_does_not_raise(self) -> None:
        class _Broken(_CodePage):
            def SetConsoleOutputCP(self, code_page: int) -> int:
                raise OSError(errno.EINVAL, "nope")

        console.ensure_utf8_code_page(_Broken(850))


class TestThreadCap:
    def test_every_numeric_pool_is_bounded(self) -> None:
        thread_pools.cap_thread_pools()

        for name in _THREAD_VARS:
            assert 1 <= int(os.environ[name]) <= 8

    def test_an_explicit_cap_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(EnvVar.BLAS_THREADS.value, "2")

        thread_pools.cap_thread_pools()

        assert all(os.environ[name] == "2" for name in _THREAD_VARS)

    def test_an_already_pinned_pool_is_not_overwritten(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")

        thread_pools.cap_thread_pools()

        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert os.environ["OMP_NUM_THREADS"] != ""

    def test_blank_override_falls_back_to_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(EnvVar.BLAS_THREADS.value, "   ")

        thread_pools.cap_thread_pools()

        assert int(os.environ["OPENBLAS_NUM_THREADS"]) >= 1


class TestVersionFastPath:
    def test_a_lone_version_flag_prints_without_the_command_tree(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        imported: list[str] = []
        real_import: Callable[..., object] = launch.importlib.import_module

        def _guarded(name: str, package: str | None = None) -> object:
            imported.append(name)
            if name == OnDemandModule.APP.value:
                msg = "the command tree must not be imported to print the version"
                raise AssertionError(msg)
            return real_import(name, package)

        monkeypatch.setattr(launch.importlib, "import_module", _guarded)
        monkeypatch.setattr(sys, "argv", ["kompend", "--version"])

        launch.main()

        assert capsys.readouterr().out.strip() == kompend.__version__
        assert OnDemandModule.APP.value not in imported

    def test_short_version_flag_is_answered_too(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr(sys, "argv", ["kompend", "-v"])

        launch.main()

        assert capsys.readouterr().out.strip() == kompend.__version__

    def test_other_arguments_delegate_to_the_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        class _FakeApp:
            @staticmethod
            def main() -> None:
                calls.append("main")

        monkeypatch.setattr(launch.importlib, "import_module", lambda _name: _FakeApp)
        monkeypatch.setattr(sys, "argv", ["kompend", "status"])

        launch.main()

        assert calls == ["main"]


class TestFailureHooks:
    """A dying thread must name itself and show its traceback.

    The interpreter's own reporter prints a header and *then* formats the
    traceback, so an allocation failure kills the report as well: all that
    survives is a spray of ``Exception in thread …`` / ``Exception in
    threading.excepthook:`` headers with nothing under them.
    """

    def test_installing_replaces_both_hooks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(threading, "excepthook", threading.__excepthook__)
        monkeypatch.setattr(sys, "unraisablehook", sys.__unraisablehook__)

        faults.install_failure_hooks()

        assert threading.excepthook is faults.log_thread_failure
        assert sys.unraisablehook is faults.log_unraisable

    def test_a_dead_thread_is_logged_with_its_name_and_error(self, caplog: pytest.LogCaptureFixture) -> None:
        error = ValueError("boom")
        args = SimpleNamespace(
            exc_type=ValueError,
            exc_value=error,
            exc_traceback=error.__traceback__,
            thread=SimpleNamespace(name="Thread-1"),
        )

        with caplog.at_level("ERROR", logger="kompend.cli.runtime.faults"):
            faults.log_thread_failure(cast("threading.ExceptHookArgs", args))

        message = caplog.records[0].getMessage()
        assert "thread Thread-1 died" in message
        assert "boom" in message

    def test_an_identical_repeat_is_demoted_to_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        def _failure(name: str, message: str) -> SimpleNamespace:
            error = ValueError(message)
            return SimpleNamespace(
                exc_type=ValueError,
                exc_value=error,
                exc_traceback=error.__traceback__,
                thread=SimpleNamespace(name=name),
            )

        with caplog.at_level("DEBUG", logger="kompend.cli.runtime.faults"):
            faults.log_thread_failure(cast("threading.ExceptHookArgs", _failure("Thread-1", "boom")))
            faults.log_thread_failure(cast("threading.ExceptHookArgs", _failure("Thread-1", "boom")))
            faults.log_thread_failure(cast("threading.ExceptHookArgs", _failure("Thread-1", "boom")))

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(errors) == 1
        assert len(debugs) == 2
        assert "died again" in debugs[0].getMessage()
        assert "#3" in debugs[1].getMessage()

    def test_a_different_failure_is_reported_in_full_again(self, caplog: pytest.LogCaptureFixture) -> None:
        first = SimpleNamespace(
            exc_type=ValueError,
            exc_value=ValueError("boom"),
            exc_traceback=None,
            thread=SimpleNamespace(name="Thread-1"),
        )
        second = SimpleNamespace(
            exc_type=OSError,
            exc_value=OSError(9, "The handle is invalid"),
            exc_traceback=None,
            thread=SimpleNamespace(name="Thread-1"),
        )

        with caplog.at_level("ERROR", logger="kompend.cli.runtime.faults"):
            faults.log_thread_failure(cast("threading.ExceptHookArgs", first))
            faults.log_thread_failure(cast("threading.ExceptHookArgs", second))

        assert len(caplog.records) == 2
        assert "The handle is invalid" in caplog.records[1].getMessage()

    def test_an_unraisable_exception_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        unraisable = SimpleNamespace(
            exc_type=RuntimeError,
            exc_value=RuntimeError("no __del__ for you"),
            exc_traceback=None,
            object=object(),
        )

        with caplog.at_level("ERROR", logger="kompend.cli.runtime.faults"):
            faults.log_unraisable(unraisable)

        assert "unraisable exception" in caplog.records[0].getMessage()

    def test_a_failing_report_falls_back_to_the_interpreter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fallbacks: list[str] = []
        monkeypatch.setattr(faults, "_failure_logger", _Exploding)
        args = SimpleNamespace(
            exc_type=ValueError,
            exc_value=ValueError("boom"),
            exc_traceback=None,
            thread=SimpleNamespace(name="Thread-1"),
        )
        monkeypatch.setattr(threading, "__excepthook__", lambda _args: fallbacks.append("excepthook"))

        faults.log_thread_failure(cast("threading.ExceptHookArgs", args))

        assert fallbacks == ["excepthook"]

    @pytest.fixture(autouse=True)
    def _fresh_failure_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The seen-failure cache is module state; tests start empty."""
        monkeypatch.setattr(faults, "_THREAD_FAILURE_SEEN", {})


class _Exploding:
    """A logger whose write fails, as a closed or starved handler would."""

    @staticmethod
    def error(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("the reporter cannot write")


class TestNeverSilent:
    """A failure must leave a trace even when the process is about to vanish."""

    def test_the_fault_log_is_opened_once_and_names_the_run(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        enabled: list[dict[str, object]] = []
        _fresh_fault_handler(monkeypatch)
        monkeypatch.setattr(faults.tempfile, "gettempdir", lambda: str(tmp_path))
        monkeypatch.setattr(faults.faulthandler, "enable", lambda **kwargs: enabled.append(kwargs))
        monkeypatch.setattr(sys, "argv", ["kompend", "ingest"])

        faults.enable_fault_handler()
        faults.enable_fault_handler()

        log = tmp_path / "kompend-faults.log"
        assert log.exists()
        assert "argv=['ingest']" in log.read_text(encoding="utf-8")
        assert len(enabled) == 1
        assert enabled[0]["all_threads"] is True

    def test_an_unopenable_log_falls_back_to_stderr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        enabled: list[dict[str, object]] = []
        _fresh_fault_handler(monkeypatch)
        monkeypatch.setattr(faults.tempfile, "gettempdir", lambda: "Z:/does-not-exist")
        monkeypatch.setattr(faults.faulthandler, "enable", lambda **kwargs: enabled.append(kwargs))

        faults.enable_fault_handler()

        assert faults.fault_log_stream() is sys.stderr
        assert enabled[0]["file"] is sys.stderr


class TestHardenedStreams:
    """A console handle that dies mid-run must not take the pipeline with it.

    Captured live on 2026-11-04: after the first heavy parse, every stderr write
    raised ``OSError(9, 'The handle is invalid', winerror 6)`` — 1468 times —
    while the pipeline itself kept running. Unwrapped, the same breakage ends a
    real CLI run as CPython's silent exit status 120.
    """

    def test_a_dead_handle_write_is_absorbed_and_reports_success(self) -> None:
        raw = _DeadHandleStream()
        stream = console.HardenedStream(raw, name="stderr")

        written = stream.write("progress bar frame\n")

        assert written == len("progress bar frame\n")
        assert raw.writes == ["progress bar frame\n"]  # the attempt was made; the handle refused it

    def test_a_dead_handle_flush_is_absorbed(self) -> None:
        stream = console.HardenedStream(_DeadHandleStream(), name="stderr")

        stream.flush()  # must not raise — this is what prevents exit status 120

    def test_any_other_oserror_still_propagates(self) -> None:
        raw = _DeadHandleStream(failure=OSError(28, "No space left on device"))
        stream = console.HardenedStream(raw, name="stdout")

        with pytest.raises(OSError, match="No space left"):
            stream.write("data")
        with pytest.raises(OSError, match="No space left"):
            stream.flush()

    def test_the_death_is_noted_once_and_the_sibling_is_told(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(faults.tempfile, "gettempdir", lambda: str(tmp_path))
        sibling_writes: list[str] = []
        monkeypatch.setattr(sys, "stdout", _RecordingStream(sibling_writes), raising=False)
        stream = console.HardenedStream(_DeadHandleStream(), name="stderr")

        stream.write("one")
        stream.write("two")
        stream.flush()

        fault_log = tmp_path / "kompend-faults.log"
        assert "console stderr died" in fault_log.read_text(encoding="utf-8")
        assert len(sibling_writes) == 1
        assert "no longer writable" in sibling_writes[0]

    def test_a_rebound_handle_is_repaired_and_the_write_succeeds(self, tmp_path: Path) -> None:
        """A backend rebinding fd 2 must not silence the console for the whole run.

        The captured cause (2026-11-04): a native capture window re-points
        descriptor 2 with ``os.dup2`` around a conversion; the buffered stream
        bound to the old handle keeps raising ``OSError 9``. The proxy reopens a
        fresh stream over the current descriptor, so the console comes back.
        """
        target = tmp_path / "console.log"
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            stream = console.HardenedStream(_DeadHandleStream(descriptor=descriptor), name="stderr")

            written = stream.write("after repair\n")

            assert written == len("after repair\n")
            assert stream._repairs == 1
            assert stream._noted is False
            assert "after repair" in target.read_text(encoding="utf-8")
        finally:
            os.close(descriptor)

    def test_a_handle_that_never_recovers_is_noted_after_the_repair_budget(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(faults.tempfile, "gettempdir", lambda: str(tmp_path))
        stream = console.HardenedStream(_DeadHandleStream(), name="stderr")
        stream._repairs = console.REPAIR_LIMIT

        assert stream._repair() is False
        stream.write("gone")

        assert stream._noted is True

    def test_everything_else_delegates_to_the_raw_stream(self) -> None:
        raw = SimpleNamespace(isatty=lambda: True, fileno=lambda: 2, encoding="utf-8")
        stream = console.HardenedStream(raw, name="stderr")

        assert stream.isatty() is True
        assert stream.fileno() == 2
        assert stream.encoding == "utf-8"

    def test_hardening_wraps_both_streams_and_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stdout", SimpleNamespace(), raising=False)
        monkeypatch.setattr(sys, "stderr", SimpleNamespace(), raising=False)

        console.harden_console_streams()
        first_stdout, first_stderr = sys.stdout, sys.stderr

        console.harden_console_streams()

        assert isinstance(first_stdout, console.HardenedStream)
        assert isinstance(first_stderr, console.HardenedStream)
        assert sys.stdout is first_stdout
        assert sys.stderr is first_stderr

    def test_a_dead_handle_is_recognised_in_both_shapes(self) -> None:
        assert console.is_dead_handle(OSError(9, "The handle is invalid", None, 6, None))
        assert console.is_dead_handle(OSError(9, "Bad file descriptor"))
        assert not console.is_dead_handle(OSError(28, "No space left on device"))
        assert not console.is_dead_handle(ValueError("not an OS error"))


class _DeadHandleStream:
    """A stream whose handle was invalidated by native code mid-run."""

    def __init__(self, *, failure: OSError | None = None, descriptor: int | None = None) -> None:
        self.failure = failure or OSError(9, "The handle is invalid", None, 6, None)
        self.writes: list[str] = []
        self._descriptor = descriptor

    def write(self, data: str) -> int:
        self.writes.append(data)
        raise self.failure

    def flush(self) -> None:
        raise self.failure

    def fileno(self) -> int:
        if self._descriptor is None:
            raise OSError(9, "The handle is invalid")
        return self._descriptor


class _RecordingStream:
    """A stream that records writes instead of touching a console."""

    def __init__(self, writes: list[str]) -> None:
        self._writes = writes

    def write(self, data: str) -> int:
        self._writes.append(data)
        return len(data)

    def flush(self) -> None:
        return None


def _fresh_fault_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the cached fault-handler setup so a test can observe the first call.

    Both the enable call and the stream it writes to are cached; clearing the
    caches lets each test see a first invocation. ``faulthandler.enable`` is
    stubbed by every caller, so no real handler is installed.
    """
    monkeypatch.setattr(faults, "enable_fault_handler", faults.enable_fault_handler)
    faults.enable_fault_handler.cache_clear()
    faults.fault_log_stream.cache_clear()


@pytest.fixture(autouse=True)
def _clean_pool_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the parent process's own caps so each test starts from a known state."""
    for name in (*_THREAD_VARS, EnvVar.BLAS_THREADS.value):
        monkeypatch.delenv(name, raising=False)
