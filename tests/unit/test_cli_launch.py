"""The launcher's thread-pool cap, its version fast path and its failure hooks."""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest

import kompend
from kompend.cli import launch
from kompend.cli._registry import OnDemandModule
from kompend.core.env import EnvVar

_THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class TestThreadCap:
    def test_every_numeric_pool_is_bounded(self) -> None:
        launch._cap_thread_pools()

        for name in _THREAD_VARS:
            assert 1 <= int(os.environ[name]) <= 8

    def test_an_explicit_cap_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(EnvVar.BLAS_THREADS.value, "2")

        launch._cap_thread_pools()

        assert all(os.environ[name] == "2" for name in _THREAD_VARS)

    def test_an_already_pinned_pool_is_not_overwritten(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")

        launch._cap_thread_pools()

        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert os.environ["OMP_NUM_THREADS"] != ""

    def test_blank_override_falls_back_to_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(EnvVar.BLAS_THREADS.value, "   ")

        launch._cap_thread_pools()

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
        monkeypatch.setattr(launch.sys, "argv", ["kompend", "--version"])

        launch.main()

        assert capsys.readouterr().out.strip() == kompend.__version__
        assert OnDemandModule.APP.value not in imported

    def test_short_version_flag_is_answered_too(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr(launch.sys, "argv", ["kompend", "-v"])

        launch.main()

        assert capsys.readouterr().out.strip() == kompend.__version__

    def test_other_arguments_delegate_to_the_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        class _FakeApp:
            @staticmethod
            def main() -> None:
                calls.append("main")

        monkeypatch.setattr(launch.importlib, "import_module", lambda _name: _FakeApp)
        monkeypatch.setattr(launch.sys, "argv", ["kompend", "status"])

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

        launch._install_failure_hooks()

        assert threading.excepthook is launch._log_thread_failure
        assert sys.unraisablehook is launch._log_unraisable

    def test_a_dead_thread_is_logged_with_its_name_and_error(self, caplog: pytest.LogCaptureFixture) -> None:
        error = ValueError("boom")
        args = SimpleNamespace(
            exc_type=ValueError,
            exc_value=error,
            exc_traceback=error.__traceback__,
            thread=SimpleNamespace(name="Thread-1"),
        )

        with caplog.at_level("ERROR", logger="kompend.cli.launch"):
            launch._log_thread_failure(cast("threading.ExceptHookArgs", args))

        message = caplog.records[0].getMessage()
        assert "thread Thread-1 died" in message
        assert "boom" in message

    def test_an_unraisable_exception_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        unraisable = SimpleNamespace(
            exc_type=RuntimeError,
            exc_value=RuntimeError("no __del__ for you"),
            exc_traceback=None,
            object=object(),
        )

        with caplog.at_level("ERROR", logger="kompend.cli.launch"):
            launch._log_unraisable(unraisable)

        assert "unraisable exception" in caplog.records[0].getMessage()

    def test_a_failing_report_falls_back_to_the_interpreter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fallbacks: list[str] = []
        monkeypatch.setattr(launch, "_failure_logger", _Exploding)
        args = SimpleNamespace(
            exc_type=ValueError,
            exc_value=ValueError("boom"),
            exc_traceback=None,
            thread=SimpleNamespace(name="Thread-1"),
        )
        monkeypatch.setattr(threading, "__excepthook__", lambda _args: fallbacks.append("excepthook"))

        launch._log_thread_failure(cast("threading.ExceptHookArgs", args))

        assert fallbacks == ["excepthook"]


class _Exploding:
    """A logger whose write fails, as a closed or starved handler would."""

    @staticmethod
    def error(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("the reporter cannot write")


@pytest.fixture(autouse=True)
def _clean_pool_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the parent process's own caps so each test starts from a known state."""
    for name in (*_THREAD_VARS, EnvVar.BLAS_THREADS.value):
        monkeypatch.delenv(name, raising=False)
