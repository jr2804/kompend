"""The launcher's thread-pool cap and its version fast path."""

from __future__ import annotations

import os
from collections.abc import Callable

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


@pytest.fixture(autouse=True)
def _clean_pool_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the parent process's own caps so each test starts from a known state."""
    for name in (*_THREAD_VARS, EnvVar.BLAS_THREADS.value):
        monkeypatch.delenv(name, raising=False)
