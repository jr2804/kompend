"""Failure reporting must survive a console that has stopped accepting writes.

A Windows terminal that loses its pipe raises ``OSError: [WinError 6] The handle
is invalid`` from ``sys.stderr.write``. Before the fallback existed, that turned
a command failure into a silent exit: the handler raised while reporting the
error, and the second exception went to the same dead stream.
"""

from __future__ import annotations

import pytest

from kompend.cli import app as cli_app


class _DeadStream:
    """A stream whose every write fails, like a closed or orphaned console."""

    @staticmethod
    def write(_text: str) -> int:
        raise OSError(9, "The handle is invalid")

    @staticmethod
    def flush() -> None:
        raise OSError(9, "The handle is invalid")


class TestWriteFailure:
    def test_the_message_goes_to_stderr_when_it_can(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        cli_app._write_failure("boom")

        assert capsys.readouterr().err == "boom\n"

    def test_a_dead_stderr_falls_back_to_the_raw_descriptor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        written: list[bytes] = []
        monkeypatch.setattr(cli_app.sys, "stderr", _DeadStream())
        monkeypatch.setattr(cli_app.os, "write", lambda _fd, data: written.append(data) or len(data))

        cli_app._write_failure("boom")

        assert written == [b"boom\n"]

    def test_a_newline_is_added_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        written: list[bytes] = []
        monkeypatch.setattr(cli_app.sys, "stderr", _DeadStream())
        monkeypatch.setattr(cli_app.os, "write", lambda _fd, data: written.append(data) or len(data))

        cli_app._write_failure("boom\n")

        assert written == [b"boom\n"]

    def test_both_writers_dead_is_still_not_an_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _broken(_fd: int, _data: bytes) -> int:
            raise OSError(9, "The handle is invalid")

        monkeypatch.setattr(cli_app.sys, "stderr", _DeadStream())
        monkeypatch.setattr(cli_app.os, "write", _broken)

        cli_app._write_failure("boom")


class TestEndRun:
    def test_a_broken_audit_trail_does_not_replace_the_outcome(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _broken(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("the run log is unavailable")

        monkeypatch.setattr(cli_app, "log_run_end", _broken)

        cli_app._end_run(None, status="error", error="boom")
