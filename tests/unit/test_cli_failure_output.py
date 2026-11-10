"""Failure reporting must survive a console that has stopped accepting writes.

A Windows terminal that loses its pipe raises ``OSError: [WinError 6] The handle
is invalid`` from ``sys.stderr.write``. Before the fallback existed, that turned
a command failure into a silent exit: the handler raised while reporting the
error, and the second exception went to the same dead stream.
"""

from __future__ import annotations

import logging

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

    def test_a_closed_stream_falls_back_to_the_raw_descriptor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        written: list[bytes] = []

        class _ClosedStream:
            """A stream already closed raises ValueError, not OSError."""

            @staticmethod
            def write(_text: str) -> int:
                raise ValueError("I/O operation on closed file")

            @staticmethod
            def flush() -> None:
                raise ValueError("I/O operation on closed file")

        monkeypatch.setattr(cli_app.sys, "stderr", _ClosedStream())
        monkeypatch.setattr(cli_app.os, "write", lambda _fd, data: written.append(data) or len(data))

        cli_app._write_failure("boom")

        assert written == [b"boom\n"]

    def test_the_both_writers_dead_is_still_not_an_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_the_swallowed_failure_is_recorded(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def _broken(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("the run log is unavailable")

        monkeypatch.setattr(cli_app, "log_run_end", _broken)

        with caplog.at_level(logging.DEBUG, logger="kompend.cli.app"):
            cli_app._end_run(None, status="error", error="boom")

        assert "run record could not be closed" in caplog.text

    def test_an_interrupt_is_not_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ctrl-C must keep propagating: the audit trail is not a cancellation boundary."""

        def _interrupted(*_args: object, **_kwargs: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli_app, "log_run_end", _interrupted)

        with pytest.raises(KeyboardInterrupt):
            cli_app._end_run(None, status="error", error="boom")
