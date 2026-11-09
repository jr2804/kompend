"""Flight recorder: the per-run log file and its DEBUG ring.

``kompend.logging.RunLogHandler`` writes INFO+ straight to
``.kompend/logs/run-<id>.log`` and holds the last DEBUG records in a bounded
ring; ``log_run_start`` attaches it (the run id is minted there) and
``log_run_end`` detaches it, dumping the ring only when the run did not
finish clean.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from kompend.cli.app import _print_run_log_footer
from kompend.core.paths import VaultPaths
from kompend.core.reporting import runlog
from kompend.core.reporting.handle import RunHandle
from kompend.logging import (
    RunLogHandler,
    attach_run_log,
    detach_run_log,
    get_logger,
    set_verbosity,
)


class TestRunLogHandler:
    def test_clean_run_writes_info_only(self, tmp_path: Path) -> None:
        attach_run_log("x1", tmp_path)
        logging.getLogger("kompend.test").debug("hidden detail")
        logging.getLogger("kompend.test").info("visible line")
        detach_run_log(failed=False)

        text = _read(tmp_path / "run-x1.log")
        assert "visible line" in text
        assert "hidden detail" not in text

    def test_failed_run_dumps_the_ring(self, tmp_path: Path) -> None:
        attach_run_log("f1", tmp_path)
        logging.getLogger("kompend.test").debug("ring detail one")
        logging.getLogger("kompend.test").debug("ring detail two")
        logging.getLogger("kompend.test").info("direct line")
        detach_run_log(failed=True, reason="run ended with status='error'")

        text = _read(tmp_path / "run-f1.log")
        assert "direct line" in text
        assert "ring detail one" in text
        assert "ring detail two" in text
        assert "debug ring" in text

    def test_ring_is_bounded_and_keeps_the_newest(self, tmp_path: Path) -> None:
        log_path = tmp_path / "run-b1.log"
        handler = RunLogHandler(log_path, capacity=3)
        for i in range(5):
            handler.handle(logging.LogRecord("kompend.test", logging.DEBUG, __file__, 1, f"detail {i}", (), None))
        handler.dump_ring(reason="failed")
        handler.close()

        text = _read(log_path)
        assert "detail 0" not in text
        assert "detail 1" not in text
        assert "detail 2" in text
        assert "detail 4" in text

    def test_unwritable_vault_is_a_silent_no_op(self, tmp_path: Path) -> None:
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("occupies the path", encoding="utf-8")

        attach_run_log("f3", blocker / "logs")  # mkdir fails on a file parent

        logging.getLogger("kompend.test").debug("dropped silently")
        logging.getLogger("kompend.test").info("also dropped")
        detach_run_log(failed=True)  # must not raise

        assert not (blocker / "logs").exists()

    def test_attach_replaces_the_previous_recorder(self, tmp_path: Path) -> None:
        attach_run_log("first", tmp_path)
        attach_run_log("second", tmp_path)
        logging.getLogger("kompend.test").info("second only")
        detach_run_log(failed=False)

        assert "second only" in _read(tmp_path / "run-second.log")
        assert _read(tmp_path / "run-first.log") == ""


class TestTerminalUnchanged:
    def test_root_stays_debug_aware_and_terminal_follows_set_verbosity(self) -> None:
        get_logger("kompend.test")  # ensure configure_logging() has run
        set_verbosity(logging.WARNING)
        attach_run_log("v1", Path("."))
        try:
            rich = next(  # the terminal handler is the non-flight-recorder one
                h for h in logging.root.handlers if not isinstance(h, RunLogHandler)
            )
            assert logging.root.level == logging.DEBUG
            assert rich.level == logging.WARNING

            set_verbosity(logging.INFO)
            assert rich.level == logging.INFO
            assert logging.root.level == logging.DEBUG
        finally:
            detach_run_log(failed=False)

    def test_info_records_reach_the_file_at_default_verbosity(self, tmp_path: Path) -> None:
        set_verbosity(logging.WARNING)
        attach_run_log("v2", tmp_path)
        logging.getLogger("kompend.test").info("info while terminal is warning")
        logging.getLogger("kompend.test").debug("debug while terminal is warning")
        detach_run_log(failed=False)

        text = _read(tmp_path / "run-v2.log")
        assert "info while terminal is warning" in text
        assert "debug while terminal is warning" not in text


class TestRunlogIntegration:
    def test_start_attaches_and_end_dumps_on_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _seed_vault(tmp_path, monkeypatch)

        handle = runlog.log_run_start(cmd_tokens=["compile"])
        assert handle is not None
        vault = VaultPaths(tmp_path)
        assert handle.log_file == vault.run_log_file(handle.run_id)

        get_logger("kompend.pipeline").debug("parse stall detail")
        runlog.log_run_end(handle, status="error", error="boom")

        text = _read(vault.logs_dir / f"run-{handle.run_id}.log")
        assert "parse stall detail" in text
        assert "status='error'" in text

    def test_clean_end_leaves_info_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _seed_vault(tmp_path, monkeypatch)

        handle = runlog.log_run_start(cmd_tokens=["compile"])
        assert handle is not None
        get_logger("kompend.pipeline").debug("clean-run detail")
        runlog.log_run_end(handle, status="ok")

        assert handle.log_file is not None
        assert "clean-run detail" not in _read(handle.log_file)

    def test_start_without_vault_attaches_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)  # no .kompend dir
        assert runlog.log_run_start(cmd_tokens=["compile"]) is None
        detach_run_log(failed=True)  # must be a no-op, not a crash
        assert not (tmp_path / "logs").exists()


class TestRunLogFooter:
    """The one-line footer naming the flight recorder."""

    def test_names_the_file_relative_to_the_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.chdir(tmp_path)
        path = tmp_path / ".kompend" / "logs" / "run-r1.log"
        path.parent.mkdir(parents=True)
        path.write_text("row\n", encoding="utf-8")
        handle = RunHandle(run_id="r1", cmd="compile", db_path=tmp_path / ".kompend" / "kompend.db", start_iso="", log_file=path)

        _print_run_log_footer(handle)

        assert capsys.readouterr().out.strip() == f"Full log: {Path('.kompend/logs/run-r1.log')}"

    def test_silent_without_a_written_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.chdir(tmp_path)
        missing = tmp_path / ".kompend" / "logs" / "run-none.log"
        handle = RunHandle(run_id="none", cmd="compile", db_path=tmp_path / "kompend.db", start_iso="", log_file=missing)

        _print_run_log_footer(handle)
        _print_run_log_footer(None)

        assert capsys.readouterr().out == ""

    def test_falls_back_to_the_absolute_path_outside_the_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        path = tmp_path / "logs" / "run-r2.log"
        path.parent.mkdir()
        path.write_text("row\n", encoding="utf-8")
        handle = RunHandle(run_id="r2", cmd="compile", db_path=tmp_path / "kompend.db", start_iso="", log_file=path)

        _print_run_log_footer(handle)

        assert capsys.readouterr().out.strip() == f"Full log: {path}"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _seed_vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed a minimal vault so ``log_run_start`` resolves a state dir."""
    state_dir = tmp_path / ".kompend"
    state_dir.mkdir()
    (state_dir / "config.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
