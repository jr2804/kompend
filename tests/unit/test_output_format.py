"""``format_output`` rumdl sweep — command-line batching over large vaults."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from kompend.output.renderer import (
    _command_line_batches,
    _md_files_under,
    format_output,
)

WINDOWS_CMD_LIMIT = 32_767

#: ``rumdl fmt`` file arguments start after the fixed base command
#: (``rumdl fmt --silent`` / ``rumdl fmt --check``) when no flavor/config is set.
_FILE_ARG_OFFSET = 3


class TestCommandLineBatches:
    def test_single_file_single_batch(self) -> None:
        files = [Path("a.md")]
        assert list(_command_line_batches(files, ["rumdl", "fmt", "--silent"])) == [files]

    def test_long_paths_split_across_batches(self) -> None:
        base = ["rumdl", "fmt", "--silent"]
        files = [Path(f"{'x' * 100}_{i:04d}.md") for i in range(500)]
        batches = list(_command_line_batches(files, base))
        assert len(batches) > 1
        assert [p for batch in batches for p in batch] == files  # once each, in order
        for batch in batches:
            cmdline = subprocess.list2cmdline([*base, *(str(p) for p in batch)])
            assert len(cmdline) < WINDOWS_CMD_LIMIT

    def test_oversized_single_path_still_scheduled(self) -> None:
        # A single path longer than the budget must not be dropped.
        files = [Path(f"{'y' * 20_000}.md")]
        assert list(_command_line_batches(files, ["rumdl", "fmt"])) == [files]


class TestFormatOutputBatching:
    def test_vault_sweep_stays_under_windows_limit(self, tmp_path: Path, rumdl_ok: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """657 ~100-char paths reproduce the p863 WinError 206 failure."""
        for i in range(657):
            (tmp_path / f"{'p' * 90}-{i:04d}.md").write_text("# t\n")
        files = _md_files_under([tmp_path])
        assert len(files) == 657

        calls: list[list[str]] = []
        _fake_run(monkeypatch, calls)
        assert format_output(tmp_path) == []  # fake run touches nothing

        assert len(calls) > 1
        for cmd in calls:
            assert len(subprocess.list2cmdline(cmd)) < WINDOWS_CMD_LIMIT
        scheduled = [Path(arg) for cmd in calls for arg in cmd[_FILE_ARG_OFFSET:]]
        assert sorted(scheduled) == files

    def test_changed_files_aggregate_across_batches(self, tmp_path: Path, rumdl_ok: None, monkeypatch: pytest.MonkeyPatch) -> None:
        for i in range(40):
            (tmp_path / f"{i:03d}.md").write_text("# t\n")
        files = sorted(tmp_path.glob("*.md"))

        calls: list[list[str]] = []

        def run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            for arg in cmd[_FILE_ARG_OFFSET:]:
                Path(arg).write_text("# t\n\n")  # rumdl "formats"
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr("kompend.output.renderer.subprocess.run", run)
        assert sorted(format_output(tmp_path)) == files

    def test_mid_sweep_failure_reports_formatted_prefix(self, tmp_path: Path, rumdl_ok: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed batch stops the sweep but earlier batches' edits still count."""
        monkeypatch.setattr("kompend.output.renderer._COMMAND_LINE_CHAR_BUDGET", 40)
        for i in range(10):
            (tmp_path / f"{i:03d}.md").write_text("# t\n")

        calls: list[list[str]] = []
        counter = {"n": 0}

        def run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
            counter["n"] += 1
            if counter["n"] == 2:  # fail only the second invocation
                raise OSError("[WinError 206] The filename or extension is too long")
            calls.append(cmd)
            for arg in cmd[_FILE_ARG_OFFSET:]:
                Path(arg).write_text("# t\n\n")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr("kompend.output.renderer.subprocess.run", run)
        changed = format_output(tmp_path)

        # Exactly two invocations attempted — the sweep stopped at the failure —
        # and only the first (successful) batch's edits are reported.
        assert counter["n"] == 2
        assert sorted(changed) == sorted(Path(arg) for arg in calls[0][_FILE_ARG_OFFSET:])

    def test_check_only_aggregates_batches(self, tmp_path: Path, rumdl_ok: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("kompend.output.renderer._COMMAND_LINE_CHAR_BUDGET", 40)
        for i in range(10):
            (tmp_path / f"{i:03d}.md").write_text("# t\n")
        files = sorted(tmp_path.glob("*.md"))

        calls: list[list[str]] = []
        _fake_run(monkeypatch, calls, returncode=1)  # every batch would change
        assert format_output(tmp_path, check_only=True) == files
        assert len(calls) > 1

    def test_rumdl_missing_is_silent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("kompend.output.renderer._check_rumdl", lambda: False)
        assert format_output(tmp_path) == []


@pytest.fixture
def rumdl_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the PATH probe so the sweep logic is testable anywhere."""
    monkeypatch.setattr("kompend.output.renderer._check_rumdl", lambda: True)


def _fake_run(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[list[str]],
    *,
    returncode: int = 0,
    fail_on: int | None = None,
) -> None:
    """Stub ``subprocess.run`` as seen from ``kompend.output.renderer``."""

    def run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if fail_on is not None and len(calls) == fail_on:
            raise OSError("[WinError 206] The filename or extension is too long")
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=returncode, stdout="", stderr="")

    monkeypatch.setattr("kompend.output.renderer.subprocess.run", run)
