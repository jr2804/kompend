"""Smoke layer: run the real CLI, the way the user runs it.

Unit tests mock the seams where these bugs live - a test double returns `str`
while the real agent returns `BaseModel | str`, or a cache entry is written and
read by the same in-process object. These tests spawn `python -m kompend`, so
Typer parsing, plugin loading and the real wiring all run, and a crash shows up
as a traceback instead of a green assertion.

They are cheap (no model calls) and cover the surfaces that broke before:

- `--config` is a root flag only
- `init` scaffolds a vault that `config check`, `status` and `doctor` accept
- `lint show` reads a per-run artefact
- `compile` on an empty vault fails with a clean message, not a traceback
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

#: A spawned CLI must never print a traceback at the user.
TRACEBACK = "Traceback (most recent call last)"


class TestCliSurface:
    def test_version(self) -> None:
        result = _run("--version")

        assert result.returncode == 0
        assert TRACEBACK not in _output(result)

    def test_root_help_offers_config(self) -> None:
        result = _run("--help")

        assert result.returncode == 0
        assert "--config" in result.stdout

    def test_commands_do_not_redeclare_config(self) -> None:
        for command in (("lint", "all"), ("sources", "list"), ("init",)):
            result = _run(*command, "--help")
            assert result.returncode == 0, _output(result)
            assert "--config" not in result.stdout, f"{' '.join(command)} still declares --config"


class TestScaffoldedVault:
    def test_config_check(self, vault: Path) -> None:
        result = _run("-w", str(vault), "config", "check")

        assert result.returncode == 0, _output(result)
        assert TRACEBACK not in _output(result)

    def test_status(self, vault: Path) -> None:
        result = _run("-w", str(vault), "status")

        assert result.returncode == 0, _output(result)
        assert TRACEBACK not in _output(result)

    def test_doctor_runs_without_a_traceback(self, vault: Path) -> None:
        result = _run("-w", str(vault), "doctor")

        # Findings are a non-zero exit; a crash is not.
        assert result.returncode in (0, 1), _output(result)
        assert TRACEBACK not in _output(result)

    def test_compile_without_sources_is_a_clean_error(self, vault: Path) -> None:
        result = _run("-w", str(vault), "compile")

        assert result.returncode == 1
        assert TRACEBACK not in _output(result)
        assert "No supported files found" in _output(result)


class TestLintReport:
    def test_lint_show_reads_the_newest_report(self, vault: Path) -> None:
        log_dir = vault / ".kompend" / "lint-log"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "2026-01-02T03-04-05.md").write_text("# Lint report\n\nsentinel body\n", encoding="utf-8")

        result = _run("-w", str(vault), "lint", "show")

        assert result.returncode == 0, _output(result)
        assert "sentinel body" in result.stdout

    def test_lint_show_without_a_report_is_a_clean_error(self, vault: Path) -> None:
        result = _run("-w", str(vault), "lint", "show")

        assert result.returncode == 1
        assert TRACEBACK not in _output(result)


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """A freshly scaffolded vault (no sources, no model calls)."""
    result = _run("init", str(tmp_path / "vault"))
    assert result.returncode == 0, _output(result)
    return tmp_path / "vault"


def _run(*args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "kompend", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return f"{result.stdout}\n{result.stderr}"
