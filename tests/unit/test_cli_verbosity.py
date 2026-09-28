"""Tests for the global CLI verbosity toggle (quiet by default).

``--verbose`` / ``KNOX_VERBOSE`` controls traceback visibility for both the
recoverable path (a failed parse logs a one-line warning and continues) and the
fatal path (``main()`` prints one line instead of a traceback).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from knox.cli import app as app_module
from knox.cli._shared import format_exception_for_cli
from knox.cli.app import app
from knox.core.pipeline.export import _load_asset_store
from knox.core.pipeline.ingest import ingest_documents
from knox.core.pipeline.runtime import PipelineRuntime
from knox.core.pipeline.stages._parse import _parse_source
from knox.core.verbosity import is_verbose, set_verbose
from knox.preprocessing.parsers.base import ParserRouter

_cli_runner = CliRunner()


# ── recoverable parse failure ────────────────────────────────────────────


class _FailingRouter:
    def fingerprint(self) -> str:
        return "failing-router"

    def route(self, source_file: Path) -> MagicMock:  # noqa: ARG002
        parser = MagicMock()
        parser.parse.side_effect = ValueError("bad parse")
        return parser


# ── default one-liners carry the cause (TASK-48) ─────────────────────────
#
# Suppressing a traceback is only safe if the one-line message names the cause.
# These fail against the pre-TASK-48 messages ("Failed to parse <file>" with no
# exception text).


class _RaisingParser:
    def parse(self, target: Path) -> Any:  # noqa: ARG002
        raise ValueError("docling exploded")


class _Router:
    def route(self, source_file: Path) -> _RaisingParser:  # noqa: ARG002
        return _RaisingParser()


class _Pipeline:
    router = _Router()

    def cache_directory(self) -> Path:
        return Path("/nonexistent-cache")  # noqa: S108 — never touched: parse raises first


@pytest.fixture(autouse=True)
def _reset_verbose() -> Any:
    set_verbose(False)
    yield
    set_verbose(False)


# ── format_exception_for_cli ─────────────────────────────────────────────


def test_format_exception_default_is_one_line() -> None:
    try:
        raise ValueError("boom")
    except ValueError as exc:
        text = format_exception_for_cli(exc, verbose=False)
    assert text == "error: ValueError: boom"
    assert "Traceback" not in text


def test_format_exception_verbose_is_full_traceback() -> None:
    try:
        raise ValueError("boom")
    except ValueError as exc:
        text = format_exception_for_cli(exc, verbose=True)
    assert "Traceback" in text
    assert "ValueError: boom" in text


# ── flag + env var ───────────────────────────────────────────────────────


def test_verbose_flag_accepted() -> None:
    """Fails on the old app (no ``--verbose``): exit 2, 'No such option'."""
    result = _cli_runner.invoke(app, ["--verbose", "--version"])
    assert result.exit_code == 0, result.output
    assert "No such option" not in result.output


def test_verbose_env_var_enables_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOX_VERBOSE", "1")
    result = _cli_runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert is_verbose() is True


def test_verbose_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KNOX_VERBOSE", raising=False)
    result = _cli_runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert is_verbose() is False


def test_failed_parse_logs_one_line_and_continues(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = _seed_vault(tmp_path)
    with caplog.at_level(logging.WARNING):
        result = _ingest_failing_parse(base)

    # Recoverable path continues.
    assert result.failed == 1

    records = [r for r in caplog.records if r.message.startswith("Failed to parse")]
    assert records, "expected a one-line 'Failed to parse' warning"
    # No traceback attached by default (logging stores the falsy exc_info=False).
    assert not records[0].exc_info


def test_failed_parse_attaches_traceback_when_verbose(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = _seed_vault(tmp_path)
    set_verbose(True)
    with caplog.at_level(logging.WARNING):
        result = _ingest_failing_parse(base)

    assert result.failed == 1
    records = [r for r in caplog.records if r.message.startswith("Failed to parse")]
    assert records
    assert records[0].exc_info  # a truthy (type, exc, tb) tuple


def _seed_vault(tmp_path: Path) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".knox").mkdir(parents=True)
    (base / ".knox" / "config.toml").write_text("", encoding="utf-8")
    return base


def _ingest_failing_parse(base: Path) -> Any:
    src = base / "sources" / "doc.pdf"
    src.write_bytes(b"%PDF fake")
    return ingest_documents(files=[src], base_dir=base, router=cast("ParserRouter", _FailingRouter()), no_cache=True, config=None)


def test_main_fatal_is_one_line_without_verbose(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("KNOX_LOGFIRE_DISABLED", "1")
    monkeypatch.setattr(app_module, "app", _raise_runtime_error)
    with pytest.raises(SystemExit) as exc_info:
        app_module.main()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "error: RuntimeError: boom" in err
    assert "Traceback" not in err


def test_main_fatal_is_traceback_with_verbose(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("KNOX_LOGFIRE_DISABLED", "1")
    monkeypatch.setattr(app_module, "app", _raise_runtime_error)
    set_verbose(True)
    with pytest.raises(SystemExit) as exc_info:
        app_module.main()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Traceback" in err
    assert "RuntimeError: boom" in err


def test_parse_source_default_message_carries_cause(caplog: pytest.LogCaptureFixture) -> None:
    set_verbose(False)
    with caplog.at_level(logging.ERROR):
        out = _parse_source(cast("PipelineRuntime", _Pipeline()), Path("sources/foo.pdf"), use_cache=False, content_hash="h")
    assert out is None
    records = [r for r in caplog.records if "Failed to parse" in r.getMessage()]
    assert records
    assert "docling exploded" in records[0].getMessage()
    assert not records[0].exc_info


def test_load_asset_store_default_message_carries_cause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("asset store kaput")

    monkeypatch.setattr("knox.core.pipeline.export.AssetStore", _boom)
    set_verbose(False)
    with caplog.at_level(logging.DEBUG):
        out = _load_asset_store(tmp_path)
    assert out is None
    records = [r for r in caplog.records if "Asset store unavailable" in r.getMessage()]
    assert records
    assert "asset store kaput" in records[0].getMessage()


# ── fatal path ───────────────────────────────────────────────────────────


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")
