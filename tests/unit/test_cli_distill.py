"""Distill CLI wiring: shared helpers, the ``kompend distill`` contract, and agent fallback."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from typer.testing import CliRunner

from kompend.cli._distill_helpers import (
    build_distill_agent,
    count_pending_notes,
    distill_result_lines,
    pending_notes_message,
    run_distill,
)
from kompend.cli.app import app
from kompend.core.config import ResolvedConfig


class _SentinelAgent:
    """Stand-in for a distill agent; the helpers never call into it."""


def test_count_pending_notes_reads_both_zones_top_level_only(tmp_path: Path) -> None:
    """The count mirrors what distill reads: top-level *.md in _captured/ and _learned/."""
    sources = _make_vault(tmp_path)
    captured = sources / "_captured"
    captured.mkdir()
    (captured / "a.md").write_text("a")
    learned = sources / "_learned"
    learned.mkdir()
    (learned / "b.md").write_text("b")
    (learned / "b.txt").write_text("not a note")
    nested = learned / "nested"
    nested.mkdir()
    (nested / "c.md").write_text("not read by distill")

    assert count_pending_notes(sources) == 2


def test_count_pending_notes_without_capture_zones(tmp_path: Path) -> None:
    sources = _make_vault(tmp_path)
    assert count_pending_notes(sources) == 0


def test_pending_notes_message_names_both_zones_and_the_flag() -> None:
    message = pending_notes_message(3)
    assert "3 note(s)" in message
    assert "sources/_captured/" in message
    assert "sources/_learned/" in message
    assert "--distill" in message


def test_distill_result_lines_report_empty_corpus(tmp_path: Path) -> None:
    assert distill_result_lines([], tmp_path) == ["No captured notes found to distill."]


def test_distill_result_lines_list_pages(tmp_path: Path) -> None:
    page = tmp_path / "_distilled" / "distilled-20261019T101112-stage-3.md"
    lines = distill_result_lines([page], tmp_path)

    assert lines[0] == f"Distilled 1 page(s) to {tmp_path / '_distilled'}"
    assert lines[1] == f"  {page.name}"


def test_build_distill_agent_without_model_returns_none() -> None:
    assert build_distill_agent(_cfg(None)) is None


def test_build_distill_agent_prefers_agentic(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = _SentinelAgent()
    monkeypatch.setattr("kompend.cli._distill_helpers.make_agentic_distill_agent", lambda *a, **k: sentinel)
    monkeypatch.setattr("kompend.cli._distill_helpers.DistillAgent", lambda *a, **k: pytest.fail("fallback used"))

    assert build_distill_agent(_cfg("ollama/test-model")) is sentinel


def test_build_distill_agent_falls_back_when_agentic_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = _SentinelAgent()

    def _boom(*args: object, **kwargs: object) -> object:
        raise ValueError("no agentic support")

    monkeypatch.setattr("kompend.cli._distill_helpers.make_agentic_distill_agent", _boom)
    monkeypatch.setattr("kompend.cli._distill_helpers.DistillAgent", lambda *a, **k: sentinel)

    assert build_distill_agent(_cfg("ollama/test-model")) is sentinel


def test_build_distill_agent_returns_none_when_both_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> object:
        raise ValueError("unavailable")

    monkeypatch.setattr("kompend.cli._distill_helpers.make_agentic_distill_agent", _boom)
    monkeypatch.setattr("kompend.cli._distill_helpers.DistillAgent", _boom)

    assert build_distill_agent(_cfg("ollama/test-model")) is None


def test_run_distill_archives_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources = _make_vault(tmp_path)
    seen: dict[str, object] = {}

    def _fake_distill(sources_dir: Path, *, distill_agent: object, archive: bool) -> list[Path]:
        seen["sources_dir"] = sources_dir
        seen["archive"] = archive
        return [sources_dir / "_distilled" / "page.md"]

    monkeypatch.setattr("kompend.cli._distill_helpers.distill", _fake_distill)

    paths = run_distill(sources, _cfg(None))

    assert seen == {"sources_dir": sources, "archive": True}
    assert paths == [sources / "_distilled" / "page.md"]


def test_run_distill_honours_no_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources = _make_vault(tmp_path)
    seen: dict[str, object] = {}

    def _fake_distill(sources_dir: Path, *, distill_agent: object, archive: bool) -> list[Path]:
        seen["archive"] = archive
        return []

    monkeypatch.setattr("kompend.cli._distill_helpers.distill", _fake_distill)

    assert run_distill(sources, _cfg(None), archive=False) == []
    assert seen == {"archive": False}


def _cfg(default_model: str | None) -> ResolvedConfig:
    """Duck-typed ResolvedConfig carrying only the fields the helpers read."""
    llm = SimpleNamespace(default_model=default_model, timeout=120.0, output_retries=3)
    return cast("ResolvedConfig", SimpleNamespace(kompend=SimpleNamespace(compile=SimpleNamespace(llm=llm))))


def test_distill_cmd_prints_page_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`kompend distill` still reports the pages it wrote (now via the shared helper)."""
    sources = _make_vault(tmp_path)
    page = sources / "_distilled" / "distilled-20261019T101112-stage-3.md"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kompend.cli._distill_helpers.distill", lambda *a, **k: [page])

    result = CliRunner().invoke(app, ["distill"])

    assert result.exit_code == 0
    assert "Distilled 1 page(s) to" in result.output
    assert "_distilled" in result.output
    assert page.name in result.output


def test_distill_cmd_reports_empty_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_vault(tmp_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kompend.cli._distill_helpers.distill", lambda *a, **k: [])

    result = CliRunner().invoke(app, ["distill"])

    assert result.exit_code == 0
    assert "No captured notes found to distill." in result.output


def test_distill_cmd_no_archive_flag_reaches_the_core(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-archive is passed through to distill(archive=False)."""
    _make_vault(tmp_path)
    seen: dict[str, object] = {}

    def _fake_distill(sources_dir: Path, *, distill_agent: object, archive: bool) -> list[Path]:
        seen["archive"] = archive
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kompend.cli._distill_helpers.distill", _fake_distill)

    result = CliRunner().invoke(app, ["distill", "--no-archive"])

    assert result.exit_code == 0
    assert seen == {"archive": False}


def _make_vault(tmp_path: Path) -> Path:
    """Create a minimal vault and return its sources directory."""
    sources = tmp_path / "sources"
    sources.mkdir()
    kompend_dir = tmp_path / ".kompend"
    kompend_dir.mkdir()
    (kompend_dir / "config.toml").write_text("", encoding="utf-8")
    return sources
