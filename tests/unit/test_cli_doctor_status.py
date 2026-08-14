"""Tests for CLI doctor and status commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from knox.cli.app import app
from knox.cli.cmds.pipeline.status import collect_stats, print_status
from knox.cli.cmds.quality.doctor import CheckResult, print_doctor_results, run_doctor
from knox.core.paths import COMPILE_STATE_FILE
from knox.core.pipeline.state import SourceState, TierState, TierStateStore


class TestCheckResult:
    def test_to_dict(self) -> None:
        result = CheckResult("test-check", True, "All good")
        d = result.to_dict()
        assert d["name"] == "test-check"
        assert d["passed"] is True
        assert d["message"] == "All good"

    def test_to_dict_failed(self) -> None:
        result = CheckResult("failing-check", False, "Something broke")
        d = result.to_dict()
        assert d["passed"] is False


class TestPrintDoctorResults:
    def test_all_passed_exit_code(self) -> None:
        results = [
            CheckResult("a", True, "OK"),
            CheckResult("b", True, "OK"),
        ]
        code = print_doctor_results(results)
        assert code == 0

    def test_some_failed_exit_code(self) -> None:
        results = [
            CheckResult("a", True, "OK"),
            CheckResult("b", False, "FAIL"),
        ]
        code = print_doctor_results(results)
        assert code == 1

    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [
            CheckResult("a", True, "OK"),
            CheckResult("b", False, "FAIL"),
        ]
        print_doctor_results(results, json_output=True)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["passed"] == 1
        assert data["failed"] == 1
        assert data["total"] == 2


class TestRunDoctor:
    def test_basic_run(self) -> None:
        """Doctor runs without crashing, returns CheckResult list."""
        results = run_doctor()
        assert isinstance(results, list)
        assert len(results) >= 5
        assert all(isinstance(r, CheckResult) for r in results)

    def test_with_paths(self, tmp_path: Path) -> None:
        """Doctor accepts explicit paths."""
        results = run_doctor(
            sources_dir=tmp_path,
            output_dir=tmp_path,
        )
        # sources-directory and output-directory checks should pass
        src_check = next(r for r in results if r.name == "sources-directory")
        out_check = next(r for r in results if r.name == "output-directory")
        assert src_check.passed is True
        assert out_check.passed is True

    def test_no_model_fails_llm(self) -> None:
        """Without a model, LLM connectivity check is a hard failure."""
        results = run_doctor()
        llm_check = next(r for r in results if r.name == "connectivity-llm")
        assert llm_check.passed is False
        assert "skipped" in llm_check.message

    def test_vlm_skipped_without_model(self) -> None:
        """Without a VLM model, VLM check is a warning, not a pass."""
        results = run_doctor()
        vlm_check = next(r for r in results if r.name == "connectivity-vlm")
        assert vlm_check.passed is True
        assert vlm_check.warning is True
        assert "skipped" in vlm_check.message

    def test_embed_skipped_without_model(self) -> None:
        """Without an embed model, embed check is a warning, not a pass."""
        results = run_doctor()
        embed_check = next(r for r in results if r.name == "connectivity-embed")
        assert embed_check.passed is True
        assert embed_check.warning is True
        assert "skipped" in embed_check.message


class TestDoctorWikiContext:
    """CLI-level tests for wiki-context enforcement."""

    def test_fails_without_wiki_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Doctor exits 1 when no --wiki-dir and no output dir exists."""
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1
        assert "No wiki found" in result.output

    def test_works_with_explicit_wiki_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Doctor runs (bypasses wiki-context gate) when --wiki-dir is set."""
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["--wiki-dir", str(tmp_path), "doctor"])
        assert "No wiki found" not in result.output
        assert "checks passed" in result.output


class TestCollectStats:
    def test_no_dirs(self) -> None:
        stats = collect_stats()
        assert stats["sources"]["count"] == 0
        assert stats["articles"]["count"] == 0
        assert stats["last_compile"] is None

    def test_with_source_files(self, tmp_path: Path) -> None:
        """Source count reflects files in the sources dir."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("test content")
        stats = collect_stats(sources_dir=tmp_path)
        assert stats["sources"]["count"] == 1
        assert stats["sources"]["total_size_bytes"] > 0

    def test_with_output_files(self, tmp_path: Path) -> None:
        """Article count from wiki.md."""
        wiki = tmp_path / "wiki.md"
        wiki.write_text("# Title\n\nContent here.\n\n# Another\n\nMore words.")
        stats = collect_stats(output_dir=tmp_path)
        assert stats["articles"]["total_words"] > 0

    def test_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        stats = collect_stats(sources_dir=tmp_path)
        print_status(stats, json_output=True)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["wiki_name"] == "KNOX Wiki"

    def test_compile_status_not_initialized(self, tmp_path: Path) -> None:
        """No state_dir → not-initialized."""
        stats = collect_stats(sources_dir=tmp_path, output_dir=tmp_path, state_dir=None)
        assert stats["compile_status"]["status"] == "not-initialized"

    def test_compile_status_no_sources(self, tmp_path: Path) -> None:
        """state_dir exists but no sources → no-sources."""
        state_dir = tmp_path / ".knox"
        state_dir.mkdir()
        stats = collect_stats(state_dir=state_dir)
        assert stats["compile_status"]["status"] == "no-sources"

    def test_compile_status_not_indexed(self, tmp_path: Path) -> None:
        """Sources exist but no tier state file → not-indexed."""
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "doc.pdf").write_text("x")
        state_dir = tmp_path / ".knox"
        state_dir.mkdir()
        stats = collect_stats(sources_dir=sources, state_dir=state_dir)
        assert stats["compile_status"]["status"] == "not-indexed"
        assert stats["compile_status"]["total"] == 1

    def test_compile_status_in_progress(self, tmp_path: Path) -> None:
        """Tier state with entries at tier < 4 → in-progress."""
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "doc.pdf").write_text("x")
        state_dir = tmp_path / ".knox"
        state_dir.mkdir()
        state = TierState(
            entries={
                "doc": SourceState(stem="doc", source_file="doc.pdf", content_hash="h", current_tier=1),
            }
        )
        TierStateStore(state_dir / COMPILE_STATE_FILE).save(state)
        stats = collect_stats(sources_dir=sources, state_dir=state_dir)
        assert stats["compile_status"]["status"] == "in-progress"
        assert stats["tier_distribution"] == {1: 1}

    def test_compile_status_compiled(self, tmp_path: Path) -> None:
        """All entries at tier 4 + wiki exists → compiled."""
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "doc.pdf").write_text("x")
        output = tmp_path / "wiki"
        output.mkdir()
        (output / "wiki.md").write_text("# Wiki\n\nContent.")
        state_dir = tmp_path / ".knox"
        state_dir.mkdir()
        state = TierState(
            entries={
                "doc": SourceState(stem="doc", source_file="doc.pdf", content_hash="h", current_tier=4),
            }
        )
        TierStateStore(state_dir / COMPILE_STATE_FILE).save(state)
        stats = collect_stats(sources_dir=sources, output_dir=output, state_dir=state_dir)
        assert stats["compile_status"]["status"] == "compiled"
        assert stats["compile_status"]["compiled"] == 1

    def test_compile_status_broken(self, tmp_path: Path) -> None:
        """Corrupt tier state file → broken."""
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "doc.pdf").write_text("x")
        state_dir = tmp_path / ".knox"
        state_dir.mkdir()
        (state_dir / COMPILE_STATE_FILE).write_text("not json {{{", encoding="utf-8")
        stats = collect_stats(sources_dir=sources, state_dir=state_dir)
        assert stats["compile_status"]["status"] == "broken"
        assert stats["compile_status"]["error"]


def test_trace_no_wiki() -> None:
    """Knox trace without compiled wiki should exit with error."""
    runner = CliRunner()
    result = runner.invoke(app, ["trace", "test-page"])
    assert result.exit_code != 0
    assert result.exit_code == 1


def test_trace_multiple_slugs_accepted() -> None:
    """Knox trace accepts multiple positional slugs (variadic)."""
    runner = CliRunner()
    # No wiki loaded, but Typer should accept the args without a usage error.
    result = runner.invoke(app, ["trace", "page-a", "page-b", "page-c"])
    assert result.exit_code != 0  # no wiki → error, but NOT a usage/arg error
    assert "Got unexpected extra argument" not in result.output


def test_write_command_minimal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Knox write creates a file (or errors gracefully)."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["write", "Test Article", "--frontmatter", "lang=en"])
    assert result.exit_code in (0, 1)
    if result.exit_code == 0:
        assert "Written" in result.stdout
        result_file = tmp_path / "wiki" / "write" / "test-article.md"
        assert result_file.exists()
        content = result_file.read_text(encoding="utf-8")
        assert "lang: en" in content
