"""Tests for CLI doctor and status commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knox.cli.doctor import CheckResult, print_doctor_results, run_doctor
from knox.cli.status import collect_stats, print_status


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
        assert len(results) >= 6
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
