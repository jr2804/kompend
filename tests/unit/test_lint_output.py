"""Tests for the lint output levels and the per-run lint artefacts."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from kompend.cli._shared import resolve_lint_output
from kompend.core.config._configs import LintConfig, LintOutput
from kompend.lint.base import LintResult, LintRuleOutcome, Severity
from kompend.lint.history import LintEntry, entry_from_report
from kompend.lint.log import (
    LINT_LOG_DIR,
    latest_lint_log,
    lint_log_dir,
    read_lint_log,
    render_report,
    write_lint_log,
)
from kompend.lint.runner import LintReport, report_to_compact, report_to_summary


class TestCompactReport:
    def test_keeps_the_verdict_first(self) -> None:
        report = _report(_fail("rumdl", "MD024: duplicate heading", "log.md"), LintResult(rule="x", outcome=LintRuleOutcome.PASS))

        lines = report_to_compact(report).splitlines()

        assert lines[0] == report_to_summary(report)
        assert "50% health" in lines[0]

    def test_one_line_per_failing_rule(self) -> None:
        report = _report(
            _fail("rumdl", "MD024: duplicate heading", "log.md"),
            _fail("rumdl", "MD013: line too long", "a.md"),
            _fail("coverage-gap", "ABLE model missing", "able.md"),
        )

        text = report_to_compact(report)

        assert text.count("rumdl:") == 1
        assert "rumdl: 2 finding(s) in 2 page(s)" in text
        assert "coverage-gap: 1 finding(s)" in text
        assert "e.g. MD024: duplicate heading" in text

    def test_caps_the_rule_list(self) -> None:
        report = _report(*(_fail(f"rule-{i}", f"message {i}") for i in range(9)))

        text = report_to_compact(report, max_rules=3)

        assert "... 6 more rule(s) failing" in text
        assert "rule-3" not in text

    def test_passes_are_only_counted(self) -> None:
        report = _report(LintResult(rule="broken-wikilinks", outcome=LintRuleOutcome.PASS, message="ok"))

        text = report_to_compact(report)

        assert "broken-wikilinks" not in text
        assert "1 pass" in text


class TestLintLogArtefact:
    def test_writes_markdown_and_json(self, tmp_path: Path) -> None:
        entry = self._entry(_report(_fail("rumdl", "MD024", "log.md")))

        paths = write_lint_log(entry, tmp_path)

        assert paths.markdown.parent == tmp_path / ".kompend" / LINT_LOG_DIR
        assert paths.markdown.exists()
        assert paths.payload.exists()
        body = paths.markdown.read_text(encoding="utf-8")
        assert "# Lint report" in body
        assert "- Run:" in body
        assert "MD024" in body

    def test_json_mirrors_the_history_entry(self, tmp_path: Path) -> None:
        entry = self._entry(_report(_fail("rumdl", "MD024", "log.md")))

        paths = write_lint_log(entry, tmp_path)

        payload = read_lint_log(paths.payload)
        assert payload["timestamp"] == entry["timestamp"]
        assert payload["report"]["health_score"] == 0.0

    def test_latest_returns_the_newest_report(self, tmp_path: Path) -> None:
        first = self._entry(_report(_fail("a", "one")))
        first["timestamp"] = "2026-10-24T01:00:00+02:00"
        second = self._entry(_report(_fail("b", "two")))
        second["timestamp"] = "2026-10-24T02:00:00+02:00"

        write_lint_log(first, tmp_path)
        write_lint_log(second, tmp_path)

        latest = latest_lint_log(tmp_path)
        assert latest is not None
        assert latest.name == "2026-10-24T02-00-00.md"

    def test_no_report_yet(self, tmp_path: Path) -> None:
        assert latest_lint_log(tmp_path) is None
        assert lint_log_dir(tmp_path) == tmp_path / ".kompend" / LINT_LOG_DIR

    def _entry(self, report: LintReport) -> LintEntry:
        return entry_from_report(
            report,
            duration_seconds=1.5,
            options={"context": "pipeline"},
            config={},
            node_count=12,
        )


class TestRenderReport:
    @pytest.mark.parametrize("level", [LintOutput.QUIET, LintOutput.SUMMARY, LintOutput.FULL])
    def test_every_level_points_at_the_report(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], level: LintOutput) -> None:
        entry = entry_from_report(
            _report(_fail("rumdl", "MD024", "log.md")),
            duration_seconds=1.0,
            options={"context": "pipeline"},
            config={},
            node_count=3,
        )
        paths = write_lint_log(entry, tmp_path)
        report = _report(_fail("rumdl", "MD024", "log.md"))

        render_report(report, level, paths)

        out = capsys.readouterr().out.replace("\n", "")
        assert "Full report:" in out
        assert paths.markdown.name in out
        if level is LintOutput.QUIET:
            assert "MD024" not in out
        else:
            assert "MD024" in out


class TestResolveLintOutput:
    def test_config_wins_without_flags(self) -> None:
        assert resolve_lint_output(LintOutput.SUMMARY, verbose=False, quiet=False) is LintOutput.SUMMARY

    def test_verbose_overrides(self) -> None:
        assert resolve_lint_output(LintOutput.QUIET, verbose=True, quiet=False) is LintOutput.FULL

    def test_quiet_overrides(self) -> None:
        assert resolve_lint_output(LintOutput.FULL, verbose=False, quiet=True) is LintOutput.QUIET

    def test_both_flags_is_an_error(self) -> None:
        with pytest.raises(typer.BadParameter):
            resolve_lint_output(LintOutput.SUMMARY, verbose=True, quiet=True)


class TestLintOutputConfig:
    def test_quiet_is_the_default(self) -> None:
        assert LintConfig().output is LintOutput.QUIET

    def test_levels_are_strenums(self) -> None:
        assert [level.value for level in LintOutput] == ["quiet", "summary", "full"]


def _report(*results: LintResult) -> LintReport:
    failures = sum(1 for r in results if r.outcome == LintRuleOutcome.FAIL)
    passes = sum(1 for r in results if r.outcome == LintRuleOutcome.PASS)
    skips = sum(1 for r in results if r.outcome == LintRuleOutcome.SKIP)
    total = len(results)
    return LintReport(
        results=list(results),
        total_checks=total,
        failures=failures,
        passes=passes,
        skips=skips,
        health_score=round(100.0 * passes / total, 1) if total else 0.0,
    )


def _fail(rule: str, message: str, page: str | None = None) -> LintResult:
    return LintResult(rule=rule, outcome=LintRuleOutcome.FAIL, severity=Severity.WARNING, message=message, page_id=page)
