"""Unit tests for knox.lint.history — persistent lint history."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from knox.lint.base import LintResult, LintRuleOutcome, Severity
from knox.lint.history import (
    LintEntry,
    entry_from_report,
    history_to_json,
    history_to_text,
    load_lint_history,
    prune_lint_history,
    save_lint_entry,
)
from knox.lint.runner import LintReport, _build_report, report_to_dict


class _StubConfig:
    """Minimal stand-in for ResolvedConfig — only .get() is needed."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


class TestSaveLoad:
    def test_save_and_load_single_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "lint_history.jsonl"
        entry = _make_entry()
        save_lint_entry(entry, path)

        loaded = load_lint_history(path)
        assert len(loaded) == 1
        assert loaded[0]["timestamp"] == entry["timestamp"]
        assert loaded[0]["duration_seconds"] == entry["duration_seconds"]
        assert loaded[0]["options"] == entry["options"]
        assert loaded[0]["report"]["health_score"] == entry["report"]["health_score"]

    def test_save_appends_multiple_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "lint_history.jsonl"
        for i in range(5):
            save_lint_entry(_make_entry(timestamp=f"2025-01-0{i + 1}T00:00:00+00:00"), path)

        loaded = load_lint_history(path)
        assert len(loaded) == 5
        assert loaded[0]["timestamp"] == "2025-01-01T00:00:00+00:00"
        assert loaded[-1]["timestamp"] == "2025-01-05T00:00:00+00:00"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "nested" / "lint_history.jsonl"
        save_lint_entry(_make_entry(), path)
        assert path.exists()

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        loaded = load_lint_history(tmp_path / "nonexistent.jsonl")
        assert loaded == []

    def test_load_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "lint_history.jsonl"
        entry = _make_entry()
        line = json.dumps(entry, default=str, ensure_ascii=False)
        path.write_text(f"\n{line}\n\n\n", encoding="utf-8")
        loaded = load_lint_history(path)
        assert len(loaded) == 1


class TestLastN:
    def test_last_n_returns_most_recent(self, tmp_path: Path) -> None:
        path = tmp_path / "lint_history.jsonl"
        for i in range(10):
            save_lint_entry(_make_entry(timestamp=f"2025-01-{i + 1:02d}T00:00:00+00:00"), path)

        loaded = load_lint_history(path, last_n=3)
        assert len(loaded) == 3
        assert loaded[0]["timestamp"] == "2025-01-08T00:00:00+00:00"
        assert loaded[-1]["timestamp"] == "2025-01-10T00:00:00+00:00"

    def test_last_n_zero_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "lint_history.jsonl"
        save_lint_entry(_make_entry(), path)
        loaded = load_lint_history(path, last_n=0)
        assert loaded == []

    def test_last_n_larger_than_count(self, tmp_path: Path) -> None:
        path = tmp_path / "lint_history.jsonl"
        save_lint_entry(_make_entry(), path)
        loaded = load_lint_history(path, last_n=100)
        assert len(loaded) == 1


class TestPrune:
    def test_prune_keep_last(self, tmp_path: Path) -> None:
        path = tmp_path / "lint_history.jsonl"
        for i in range(10):
            save_lint_entry(_make_entry(timestamp=f"2025-01-{i + 1:02d}T00:00:00+00:00"), path)

        removed = prune_lint_history(path, keep_last=3)
        assert removed == 7
        remaining = load_lint_history(path)
        assert len(remaining) == 3

    def test_prune_max_age_days(self, tmp_path: Path) -> None:
        path = tmp_path / "lint_history.jsonl"
        now = datetime.now(UTC)
        for days_ago in [100, 50, 10, 1, 0]:
            ts = (now - timedelta(days=days_ago)).isoformat(timespec="seconds")
            save_lint_entry(_make_entry(timestamp=ts), path)

        removed = prune_lint_history(path, max_age_days=30)
        assert removed == 2
        remaining = load_lint_history(path)
        assert len(remaining) == 3

    def test_prune_combines_keep_and_age(self, tmp_path: Path) -> None:
        path = tmp_path / "lint_history.jsonl"
        now = datetime.now(UTC)
        for days_ago in [100, 50, 10, 5, 1]:
            ts = (now - timedelta(days=days_ago)).isoformat(timespec="seconds")
            save_lint_entry(_make_entry(timestamp=ts), path)

        removed = prune_lint_history(path, max_age_days=30, keep_last=2)
        assert removed == 3
        remaining = load_lint_history(path)
        assert len(remaining) == 2

    def test_prune_noop_when_nothing_matches(self, tmp_path: Path) -> None:
        path = tmp_path / "lint_history.jsonl"
        save_lint_entry(_make_entry(), path)
        removed = prune_lint_history(path, keep_last=100)
        assert removed == 0
        assert len(load_lint_history(path)) == 1

    def test_prune_missing_file_returns_zero(self, tmp_path: Path) -> None:
        removed = prune_lint_history(tmp_path / "nonexistent.jsonl", keep_last=5)
        assert removed == 0

    def test_prune_requires_at_least_one_criterion(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="At least one"):
            prune_lint_history(tmp_path / "lint_history.jsonl")


class TestEntryFromReport:
    def test_builds_entry_with_config_snapshot(self) -> None:
        report = _make_report()
        entry = entry_from_report(
            report,
            duration_seconds=2.5,
            options={"rules": ["orphan-page"], "enable_llm": True, "fix": False, "threshold": 80},
            config={
                "compile.compiler": "wicer",
                "compile.llm.default_model": "openai/gpt-4o",
                "compile.llm.fallback_chain": ["anthropic/claude-3.5-sonnet"],
                "compile.llm.role_models": {"resolve": "openai/gpt-4o-mini"},
            },
            node_count=42,
            llm={"model": "openai/gpt-4o", "usage": {"prompt_tokens": 100}},
        )
        assert entry["duration_seconds"] == 2.5
        assert entry["options"]["rules"] == ["orphan-page"]
        assert entry["config_snapshot"]["compile.compiler"] == "wicer"
        assert entry["config_snapshot"]["compile.llm.default_model"] == "openai/gpt-4o"
        assert entry["wiki_meta"]["node_count"] == 42
        assert entry["llm"] == {"model": "openai/gpt-4o", "usage": {"prompt_tokens": 100}}
        assert entry["report"]["health_score"] == report.health_score
        assert "T" in entry["timestamp"]

    def test_auto_timestamp(self) -> None:
        report = _make_report()
        entry = entry_from_report(report, duration_seconds=0.1, options={}, config={}, node_count=1)
        assert "+" in entry["timestamp"]

    def test_custom_timestamp(self) -> None:
        report = _make_report()
        entry = entry_from_report(
            report,
            duration_seconds=0.1,
            options={},
            config={},
            node_count=1,
            timestamp="2020-01-01T00:00:00+00:00",
        )
        assert entry["timestamp"] == "2020-01-01T00:00:00+00:00"


class TestFormatting:
    def test_history_to_text_empty(self) -> None:
        text = history_to_text([])
        assert "No lint history" in text

    def test_history_to_text_single_entry(self) -> None:
        entry = _make_entry(score=85.0)
        text = history_to_text([entry])
        assert "1 entry" in text
        assert "85%" in text

    def test_history_to_text_multiple_entries_plural(self) -> None:
        entries = [_make_entry(score=s) for s in [90.0, 80.0, 70.0]]
        text = history_to_text(entries)
        assert "3 entries" in text

    def test_history_to_text_trend(self) -> None:
        entries = [_make_entry(score=s) for s in [50.0, 60.0, 70.0, 80.0]]
        text = history_to_text(entries, show_trend=True)
        assert "Trend" in text
        assert "Range" in text

    def test_history_to_text_trend_requires_two_entries(self) -> None:
        entry = _make_entry()
        text = history_to_text([entry], show_trend=True)
        assert "Trend" not in text

    def test_history_to_json_empty(self) -> None:
        result = history_to_json([])
        assert json.loads(result) == []

    def test_history_to_json_multiple(self) -> None:
        entries = [_make_entry(score=s) for s in [90.0, 80.0]]
        data = json.loads(history_to_json(entries))
        assert len(data) == 2
        assert data[0]["report"]["health_score"] == 90.0

    def test_history_to_text_shows_model_tag(self) -> None:
        entry = _make_entry()
        entry["llm"] = {"model": "openai/gpt-4o", "usage": {}}
        text = history_to_text([entry])
        assert "openai/gpt-4o" in text


def _make_entry(*, timestamp: str = "2025-01-01T00:00:00+00:00", score: float = 90.0) -> LintEntry:
    """Build a fully-populated ``LintEntry`` for history tests."""
    report = _make_report(health_score=score)
    return {
        "timestamp": timestamp,
        "duration_seconds": 1.5,
        "options": {"rules": None, "enable_llm": False, "fix": False, "threshold": None},
        "llm": None,
        "config_snapshot": {"compile.compiler": "graph", "compile.llm.default_model": "test-model"},
        "wiki_meta": {"node_count": 10},
        "report": report_to_dict(report),
    }


def _make_report(*, health_score: float = 90.0, results: list[LintResult] | None = None) -> LintReport:
    if results is None:
        results = [
            LintResult(rule="orphan-page", outcome=LintRuleOutcome.PASS, severity=Severity.INFO, message="ok"),
            LintResult(rule="broken-wikilinks", outcome=LintRuleOutcome.FAIL, severity=Severity.ERROR, message="bad link"),
        ]
    report = _build_report(results)
    report.health_score = health_score
    return report
