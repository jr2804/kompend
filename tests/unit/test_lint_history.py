"""Unit tests for kompend.lint.history — persistent lint history."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import LINT_RUNS_TABLE
from kompend.lint.base import LintResult, LintRuleOutcome, Severity
from kompend.lint.history import (
    LintConfigSnapshot,
    LintEntry,
    LintLlmMeta,
    LintRunOptions,
    LintWikiMeta,
    entry_from_report,
    history_to_json,
    history_to_text,
    load_lint_history,
    prune_lint_history,
    save_lint_entry,
)
from kompend.lint.runner import LintReport, _build_report, report_to_payload
from kompend.llm.usage import UsageSnapshot


class TestSaveLoad:
    def test_save_and_load_single_entry(self, tmp_path: Path) -> None:
        path = _hist(tmp_path)
        entry = _make_entry()
        save_lint_entry(entry, path)

        loaded = load_lint_history(path)
        assert len(loaded) == 1
        assert loaded[0].timestamp == entry.timestamp
        assert loaded[0].duration_seconds == entry.duration_seconds
        assert loaded[0].options == entry.options
        assert loaded[0].report.health_score == entry.report.health_score

    def test_save_appends_multiple_entries(self, tmp_path: Path) -> None:
        path = _hist(tmp_path)
        for i in range(5):
            save_lint_entry(_make_entry(timestamp=f"2025-01-0{i + 1}T00:00:00+00:00"), path)

        loaded = load_lint_history(path)
        assert len(loaded) == 5
        assert loaded[0].timestamp == "2025-01-01T00:00:00+00:00"
        assert loaded[-1].timestamp == "2025-01-05T00:00:00+00:00"

    def test_save_creates_store(self, tmp_path: Path) -> None:
        save_lint_entry(_make_entry(), _hist(tmp_path))
        assert (tmp_path / ".kompend" / "kompend.db").exists()

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        loaded = load_lint_history(tmp_path / "nonexistent.jsonl")
        assert loaded == []

    def test_load_empty_store_returns_empty(self, tmp_path: Path) -> None:
        assert load_lint_history(_hist(tmp_path)) == []

    def test_loads_pre_migration_row(self, tmp_path: Path) -> None:
        """A row written by the TypedDict era loads unchanged into the models."""
        path = _hist(tmp_path)
        legacy_report = {
            "health_score": 77.0,
            "total_checks": 2,
            "failures": 1,
            "passes": 1,
            "skips": 0,
            "results": [
                {
                    "rule": "broken-wikilinks",
                    "outcome": "fail",
                    "severity": "error",
                    "message": "bad link",
                    "page_id": "n1",
                    "fix_applied": False,
                    "details": {"target": "x"},
                }
            ],
        }
        with open_store(vault_db_path(path)) as conn:
            conn.execute(
                f"INSERT INTO {LINT_RUNS_TABLE} "  # noqa: S608 — table name is a Final[str] constant
                "(timestamp, duration_seconds, options, llm, config_snapshot, wiki_meta, report) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "2025-01-01T00:00:00+00:00",
                    1.5,
                    json.dumps({"context": "check", "fix": False}),
                    None,
                    json.dumps({"compile.compiler": "graph", "compile.llm.default_model": "test-model", "compile.llm.fallback_chain": []}),
                    json.dumps({"node_count": 10}),
                    json.dumps(legacy_report),
                ),
            )

        loaded = load_lint_history(path)
        assert len(loaded) == 1
        assert loaded[0].options == LintRunOptions(context="check", fix=False)
        assert loaded[0].config_snapshot == LintConfigSnapshot(
            compiler="graph",
            default_model="test-model",
            fallback_chain=[],
        )
        assert loaded[0].report.health_score == 77.0
        assert loaded[0].report.results[0].severity == "error"
        assert loaded[0].report.results[0].details == {"target": "x"}

    def test_to_columns_keep_the_stored_shape(self, tmp_path: Path) -> None:
        """The model serialises to exactly the seven legacy column values."""
        entry = _make_entry()
        columns = entry.to_columns()
        assert len(columns) == 7
        assert json.loads(str(columns[2])) == {"context": "check", "fix": False}
        assert json.loads(str(columns[4])) == {
            "compile.compiler": "graph",
            "compile.llm.default_model": "test-model",
            "compile.llm.fallback_chain": [],
        }
        assert json.loads(str(columns[5])) == {"node_count": 10}
        assert json.loads(str(columns[6]))["health_score"] == 90.0


class TestLastN:
    def test_last_n_returns_most_recent(self, tmp_path: Path) -> None:
        path = _hist(tmp_path)
        for i in range(10):
            save_lint_entry(_make_entry(timestamp=f"2025-01-{i + 1:02d}T00:00:00+00:00"), path)

        loaded = load_lint_history(path, last_n=3)
        assert len(loaded) == 3
        assert loaded[0].timestamp == "2025-01-08T00:00:00+00:00"
        assert loaded[-1].timestamp == "2025-01-10T00:00:00+00:00"

    def test_last_n_zero_returns_empty(self, tmp_path: Path) -> None:
        path = _hist(tmp_path)
        save_lint_entry(_make_entry(), path)
        loaded = load_lint_history(path, last_n=0)
        assert loaded == []

    def test_last_n_larger_than_count(self, tmp_path: Path) -> None:
        path = _hist(tmp_path)
        save_lint_entry(_make_entry(), path)
        loaded = load_lint_history(path, last_n=100)
        assert len(loaded) == 1


class TestPrune:
    def test_prune_keep_last(self, tmp_path: Path) -> None:
        path = _hist(tmp_path)
        for i in range(10):
            save_lint_entry(_make_entry(timestamp=f"2025-01-{i + 1:02d}T00:00:00+00:00"), path)

        removed = prune_lint_history(path, keep_last=3)
        assert removed == 7
        remaining = load_lint_history(path)
        assert len(remaining) == 3

    def test_prune_max_age_days(self, tmp_path: Path) -> None:
        path = _hist(tmp_path)
        now = datetime.now(UTC)
        for days_ago in [100, 50, 10, 1, 0]:
            ts = (now - timedelta(days=days_ago)).isoformat(timespec="seconds")
            save_lint_entry(_make_entry(timestamp=ts), path)

        removed = prune_lint_history(path, max_age_days=30)
        assert removed == 2
        remaining = load_lint_history(path)
        assert len(remaining) == 3

    def test_prune_combines_keep_and_age(self, tmp_path: Path) -> None:
        path = _hist(tmp_path)
        now = datetime.now(UTC)
        for days_ago in [100, 50, 10, 5, 1]:
            ts = (now - timedelta(days=days_ago)).isoformat(timespec="seconds")
            save_lint_entry(_make_entry(timestamp=ts), path)

        removed = prune_lint_history(path, max_age_days=30, keep_last=2)
        assert removed == 3
        remaining = load_lint_history(path)
        assert len(remaining) == 2

    def test_prune_noop_when_nothing_matches(self, tmp_path: Path) -> None:
        path = _hist(tmp_path)
        save_lint_entry(_make_entry(), path)
        removed = prune_lint_history(path, keep_last=100)
        assert removed == 0
        assert len(load_lint_history(path)) == 1

    def test_prune_missing_file_returns_zero(self, tmp_path: Path) -> None:
        removed = prune_lint_history(tmp_path / "nonexistent.jsonl", keep_last=5)
        assert removed == 0

    def test_prune_requires_at_least_one_criterion(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="At least one"):
            prune_lint_history(_hist(tmp_path))


class TestEntryFromReport:
    def test_builds_entry_with_config_snapshot(self) -> None:
        report = _make_report()
        entry = entry_from_report(
            report,
            duration_seconds=2.5,
            options=LintRunOptions(context="check", fix=False),
            config={
                "compile.compiler": "wicer",
                "compile.llm.default_model": "openai/gpt-4o",
                "compile.llm.fallback_chain": ["anthropic/claude-3.5-sonnet"],
            },
            node_count=42,
            llm=LintLlmMeta(model="openai/gpt-4o", usage=_usage()),
        )
        assert entry.duration_seconds == 2.5
        assert entry.options == LintRunOptions(context="check", fix=False)
        assert entry.config_snapshot.compiler == "wicer"
        assert entry.config_snapshot.default_model == "openai/gpt-4o"
        assert entry.wiki_meta.node_count == 42
        assert entry.llm == LintLlmMeta(model="openai/gpt-4o", usage=_usage())
        assert entry.report.health_score == report.health_score
        assert "T" in entry.timestamp

    def test_auto_timestamp(self) -> None:
        report = _make_report()
        entry = entry_from_report(
            report,
            duration_seconds=0.1,
            options=LintRunOptions(context="check", fix=False),
            config={},
            node_count=1,
        )
        assert "+" in entry.timestamp

    def test_custom_timestamp(self) -> None:
        report = _make_report()
        entry = entry_from_report(
            report,
            duration_seconds=0.1,
            options=LintRunOptions(context="check", fix=False),
            config={},
            node_count=1,
            timestamp="2020-01-01T00:00:00+00:00",
        )
        assert entry.timestamp == "2020-01-01T00:00:00+00:00"

    def test_from_row_rejects_wrong_shape(self) -> None:
        """A corrupt row fails at the boundary, not downstream with a KeyError."""
        row = {
            "timestamp": "2020-01-01T00:00:00+00:00",
            "duration_seconds": 1.0,
            "options": '"just a string, not an object"',
            "llm": None,
            "config_snapshot": "{}",
            "wiki_meta": '{"node_count": 1}',
            "report": '{"health_score": 1.0, "total_checks": 0, "failures": 0, "passes": 0, "skips": 0, "results": []}',
        }
        with pytest.raises(ValidationError):
            LintEntry.from_row(row)


class TestDetailsRoundTrip:
    """The ``details`` payload keeps its documented keys across the history."""

    def test_the_stored_details_keep_exactly_the_documented_keys(self, tmp_path: Path) -> None:
        path = _hist(tmp_path)
        report = _build_report(
            [
                LintResult(
                    rule="unreferenced-attachments",
                    outcome=LintRuleOutcome.FAIL,
                    severity=Severity.WARNING,
                    message="Unreferenced attachment 'a/b.png'",
                    details={"path": "attachments/a/b.png"},
                ),
                LintResult(
                    rule="unreferenced-attachments",
                    outcome=LintRuleOutcome.FAIL,
                    severity=Severity.WARNING,
                    message="The scan could not read a page",
                    details={"unreadable_pages": ["sources/x.md"]},
                ),
                LintResult(
                    rule="orphan-attachments",
                    outcome=LintRuleOutcome.PASS,
                    severity=Severity.INFO,
                    message="Kept the linked files",
                    details={"kept_linked": 3},
                ),
                LintResult(
                    rule="unreferenced-attachments",
                    outcome=LintRuleOutcome.FAIL,
                    severity=Severity.WARNING,
                    message="Could not delete the attachment",
                    details={"path": "attachments/a/c.png", "error": "locked"},
                ),
            ]
        )
        entry = entry_from_report(
            report,
            duration_seconds=1.0,
            options=LintRunOptions(context="check", fix=False),
            config={},
            node_count=1,
        )
        save_lint_entry(entry, path)

        expected = [
            {"path": "attachments/a/b.png"},
            {"unreadable_pages": ["sources/x.md"]},
            {"kept_linked": 3},
            {"path": "attachments/a/c.png", "error": "locked"},
        ]
        loaded = load_lint_history(path)
        assert [r.details for r in loaded[0].report.results] == expected

        with open_store(vault_db_path(path)) as conn:
            row = conn.execute(f"SELECT report FROM {LINT_RUNS_TABLE}").fetchone()  # noqa: S608 — table name is a Final[str] constant
        stored = json.loads(row["report"])
        assert [r["details"] for r in stored["results"]] == expected

    def test_a_legacy_row_with_an_unknown_key_loads_unchanged(self, tmp_path: Path) -> None:
        """The reader drops nothing: an old record keeps every key it carries."""
        path = _hist(tmp_path)
        legacy_report = {
            "health_score": 77.0,
            "total_checks": 1,
            "failures": 1,
            "passes": 0,
            "skips": 0,
            "results": [
                {
                    "rule": "unreferenced-attachments",
                    "outcome": "fail",
                    "severity": "warning",
                    "message": "Unreferenced attachment",
                    "page_id": "",
                    "fix_applied": False,
                    "details": {"path": "attachments/a/b.png", "legacy_note": "from an older rule"},
                }
            ],
        }
        with open_store(vault_db_path(path)) as conn:
            conn.execute(
                f"INSERT INTO {LINT_RUNS_TABLE} "  # noqa: S608 — table name is a Final[str] constant
                "(timestamp, duration_seconds, options, llm, config_snapshot, wiki_meta, report) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "2025-01-01T00:00:00+00:00",
                    1.0,
                    json.dumps({"context": "check", "fix": False}),
                    None,
                    json.dumps({"compile.compiler": "graph", "compile.llm.default_model": "test-model", "compile.llm.fallback_chain": []}),
                    json.dumps({"node_count": 1}),
                    json.dumps(legacy_report),
                ),
            )

        loaded = load_lint_history(path)
        assert loaded[0].report.results[0].details == {
            "path": "attachments/a/b.png",
            "legacy_note": "from an older rule",
        }


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
        entry.llm = LintLlmMeta(model="openai/gpt-4o", usage=_usage())
        text = history_to_text([entry])
        assert "openai/gpt-4o" in text


def _hist(tmp_path: Path) -> Path:
    """Return a ``.kompend/lint_history.jsonl`` path the store resolver understands.

    The path only locates the vault's ``kompend.db``; lint history now lives in
    the ``lint_runs`` table, not in a file.
    """
    return tmp_path / ".kompend" / "lint_history.jsonl"


def _make_entry(*, timestamp: str = "2025-01-01T00:00:00+00:00", score: float = 90.0) -> LintEntry:
    """Build a fully-populated ``LintEntry`` for history tests."""
    report = _make_report(health_score=score)
    return LintEntry(
        timestamp=timestamp,
        duration_seconds=1.5,
        options=LintRunOptions(context="check", fix=False),
        llm=None,
        config_snapshot=LintConfigSnapshot(
            compiler="graph",
            default_model="test-model",
            fallback_chain=[],
        ),
        wiki_meta=LintWikiMeta(node_count=10),
        report=report_to_payload(report),
    )


def _usage() -> UsageSnapshot:
    """A fully-populated usage snapshot in the :class:`UsageTracker` wire shape."""
    return UsageSnapshot(
        prompt_tokens=100,
        completion_tokens=50,
        call_count=1,
        avg_ttft=0.1,
        overall_tok_s=20.0,
        recent_tok_s=20.0,
        per_model={"openai/gpt-4o": {"prompt": 100, "completion": 50, "calls": 1}},
        context_tokens=1000,
        context_budget=128000,
    )


def _make_report(*, health_score: float = 90.0, results: list[LintResult] | None = None) -> LintReport:
    if results is None:
        results = [
            LintResult(rule="orphan-page", outcome=LintRuleOutcome.PASS, severity=Severity.INFO, message="ok"),
            LintResult(rule="broken-wikilinks", outcome=LintRuleOutcome.FAIL, severity=Severity.ERROR, message="bad link"),
        ]
    report = _build_report(results)
    report.health_score = health_score
    return report
