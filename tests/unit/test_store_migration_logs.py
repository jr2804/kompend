"""Round-trip tests for the ``.kompend`` observability/audit artifacts now stored
in ``kompend.db``: ``runs``, ``lint_runs``, ``verification``, ``private_pages``.
"""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants.

from __future__ import annotations

import json
from pathlib import Path

from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.reporting.runlog import log_run_end, log_run_start
from kompend.core.store import open_store
from kompend.core.store.schema import PRIVATE_PAGES_TABLE, RUNS_TABLE, VERIFICATION_TABLE
from kompend.lint.history import LintEntry, load_lint_history, save_lint_entry
from kompend.trust import mark_wiki_private
from kompend.verification import ClaimVerification, VerificationResult, _persist_verification


class TestRunsRoundTrip:
    def test_run_handle_round_trip(self, tmp_path: Path) -> None:
        vault = _vault(tmp_path)
        handle = log_run_start(wiki_dir=str(vault), cmd_tokens=["compile", "src"])
        assert handle is not None
        log_run_end(handle, status="ok", exit_code=0)

        with open_store(vault / ".kompend" / "kompend.db") as conn:
            rows = [dict(r) for r in conn.execute(f"SELECT event, run_id, cmd, status, exit FROM {RUNS_TABLE} ORDER BY id")]
        assert [r["event"] for r in rows] == ["start", "end"]
        assert rows[0]["run_id"] == rows[1]["run_id"] == handle.run_id
        assert rows[1]["status"] == "ok"
        assert rows[1]["exit"] == 0

    def test_runs_append_in_order(self, tmp_path: Path) -> None:
        vault = _vault(tmp_path)
        for i in range(3):
            h = log_run_start(wiki_dir=str(vault), cmd_tokens=[f"cmd{i}"])
            log_run_end(h, status="ok")
        with open_store(vault / ".kompend" / "kompend.db") as conn:
            cmds = [r["cmd"] for r in conn.execute(f"SELECT cmd FROM {RUNS_TABLE} WHERE event = 'end' ORDER BY id")]
        assert cmds == ["cmd0", "cmd1", "cmd2"]


class TestLintRunsRoundTrip:
    def test_append_preserves_order(self, tmp_path: Path) -> None:
        path = tmp_path / ".kompend" / "lint_history.jsonl"
        for i in range(3):
            save_lint_entry(_lint_entry(f"2025-01-0{i + 1}T00:00:00+00:00", 40.0 + i), path)
        loaded = load_lint_history(path)
        assert [e["report"]["health_score"] for e in loaded] == [40.0, 41.0, 42.0]
        assert [e["timestamp"] for e in loaded] == sorted(e["timestamp"] for e in loaded)


class TestVerificationRoundTrip:
    def test_summary_counts_and_latest_wins(self, tmp_path: Path) -> None:
        _persist_verification(tmp_path, self._result(3, 2))
        _persist_verification(tmp_path, self._result(5, 4))

        with open_store(tmp_path / ".kompend" / "kompend.db") as conn:
            count = conn.execute(f"SELECT COUNT(*) FROM {VERIFICATION_TABLE}").fetchone()[0]
            latest = conn.execute(f"SELECT total, verified, unsupported, claims FROM {VERIFICATION_TABLE} ORDER BY id DESC LIMIT 1").fetchone()
        assert count == 2  # one row per run
        assert latest["total"] == 5
        assert latest["verified"] == 4
        assert latest["unsupported"] == 1
        assert len(json.loads(latest["claims"])) == 5

    def _result(self, total: int, supported: int) -> VerificationResult:
        claims = [
            ClaimVerification(
                node_id=f"n{i}",
                claim=f"c{i}",
                citation=None,
                verdict="supported" if i < supported else "unsupported",
                confidence=0.9 if i < supported else 0.2,
            )
            for i in range(total)
        ]
        return VerificationResult(
            total=total,
            verified=supported,
            unsupported=total - supported,
            average_confidence=0.75,
            claims=claims,
        )


class TestPrivatePagesRoundTrip:
    def test_snapshot_is_replaced_not_appended(self, tmp_path: Path) -> None:
        mark_wiki_private(self._wiki(["a", "b"]), tmp_path)
        mark_wiki_private(self._wiki(["b", "c"]), tmp_path)

        with open_store(tmp_path / ".kompend" / "kompend.db") as conn:
            ids = {r["id"] for r in conn.execute(f"SELECT id FROM {PRIVATE_PAGES_TABLE}")}
        assert ids == {"b", "c"}  # "a" gone — snapshot semantics preserved

    def _wiki(self, ids: list[str]) -> CompiledWiki:
        return CompiledWiki(
            nodes=[WikiNode(id=i, title=f"t-{i}", content="c", node_type="concept") for i in ids],
            edges=[],
            toc=[],
        )


def _vault(tmp_path: Path) -> Path:
    """Create a minimal vault root (``.kompend/config.toml``) for the run log."""
    (tmp_path / ".kompend").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".kompend" / "config.toml").write_text("", encoding="utf-8")
    return tmp_path


def _lint_entry(ts: str, score: float) -> LintEntry:
    return {
        "timestamp": ts,
        "duration_seconds": 1.0,
        "options": {"fix": False},
        "llm": None,
        "config_snapshot": {"compile.compiler": "graph"},
        "wiki_meta": {"node_count": 1},
        "report": {"health_score": score},
    }
