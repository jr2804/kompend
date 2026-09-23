"""On-demand pipeline log report builder (the engine behind ``knox log``).

Aggregates the structured stores produced during a build into a single
human-readable markdown report — without re-running any LLM calls:

- **Last compile**: rendered from ``last_wiki.json`` metadata (extraction
  stats, page/entity/concept counts, LLM usage, timings) via
  :func:`knox.output.wiki_index.generate_log_entry`.
- **Command timeline**: from ``runs.jsonl`` — every command that ran, with
  status, duration, and crashes (a ``start`` with no matching ``end`` flags
  a killed/hung process).
- **Lint trend**: from ``lint_history.jsonl`` — recent health scores.

This is the API surface: ``knox log`` is a thin CLI wrapper around
:func:`build_log_report`.
"""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants.

from __future__ import annotations

import sqlite3
from pathlib import Path

from knox.core.models import load_cached_wiki
from knox.core.page_builder import PageBuilder, code_span, em
from knox.core.paths import VaultPaths
from knox.core.reporting.runlog import RunEvent
from knox.core.store import open_store
from knox.core.store.schema import RUNS_TABLE
from knox.lint.history import history_to_text, load_lint_history
from knox.logging import get_logger
from knox.output.wiki_index import LogStage, generate_log_entry

logger = get_logger(__name__)


def build_log_report(base_dir: Path, *, last_n: int = 25) -> str:
    """Build the full pipeline log report for the vault at *base_dir*.

    Args:
        base_dir: Vault root (the wiki dir; ``.knox`` lives beneath it).
        last_n: Maximum command/run entries and lint entries to include.

    Returns:
        A markdown string.
    """
    paths = VaultPaths(base_dir)
    sections: list[str] = [f"# KNOX Log — {base_dir.name}", ""]

    compile_section = _compile_section(base_dir)
    if compile_section:
        sections.extend(compile_section)

    runs_section = _runs_section(paths.knox_db_file, last_n=last_n)
    if runs_section:
        sections.extend(runs_section)

    lint_section = _lint_section(paths.knox_db_file, last_n=last_n)
    if lint_section:
        sections.extend(lint_section)

    if len(sections) <= 2:
        sections.append("_No compile, run, or lint data found yet._")
    return "\n".join(sections).rstrip() + "\n"


def _compile_section(base_dir: Path) -> list[str]:
    """Render the last-compile detail from ``last_wiki.json`` metadata."""
    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        return []
    entry = generate_log_entry(wiki, stage=LogStage.COMPILE).strip()
    if not entry:
        return []
    return ["## Last Compile", "", entry, ""]


def _runs_section(db_path: Path, *, last_n: int) -> list[str]:
    """Render the command timeline from the ``runs`` table."""
    runs = _load_runs(db_path)
    if not runs:
        return []
    # Show the most recent runs; each "end" is a completed command.
    ended = [r for r in runs if r.get("event") == "end"]
    recent = ended[-last_n:]
    if not recent:
        b = PageBuilder()
        b.heading(2, "Command Timeline")
        b.paragraph(em("Commands started but none finished yet."))
        return [b.render().rstrip(), ""]
    b = PageBuilder()
    b.heading(2, "Command Timeline")
    b.table(
        headers=["When", "Command", "Status", "Duration"],
        rows=[
            [
                _short_ts(r.get("ts", "")),
                code_span(str(r.get("cmd", ""))),
                ("✓" if r.get("status") == "ok" else "✗") + f" {r.get('status', '')}",
                f"{r['duration_s']:.1f}s" if isinstance(r.get("duration_s"), int | float) else "—",
            ]
            for r in recent
        ],
        aligns=[None, None, None, "right"],
    )
    lines = [b.render().rstrip()]
    # Flag any start with no matching end (killed / hung).
    ended_ids = {r["run_id"] for r in runs if r.get("event") == "end"}
    dangling = [r for r in runs if r.get("event") == "start" and r["run_id"] not in ended_ids]
    if dangling:
        lines.append("")
        lines.append(f"_Note: {len(dangling)} command(s) started but never finished (killed or hung)._")
    return [*lines, ""]


def _lint_section(db_path: Path, *, last_n: int) -> list[str]:
    """Render the lint trend from the ``lint_runs`` table."""
    entries = load_lint_history(db_path, last_n=last_n)
    if not entries:
        return []
    text = history_to_text(entries, show_trend=True)
    return ["## Lint Trend", "", text, ""]


def _load_runs(db_path: Path) -> list[RunEvent]:
    """Read the ``runs`` table into event dicts (oldest first, nulls omitted)."""
    out: list[RunEvent] = []
    if not db_path.is_file():
        return out
    try:
        with open_store(db_path) as conn:
            rows = conn.execute(f"SELECT event, run_id, cmd, ts, status, duration_s, exit, error FROM {RUNS_TABLE} ORDER BY id").fetchall()
    except sqlite3.Error as exc:
        logger.debug("could not read runs table: %s", exc)
        return out
    for row in rows:
        event: RunEvent = {"event": row["event"], "run_id": row["run_id"], "cmd": row["cmd"], "ts": row["ts"]}
        if row["status"] is not None:
            event["status"] = row["status"]
        if row["duration_s"] is not None:
            event["duration_s"] = row["duration_s"]
        if row["exit"] is not None:
            event["exit"] = row["exit"]
        if row["error"] is not None:
            event["error"] = row["error"]
        out.append(event)
    return out


def _short_ts(ts: str) -> str:
    """Compact a full ISO timestamp to ``YYYY-MM-DD HH:MM``."""
    if not ts:
        return "—"
    # Tolerate both offset (local) and Z (legacy UTC) forms.
    return ts.replace("T", " ")[5:16] if len(ts) >= 16 else ts
