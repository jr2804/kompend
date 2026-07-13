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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knox.core.models import load_cached_wiki
from knox.core.paths import VaultPaths
from knox.core.utils.exceptions import shorten_string
from knox.lint.history import history_to_text, load_lint_history
from knox.logging import get_logger
from knox.output.wiki_index import generate_log_entry

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

    runs_section = _runs_section(paths.runs_path, last_n=last_n)
    if runs_section:
        sections.extend(runs_section)

    lint_section = _lint_section(paths.lint_history_path, last_n=last_n)
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
    entry = generate_log_entry(wiki, stage="compile").strip()
    if not entry:
        return []
    return ["## Last Compile", "", entry, ""]


def _runs_section(runs_path: Path, *, last_n: int) -> list[str]:
    """Render the command timeline from ``runs.jsonl``."""
    runs = _load_runs(runs_path)
    if not runs:
        return []
    # Show the most recent runs; each "end" is a completed command.
    ended = [r for r in runs if r.get("event") == "end"]
    recent = ended[-last_n:]
    if not recent:
        return ["## Command Timeline", "", "_Commands started but none finished yet._", ""]
    lines = ["## Command Timeline", "", "| When | Command | Status | Duration |", "|------|---------|--------|---------:|"]
    for r in recent:
        ts = _short_ts(r.get("ts", ""))
        cmd = str(r.get("cmd", ""))
        status = str(r.get("status", ""))
        dur = r.get("duration_s")
        dur_str = f"{dur:.1f}s" if isinstance(dur, int | float) else "—"
        marker = "✓" if status == "ok" else "✗"
        lines.append(f"| {ts} | `{cmd}` | {marker} {status} | {dur_str} |")
    # Flag any start with no matching end (killed / hung).
    started_ids = {r["run_id"] for r in runs if r.get("event") == "start"}
    ended_ids = {r["run_id"] for r in runs if r.get("event") == "end"}
    dangling = [r for r in runs if r.get("event") == "start" and r["run_id"] not in ended_ids]
    if dangling:
        lines.append("")
        lines.append(f"_Note: {len(dangling)} command(s) started but never finished (killed or hung)._")
    _ = started_ids  # kept for clarity / future use
    return [*lines, ""]


def _lint_section(lint_path: Path, *, last_n: int) -> list[str]:
    """Render the lint trend from ``lint_history.jsonl``."""
    if not lint_path.is_file():
        return []
    entries = load_lint_history(lint_path, last_n=last_n)
    if not entries:
        return []
    text = history_to_text(entries, show_trend=True)
    return ["## Lint Trend", "", text, ""]


def _load_runs(runs_path: Path) -> list[dict[str, Any]]:
    """Parse ``runs.jsonl`` into a list of event dicts (skips bad lines)."""
    if not runs_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for raw in runs_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            out.append(json.loads(stripped))
        except json.JSONDecodeError:
            logger.debug("skipping unparseable run entry: %s", shorten_string(stripped), exc_info=True)
    return out


def _short_ts(ts: str) -> str:
    """Compact a full ISO timestamp to ``YYYY-MM-DD HH:MM``."""
    if not ts:
        return "—"
    # Tolerate both offset (local) and Z (legacy UTC) forms.
    return ts.replace("T", " ")[5:16] if len(ts) >= 16 else ts
