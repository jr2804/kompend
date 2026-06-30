"""Persistent lint history — append-only JSONL of lint runs.

Each line is a JSON dict with keys: timestamp, duration_seconds, options,
llm, config_snapshot, wiki_meta, report.

ponytail: single JSONL append, no dataclass wrapping. Use
  save_lint_entry()/load_lint_history() for I/O; callers pass dicts.
  If typed access is needed later, add a TypedDict or pydantic model.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from knox.core.config.keys import (
    COMPILE_COMPILER,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
    COMPILE_LLM_ROLE_MODELS,
)
from knox.lint.runner import LintReport, report_to_dict

# Config keys captured in the snapshot
_SNAPSHOT_KEYS = (
    COMPILE_COMPILER,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
    COMPILE_LLM_ROLE_MODELS,
)


def _build_config_snapshot(cfg: object) -> dict[str, Any]:
    """Extract the lint-relevant subset of a resolved config."""
    return {key: getattr(cfg, "get", lambda k: None)(key) for key in _SNAPSHOT_KEYS}


def entry_from_report(
    report: LintReport,
    *,
    duration_seconds: float,
    options: dict[str, Any],
    config: object,
    node_count: int,
    llm: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a lint history dict from a lint run."""
    return {
        "timestamp": timestamp or datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_seconds": round(duration_seconds, 3),
        "options": dict(options),
        "llm": llm,
        "config_snapshot": _build_config_snapshot(config),
        "wiki_meta": {"node_count": node_count},
        "report": report_to_dict(report),
    }


def save_lint_entry(entry: dict[str, Any], path: Path) -> None:
    """Append *entry* as a single JSONL line to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")


def load_lint_history(
    path: Path,
    *,
    last_n: int | None = None,
) -> list[dict[str, Any]]:
    """Load lint history entries from *path* (oldest first).

    A missing file returns an empty list.
    """
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            entries.append(json.loads(stripped))

    if last_n is not None and last_n >= 0:
        entries = entries[-last_n:] if last_n > 0 else []
    return entries


def prune_lint_history(
    path: Path,
    *,
    keep_last: int | None = None,
    max_age_days: int | None = None,
) -> int:
    """Remove old entries from the history file.

    Returns the number of entries removed.
    """
    if keep_last is None and max_age_days is None:
        raise ValueError("At least one of keep_last or max_age_days must be set.")

    entries = load_lint_history(path)
    if not entries:
        return 0

    original_count = len(entries)
    surviving = entries

    if max_age_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
        surviving = [e for e in surviving if _parse_timestamp(e["timestamp"]) >= cutoff]

    if keep_last is not None:
        surviving = surviving[-keep_last:]

    removed = original_count - len(surviving)
    if removed == 0:
        return 0

    _rewrite(path, surviving)
    return removed


def history_to_json(entries: list[dict[str, Any]]) -> str:
    """Serialise a list of entries to a pretty JSON string."""
    return json.dumps(entries, indent=2, default=str, ensure_ascii=False)


def history_to_text(
    entries: list[dict[str, Any]],
    *,
    show_trend: bool = False,
) -> str:
    """Format lint history as a human-readable summary."""
    if not entries:
        return "No lint history found. Run 'knox lint run' first."

    lines: list[str] = []
    lines.append(f"Lint History ({len(entries)} entr{'y' if len(entries) == 1 else 'ies'})")
    lines.append("")

    for entry in entries:
        report = entry.get("report", {})
        score = report.get("health_score", 0.0)
        fails = report.get("failures", 0)
        passes = report.get("passes", 0)
        skips = report.get("skips", 0)
        llm = entry.get("llm")
        model_tag = f"  [{llm.get('model')}]" if llm else ""
        lines.append(f"  {entry['timestamp']}  score={score:.0f}%  {passes} pass / {fails} fail / {skips} skip{model_tag}")

    if show_trend and len(entries) >= 2:
        scores = [e.get("report", {}).get("health_score", 0.0) for e in entries]
        lines.append("")
        lines.append(f"  Trend: {_sparkline(scores)}")
        lines.append(f"  Range: {min(scores):.0f}% — {max(scores):.0f}%")

    return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────────

_SPARK_GLYPHS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float]) -> str:
    """Render a list of numbers as a Unicode sparkline."""
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return _SPARK_GLYPHS[-1] * len(values)
    span = hi - lo
    return "".join(_SPARK_GLYPHS[min(len(_SPARK_GLYPHS) - 1, int((v - lo) / span * (len(_SPARK_GLYPHS) - 1)))] for v in values)


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp, returning a timezone-aware datetime."""
    return datetime.fromisoformat(ts)


def _rewrite(path: Path, entries: list[dict[str, Any]]) -> None:
    """Atomically rewrite the history file with *entries*."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
    tmp.replace(path)
