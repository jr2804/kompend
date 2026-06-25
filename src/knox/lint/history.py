"""Persistent lint history — append-only JSONL of lint runs.

Each entry captures a full :class:`LintReport` plus the context in which it
was produced (options, config snapshot, wiki metadata).  This lets agents and
users answer:

* "What problems were in the last run?"
* "How did the score trend over time?"
* "What changed between run A and run B?"

Storage format: one JSON object per line (JSONL) at
``.knox/lint_history.jsonl``.  Append-only by default; old entries can be
removed with :func:`prune_lint_history`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from knox.core.config.keys import (
    COMPILE_CHUNK_SIZE,
    COMPILE_COMPILER,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
    COMPILE_LLM_ROLE_MODELS,
)
from knox.lint.runner import LintReport, report_to_dict

if TYPE_CHECKING:
    from knox.core.config import ResolvedConfig

# ── Config keys captured in the snapshot ─────────────────────────────────
_SNAPSHOT_KEYS = (
    COMPILE_COMPILER,
    COMPILE_CHUNK_SIZE,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
    COMPILE_LLM_ROLE_MODELS,
)


# ── Data model ───────────────────────────────────────────────────────────


@dataclass
class LintHistoryEntry:
    """A single lint-run record persisted in the history log.

    Attributes:
    ----------
    timestamp:
        ISO-8601 UTC string.
    duration_seconds:
        Wall-clock duration of the lint run.
    options:
        Effective lint options used (rules, enable_llm, fix, threshold,
        llm_context_budget).
    llm:
        LLM metadata ({model, usage}) or ``None`` if no LLM rules ran.
    config_snapshot:
        Relevant subset of the resolved config at lint time.
    wiki_meta:
        Wiki-level metadata ({node_count}).
    report:
        Full serialised :class:`LintReport` (via :func:`report_to_dict`).
    """

    timestamp: str
    duration_seconds: float
    options: dict[str, Any]
    llm: dict[str, Any] | None
    config_snapshot: dict[str, Any]
    wiki_meta: dict[str, Any]
    report: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialise to a single JSON-line string (no trailing newline)."""
        return json.dumps(asdict(self), default=str, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LintHistoryEntry:
        """Reconstruct from a parsed JSON dict."""
        return cls(
            timestamp=data["timestamp"],
            duration_seconds=data.get("duration_seconds", 0.0),
            options=data.get("options", {}),
            llm=data.get("llm"),
            config_snapshot=data.get("config_snapshot", {}),
            wiki_meta=data.get("wiki_meta", {}),
            report=data.get("report", {}),
        )


# ── Factory ──────────────────────────────────────────────────────────────


def _build_config_snapshot(cfg: ResolvedConfig) -> dict[str, Any]:
    """Extract the lint-relevant subset of a resolved config."""
    return {key: cfg.get(key) for key in _SNAPSHOT_KEYS}


def entry_from_report(
    report: LintReport,
    *,
    duration_seconds: float,
    options: dict[str, Any],
    config: ResolvedConfig,
    node_count: int,
    llm: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> LintHistoryEntry:
    """Build a :class:`LintHistoryEntry` from a lint run.

    Parameters
    ----------
    report:
        The aggregated :class:`LintReport`.
    duration_seconds:
        Wall-clock duration of the run.
    options:
        Effective lint options (rules, enable_llm, fix, threshold,
        llm_context_budget).
    config:
        The resolved config — only a lint-relevant subset is captured.
    node_count:
        Number of wiki nodes linted.
    llm:
        LLM metadata ({model, usage}) or ``None``.
    timestamp:
        Override the auto-generated ISO-8601 UTC timestamp (testing).
    """
    return LintHistoryEntry(
        timestamp=timestamp or datetime.now(UTC).isoformat(timespec="seconds"),
        duration_seconds=round(duration_seconds, 3),
        options=dict(options),
        llm=llm,
        config_snapshot=_build_config_snapshot(config),
        wiki_meta={"node_count": node_count},
        report=report_to_dict(report),
    )


# ── Persistence ──────────────────────────────────────────────────────────


def save_lint_entry(entry: LintHistoryEntry, path: Path) -> None:
    """Append *entry* as a single JSONL line to *path*.

    Creates parent directories as needed.  Never overwrites existing content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(entry.to_json() + "\n")


def load_lint_history(
    path: Path,
    *,
    last_n: int | None = None,
) -> list[LintHistoryEntry]:
    """Load lint history entries from *path*.

    Returns entries in chronological order (oldest first).  When *last_n* is
    given, only the most recent *last_n* entries are returned.

    A missing file returns an empty list — this is an *absence*, not a failure.
    A corrupt line raises :class:`json.JSONDecodeError` (let it burn).
    """
    if not path.exists():
        return []

    entries: list[LintHistoryEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            entries.append(LintHistoryEntry.from_dict(json.loads(stripped)))

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

    Parameters
    ----------
    keep_last:
        If set, keep only the most recent *keep_last* entries.
    max_age_days:
        If set, remove entries older than *max_age_days* days (measured from
        their timestamp).

    At least one of *keep_last* / *max_age_days* must be provided.  Returns
    the number of entries removed.  A missing file returns 0.
    """
    if keep_last is None and max_age_days is None:
        raise ValueError("At least one of keep_last or max_age_days must be set.")

    entries = load_lint_history(path)
    if not entries:
        return 0

    original_count = len(entries)
    surviving = entries

    if max_age_days is not None:
        cutoff = datetime.now(UTC) - _days_timedelta(max_age_days)
        surviving = [e for e in surviving if _parse_timestamp(e.timestamp) >= cutoff]

    if keep_last is not None:
        surviving = surviving[-keep_last:]

    removed = original_count - len(surviving)
    if removed == 0:
        return 0

    _rewrite(path, surviving)
    return removed


# ── Output formatting ────────────────────────────────────────────────────


def history_to_json(entries: list[LintHistoryEntry]) -> str:
    """Serialise a list of entries to a pretty JSON string."""
    return json.dumps([asdict(e) for e in entries], indent=2, default=str, ensure_ascii=False)


def history_to_text(
    entries: list[LintHistoryEntry],
    *,
    show_trend: bool = False,
) -> str:
    """Format lint history as a human-readable summary.

    When *show_trend* is ``True``, include an ASCII sparkline of health scores.
    """
    if not entries:
        return "No lint history found. Run 'knox lint run' first."

    lines: list[str] = []
    lines.append(f"Lint History ({len(entries)} entr{'y' if len(entries) == 1 else 'ies'})")
    lines.append("")

    for entry in entries:
        report = entry.report
        score = report.get("health_score", 0.0)
        fails = report.get("failures", 0)
        passes = report.get("passes", 0)
        skips = report.get("skips", 0)
        model = entry.llm.get("model") if entry.llm else None
        model_tag = f"  [{model}]" if model else ""
        lines.append(f"  {entry.timestamp}  score={score:.0f}%  {passes} pass / {fails} fail / {skips} skip{model_tag}")

    if show_trend and len(entries) >= 2:
        scores = [e.report.get("health_score", 0.0) for e in entries]
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


def _days_timedelta(days: int) -> timedelta:
    return timedelta(days=days)


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp, returning a timezone-aware datetime."""
    return datetime.fromisoformat(ts)


def _rewrite(path: Path, entries: list[LintHistoryEntry]) -> None:
    """Atomically rewrite the history file with *entries*."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(entry.to_json() + "\n")
    tmp.replace(path)
