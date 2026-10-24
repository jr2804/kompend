"""Per-run lint artefacts: the full report on disk, a bounded summary on screen.

Every lint run writes the same payload it stores in the ``lint_runs`` table as
two files under ``.kompend/lint-log/``: a Markdown report a human can read and
the JSON entry a tool can parse. The terminal then shows only what fits — the
verdict, one line per failing rule, and the path of the full report. Without
this, a vault with a few hundred findings pushed the health score and every
compile warning out of the scrollback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from kompend.core.config._configs import LintOutput
from kompend.core.models.enums import MetadataField
from kompend.core.paths import VaultPaths
from kompend.lint.history import LintEntry, report_from_dict
from kompend.lint.runner import LintReport, report_to_compact, report_to_summary, report_to_text
from kompend.logging import get_console
from kompend.types import JSONDict

#: Directory name under ``.kompend/`` holding one report per lint run.
LINT_LOG_DIR: Final[str] = "lint-log"


@dataclass(frozen=True)
class LintLogPaths:
    """Where one lint run's artefacts live."""

    markdown: Path
    payload: Path


def write_lint_log(entry: LintEntry, base_dir: Path) -> LintLogPaths:
    """Write the Markdown + JSON artefacts for *entry* and return their paths.

    The JSON file is the history entry verbatim, so ``lint show`` and a
    consumer of the ``lint_runs`` table see the same payload. The Markdown file
    is the full human report that no longer has to reach the terminal.
    """
    directory = lint_log_dir(base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(str(entry["timestamp"]))
    paths = LintLogPaths(markdown=directory / f"{stamp}.md", payload=directory / f"{stamp}.json")

    report = report_from_dict(entry["report"])
    options = entry.get("options") or {}
    wiki_meta = entry.get("wiki_meta") or {}
    duration = entry["duration_seconds"]  # nocheck - lint-history schema, not WikiField
    page_count = wiki_meta.get(MetadataField.NODE_COUNT, 0)
    header = [
        "# Lint report",
        "",
        f"- Run: {entry['timestamp']} ({options.get('context', 'lint')})",
        f"- Duration: {duration:.1f}s over {page_count} page(s)",
    ]
    model = (entry.get("llm") or {}).get(MetadataField.MODEL)
    if model:
        header.append(f"- Model: {model}")
    paths.markdown.write_text("\n".join(header) + "\n\n" + report_to_text(report) + "\n", encoding="utf-8")
    paths.payload.write_text(json.dumps(entry, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return paths


def _stamp(timestamp: str) -> str:
    """Turn an ISO timestamp into a filename-safe stamp.

    ``2026-10-24T01:42:11+02:00`` becomes ``2026-10-24T01-42-11``; the offset is
    dropped because the stamp is read as a local wall-clock identifier and must
    sort lexicographically.
    """
    head = timestamp.split("+", 1)[0].split("Z", 1)[0]
    return head.replace(":", "-")


def latest_lint_log(base_dir: Path) -> Path | None:
    """Newest Markdown report for *base_dir*, or ``None`` when no run exists."""
    directory = lint_log_dir(base_dir)
    if not directory.is_dir():
        return None
    reports = sorted(directory.glob("*.md"))
    return reports[-1] if reports else None


def lint_log_dir(base_dir: Path) -> Path:
    """Directory holding the per-run lint reports for *base_dir*."""
    return VaultPaths(base_dir).kompend_dir / LINT_LOG_DIR


def read_lint_log(path: Path) -> JSONDict | str:
    """Read a lint artefact; ``.json`` returns the parsed entry, ``.md`` text."""
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def render_report(report: LintReport, level: LintOutput, paths: LintLogPaths) -> None:
    """Print *report* at *level*, ending with the path of the full report.

    ``quiet`` is the default for a compile run: the verdict and where the
    details are. ``summary`` adds one line per failing rule. ``full`` keeps the
    old behaviour for an interactive ``kompend lint all``.
    """
    console = get_console()
    if level is LintOutput.FULL:
        console.print(report_to_text(report))
    elif level is LintOutput.SUMMARY:
        console.print(report_to_compact(report))
    else:
        console.print(report_to_summary(report))
    console.print(f"  Full report: {paths.markdown}")
