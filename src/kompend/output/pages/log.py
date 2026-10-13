"""Generate ``log.md`` — append-only activity journal entries.

Two concerns live here:

- :func:`generate_log_entry` — rich per-compile entry (stats, tables) built
  through :class:`~kompend.output.page_builder.PageBuilder`.
- :func:`append_log_entry` / :func:`log_stage` — OKF §9 one-line journal
  entries (date-grouped, newest first).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from wenmode.nodes import Node

from kompend.core.models import CompiledWiki, ExtractionStats, MetadataField, WikiField
from kompend.core.page_builder import PageBuilder, code_span, strong
from kompend.core.paths import LOG_FILE, VaultPaths
from kompend.core.pipeline.config import Tier
from kompend.core.utils.time import iso_local
from kompend.logging import get_logger

__all__ = [
    "LogStage",
    "append_log_entry",
    "generate_log_entry",
    "log_stage",
]

logger = get_logger(__name__)


class LogStage(StrEnum):
    """Pipeline stage identifier for rich log entries."""

    COMPILE = "compile"
    INGEST = "ingest"
    LINT = "lint"


def log_stage(vault: VaultPaths, *, action: str, detail: str) -> None:
    """Append a stage entry (``Ingest``/``Query``/``Lint``/…) to the bundle log.

    Best-effort: a log-write failure must never break the command that
    produced the event.
    """
    try:
        append_log_entry(vault.output_dir / LOG_FILE, action=action, detail=detail)
    except OSError as exc:
        logger.warning("Could not append %s entry to log: %s", action, exc)


def generate_log_entry(wiki: CompiledWiki, *, stage: LogStage) -> str:
    """Generate a rich log entry for ``log.md``.

    The heading line is assembled verbatim (timestamp brackets and the
    ``|`` separator are intentional literal formatting); every table and
    stat block below it is built through :class:`PageBuilder`.
    """
    now = iso_local()
    meta = wiki.metadata
    source_count = meta.get(MetadataField.SOURCE_COUNT, 0)
    article_count = len(wiki.nodes)
    duration = meta.get(WikiField.DURATION_SECONDS)
    llm_model = meta.get(WikiField.LLM_MODEL)

    header_by_stage: dict[LogStage, str] = {
        LogStage.COMPILE: f"{source_count} source(s) → {article_count} page(s)",
        LogStage.INGEST: f"{source_count} source(s)",
        LogStage.LINT: f"{article_count} page(s)",
    }
    parts = [header_by_stage[stage]]
    if duration is not None:
        parts.append(_fmt_duration(duration))
    if llm_model:
        parts.append(llm_model)
    head = f"## [{now}] {stage.value} | " + " | ".join(parts)

    b = PageBuilder()

    # Sources: short inline list (≤5) or table with titles.
    source_names = meta.get(MetadataField.SOURCE_NAMES, [])
    extraction_stats_raw = meta.get("extraction_stats") or []
    extraction_stats: list[ExtractionStats] = [ExtractionStats(**d) for d in extraction_stats_raw] if extraction_stats_raw else []
    titles = _extraction_titles(extraction_stats)
    if source_names:
        _add_sources_block(b, source_names, titles or None)

    per_source = _per_source_stats(wiki)
    if per_source:
        _add_per_source_table(b, per_source)

    if extraction_stats:
        _add_extraction_block(b, extraction_stats)

    if wiki.nodes:
        total_words = sum(len(n.content.split()) for n in wiki.nodes)
        avg_words = total_words // article_count if article_count else 0
        total_chars = sum(len(n.content or "") for n in wiki.nodes)
        tokens_est = max(total_chars // 4, 1)
        b.paragraph(
            strong("Content:"),
            f" {_fmt_tokens(total_words)} words ({_fmt_tokens(total_chars)} chars, ~{_fmt_tokens(tokens_est)} tokens) — avg {avg_words:,} words/page",
        )
        type_counts = sorted(_node_type_counts(wiki).items(), key=lambda x: -x[1])
        breakdown = ", ".join(f"{count} {ntype}" for ntype, count in type_counts)
        b.paragraph(strong("Created:"), f" {article_count} pages ({breakdown})")

    edge_count = len(wiki.edges)
    if edge_count:
        edge_counts = sorted(_edge_type_counts(wiki).items(), key=lambda x: -x[1])
        breakdown = ", ".join(f"{count} {etype}" for etype, count in edge_counts)
        b.paragraph(strong("Cross-references:"), f" {edge_count:,} edges ({breakdown})")

    llm_usage = meta.get(WikiField.LLM_USAGE, {})
    if usage_line := _llm_usage_line(llm_usage, llm_model):
        b.paragraph(usage_line)

    tier_timings = meta.get(WikiField.TIER_TIMINGS, {})
    if tier_timings:
        _add_tier_timings_table(b, tier_timings, duration)

    if stage == LogStage.LINT and wiki.quality_score:
        b.paragraph(strong("Health score:"), f" {wiki.quality_score:.0f}/100")

    body = b.render().rstrip("\n")
    return f"{head}\n{body}\n" if body else f"{head}\n"


# ── block builders ───────────────────────────────────────────────────────


def _add_per_source_table(b: PageBuilder, stats: list[dict[str, str | int]]) -> None:
    """Per-source stats table (pages, chars, tokens)."""
    b.table(
        headers=["Source", "Pages", "Chars", "~Tokens"],
        rows=[[str(s["stem"]), str(s["pages"]), _fmt_tokens(int(s["chars"])), _fmt_tokens(int(s["tokens_est"]))] for s in stats],
        aligns=[None, "right", "right", "right"],
    )


def _add_tier_timings_table(b: PageBuilder, timings: dict[str, float], total_secs: float | None) -> None:
    """Tier timing rows from a ``{tier: seconds}`` dict."""
    rows: list[list[str | Node]] = [[tier.value, _fmt_duration(timings[tier.value])] for tier in Tier if tier.value in timings]
    if total_secs is not None:
        rows.append([strong("total"), strong(_fmt_duration(total_secs))])
    b.paragraph(strong("Timing:"))
    b.table(headers=["Stage", "Duration"], rows=rows, aligns=[None, "right"])


def _add_sources_block(b: PageBuilder, sources: list[str], titles: dict[str, str] | None) -> None:
    """Sources as an inline code list (≤5, untitled) or a table."""
    names = [Path(s).name for s in sources]
    if len(names) <= 5 and not titles:
        parts: list[str | Node] = []
        for i, n in enumerate(names):
            if i:
                parts.append(", ")
            parts.append(code_span(n))
        b.paragraph(*parts)
        return
    rows: list[list[str | Node]] = []
    for i, name in enumerate(names, 1):
        row: list[str | Node] = [str(i), name]
        if titles:
            row.append(titles.get(sources[i - 1], "").strip().replace("\n", " "))
        rows.append(row)
    headers = ["#", "Source", "Title"] if titles else ["#", "Source"]
    b.table(headers=headers, rows=rows)


def _add_extraction_block(b: PageBuilder, stats: list[ExtractionStats]) -> None:
    """Per-source extraction timing + counts table with a total row."""
    b.paragraph(strong("Extraction:"))
    rows: list[list[str | Node]] = []
    total_time = 0.0
    total_words = 0
    for s in stats:
        total_time += s.duration_s
        total_words += s.summary_words
        title = s.summary.strip().replace("\n", " ").replace("|", "\\|")
        rows.append(
            [
                Path(s.source).name,
                title,
                f"{s.duration_s:.1f}s",
                f"{s.summary_words}w" if s.summary_words else "-",
                str(s.entity_count),
                str(s.concept_count),
                str(s.tag_count),
                str(s.batches),
            ]
        )
    if len(stats) > 1:
        rows.append(
            [
                strong("total"),
                "",
                strong(f"{total_time:.1f}s"),
                strong(f"{total_words}w"),
                "",
                "",
                "",
                "",
            ]
        )
    headers = ["Source", "Title", "Time", "Summary", "Entities", "Concepts", "Tags", "Batches"]
    b.table(headers=headers, rows=rows, aligns=[None, None, "right", "right", "right", "right", "right", "right"])


def _llm_usage_line(llm_usage: dict[str, Any], llm_model: str | None) -> str | None:
    """One-line LLM usage summary, or None when no calls were made."""
    calls = llm_usage.get("call_count", 0)
    if calls <= 0:
        return None
    per_model = llm_usage.get("per_model", {})
    models = ", ".join(per_model.keys()) if per_model else (llm_model or "unknown")
    pt = llm_usage.get("prompt_tokens", 0)
    ct = llm_usage.get("completion_tokens", 0)
    ttft = llm_usage.get("avg_ttft", 0)
    overall = llm_usage.get("overall_tok_s", 0)
    recent = llm_usage.get("recent_tok_s", 0)
    rate_s = f" | {overall:.1f} tok/s avg ({recent:.1f} recent)" if overall else ""
    return f"{strong('LLM:')} {calls} calls | {_fmt_tokens(pt)} in / {_fmt_tokens(ct)} out{rate_s} | avg TTFT {ttft:.1f}s | {models}"


# ── small helpers ────────────────────────────────────────────────────────


def _extraction_titles(stats: list[ExtractionStats] | None) -> dict[str, str]:
    """Map extraction source → summary for the source-list Title column."""
    if not stats:
        return {}
    return {s.source: s.summary for s in stats if s.source and s.summary}


def _node_type_counts(wiki: CompiledWiki) -> dict[str, int]:
    """Count nodes by ``node_type``."""
    counts: dict[str, int] = {}
    for node in wiki.nodes:
        counts[node.node_type] = counts.get(node.node_type, 0) + 1
    return counts


def _edge_type_counts(wiki: CompiledWiki) -> dict[str, int]:
    """Count edges by ``type`` field."""
    counts: dict[str, int] = {}
    for edge in wiki.edges:
        etype = edge.get("type", "untyped") if isinstance(edge, dict) else "untyped"
        counts[etype] = counts.get(etype, 0) + 1
    return counts


def _per_source_stats(wiki: CompiledWiki) -> list[dict[str, str | int]]:
    """Group generated pages by source stem; sorted by descending page count."""
    source_map: dict[str, list[int]] = {}
    for node in wiki.nodes:
        sc = node.source_chunks or []
        if not sc:
            continue
        stem = sc[0].split("/")[0]
        entry = source_map.setdefault(stem, [0, 0])
        entry[0] += 1
        entry[1] += len(node.content or "")
    return [
        {"stem": stem, "pages": pages, "chars": chars, "tokens_est": max(chars // 4, 1)}
        for stem, (pages, chars) in sorted(source_map.items(), key=lambda x: -x[1][0])
    ]


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration: ``1.2s``, ``3m 24s``, ``1h 02m``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _fmt_tokens(n: int) -> str:
    """Compact token/count format: ``999``, ``12K``, ``1.3M``."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return f"{n:,}"


# ── OKF §9 append-only journal ───────────────────────────────────────────


def append_log_entry(log_file: Path, *, action: str, detail: str) -> None:
    """Record one OKF §9 log entry, newest first.

    Entries are date-grouped under ``## YYYY-MM-DD`` headings (per §9), each
    entry a list item with a ``[HH:mm:ss]`` timestamp prefix and a bold
    action. Newest date section first; newest entry first within a day.
    The file is created with a ``# Bundle Update Log`` header when missing
    and is never deleted by recompiles.
    """
    header = "# Bundle Update Log"
    now = datetime.now(UTC)
    date_heading = f"## {now.strftime('%Y-%m-%d')}"
    entry = f"- [{now.strftime('%H:%M:%S')}] **{action}**: {detail}\n"

    existing = log_file.read_text(encoding="utf-8").strip() if log_file.exists() else ""
    if not existing:
        log_file.write_text(f"{header}\n\n{date_heading}\n\n{entry}", encoding="utf-8")
        return
    if not existing.startswith(header):
        existing = f"{header}\n\n{existing}"

    if date_heading in existing:
        anchor_end = existing.index(date_heading) + len(date_heading)
        rest = existing[anchor_end:].lstrip("\n")
        body = f"{existing[:anchor_end]}\n\n{entry}{rest}"
    else:
        rest = existing[len(header) :].lstrip("\n")
        body = f"{header}\n\n{date_heading}\n\n{entry}\n{rest}"
    log_file.write_text(body, encoding="utf-8")
