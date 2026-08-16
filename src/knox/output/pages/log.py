"""Generate ``log.md`` — append-only activity journal entries."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from knox.core.models import CompiledWiki, MetadataField, WikiField
from knox.core.paths import LOG_FILE, VaultPaths
from knox.core.utils.time import iso_local
from knox.logging import get_logger

logger = get_logger(__name__)


def log_stage(vault: VaultPaths, *, action: str, detail: str) -> None:
    """Append a stage entry (``Ingest``/``Query``/``Lint``/\u2026) to the bundle log.

    Best-effort: a log-write failure must never break the command that
    produced the event.
    """
    try:
        append_log_entry(vault.output_dir / LOG_FILE, action=action, detail=detail)
    except OSError as exc:
        logger.warning("Could not append %s entry to log: %s", action, exc)


def generate_log_entry(wiki: CompiledWiki, *, stage: str = "compile") -> str:  # noqa: PLR0915
    """Generate a rich log entry for ``log.md``.

    Parameters
    ----------
    wiki:
        The compiled wiki (may be empty for non-compile stages).
    stage:
        Pipeline stage identifier: ``"compile"``, ``"ingest"``, ``"lint"``, etc.
    """
    now = iso_local()
    source_count = wiki.metadata.get(MetadataField.SOURCE_COUNT, 0)
    article_count = len(wiki.nodes)
    duration = wiki.metadata.get(WikiField.DURATION_SECONDS)
    llm_model = wiki.metadata.get(WikiField.LLM_MODEL)
    tier_timings = wiki.metadata.get(WikiField.TIER_TIMINGS, {})
    llm_usage = wiki.metadata.get(WikiField.LLM_USAGE, {})

    header_parts: list[str] = []
    if stage == "compile":
        header_parts.append(f"{source_count} source(s) → {article_count} page(s)")
    elif stage == "ingest":
        header_parts.append(f"{source_count} source(s)")
    elif stage == "lint":
        header_parts.append(f"{article_count} page(s)")
    if duration is not None:
        header_parts.append(_fmt_duration(duration))
    if llm_model:
        header_parts.append(llm_model)

    header_suffix = " | ".join(header_parts)
    lines: list[str] = [f"## [{now}] {stage}"]
    if header_suffix:
        lines[-1] += f" | {header_suffix}"

    # Source list and per-source stats
    source_names = wiki.metadata.get(MetadataField.SOURCE_NAMES, [])
    extraction_stats = wiki.metadata.get("extraction_stats")
    # Build source→summary map so the source list table can show titles.
    source_titles: dict[str, str] = {}
    if extraction_stats:
        for es in extraction_stats:
            src = str(es.get("source", ""))
            summary = str(es.get("summary", ""))
            if src and summary:
                source_titles[src] = summary
    if source_names:
        lines.append("")
        lines.extend(_format_sources_table(source_names, titles=source_titles or None))

    per_source = _per_source_stats(wiki)
    if per_source:
        lines.extend(_format_per_source_stats(per_source))

    if extraction_stats:
        lines.extend(_format_extraction_stats(extraction_stats))

    # Aggregate content stats
    if wiki.nodes:
        total_words = sum(len(n.content.split()) for n in wiki.nodes)
        avg_words = total_words // article_count if article_count else 0
        total_chars = sum(len(n.content or "") for n in wiki.nodes)
        total_tokens_est = max(total_chars // 4, 1)
        lines.append("")
        lines.append(
            f"**Content:** {_fmt_tokens(total_words)} words "
            f"({_fmt_tokens_k(total_chars)} chars, ~{_fmt_tokens_k(total_tokens_est)} tokens) "
            f"— avg {avg_words:,} words/page"
        )

    # Created: counts by node_type
    if wiki.nodes:
        type_counts = _node_type_counts(wiki)
        parts = [f"{count} {ntype}" for ntype, count in sorted(type_counts.items(), key=lambda x: -x[1])]
        lines.append("")
        lines.append(f"**Created:** {article_count} pages ({', '.join(parts)})")

    # Cross-references: edge type breakdown
    edge_count = len(wiki.edges)
    if edge_count:
        edge_types = _edge_type_counts(wiki)
        parts = [f"{count} {etype}" for etype, count in sorted(edge_types.items(), key=lambda x: -x[1])]
        lines.append("")
        lines.append(f"**Cross-references:** {edge_count:,} edges ({', '.join(parts)})")

    lines.extend(_format_llm_usage_block(llm_usage, llm_model))
    lines.extend(_format_tier_timings_block(tier_timings, duration))

    # Health score (lint stage)
    if stage == "lint" and wiki.quality_score:
        lines.append("")
        lines.append(f"**Health score:** {wiki.quality_score:.0f}/100")

    return "\n".join(lines) + "\n"


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
    """Group generated pages by source stem with per-source stats.

    Returns a list of dicts sorted by descending page count, each with:
    - ``stem`` (str)
    - ``pages`` (int)
    - ``chars`` (int)
    - ``tokens_est`` (int, chars/4)
    """
    source_map: dict[str, list[int]] = {}
    for node in wiki.nodes:
        sc = node.source_chunks or []
        if not sc:
            continue
        stem = sc[0].split("/")[0] if "/" in sc[0] else sc[0].rsplit("_", 1)[0] if "_" in sc[0] else sc[0]
        source_map.setdefault(stem, [0, 0])
        source_map[stem][0] += 1
        source_map[stem][1] += len(node.content or "")

    result: list[dict[str, str | int]] = []
    for stem, (pages, chars) in sorted(source_map.items(), key=lambda x: -x[1][0]):
        result.append({"stem": stem, "pages": pages, "chars": chars, "tokens_est": max(chars // 4, 1)})
    return result


def _format_per_source_stats(stats: list[dict[str, str | int]]) -> list[str]:
    """Format per-source stats as a markdown table."""
    if not stats:
        return []
    lines = ["", "| Source | Pages | Chars | ~Tokens |", "|--------|------:|------:|--------:|"]
    for s in stats:
        lines.append(f"| {s['stem']} | {s['pages']} | {_fmt_tokens_k(int(s['chars']))} | {_fmt_tokens_k(int(s['tokens_est']))} |")
    return lines


def _fmt_tokens_k(n: int) -> str:
    """Format number with K/M suffix (integer-friendly)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _format_extraction_stats(stats: list[dict[str, Any]]) -> list[str]:
    """Format per-source extraction timing + summary stats as a markdown table.

    One row per source: extraction time, summary length, and counts of
    entities/concepts/tags extracted plus the number of LLM batches needed.
    The Title column shows the first ~80 chars of the extraction summary.
    """
    if not stats:
        return []
    lines = [
        "",
        "**Extraction:**",
        "",
        "| Source | Title | Time | Summary | Entities | Concepts | Tags | Batches |",
        "|--------|-------|-----:|--------:|---------:|---------:|-----:|--------:|",
    ]
    total_time = 0.0
    total_words = 0
    for s in stats:
        dur = float(s.get("duration_s", 0.0))
        total_time += dur
        words = int(s.get("summary_words", 0))
        total_words += words
        summary_cell = f"{words}w" if words else "-"
        title = _display_title(str(s.get("summary", "")))
        lines.append(
            f"| {_short_source(str(s.get('source', 'unknown')))} | {title} | {dur:.1f}s | {summary_cell} | "
            f"{s.get('entity_count', 0)} | {s.get('concept_count', 0)} | "
            f"{s.get('tag_count', 0)} | {s.get('batches', 0)} |"
        )
    if len(stats) > 1:
        lines.append(f"| **total** | | **{total_time:.1f}s** | **{total_words}w** | | | | |")
    return lines


def _format_llm_usage_block(llm_usage: dict, llm_model: str | None) -> list[str]:
    """Build the LLM usage lines for the log entry."""
    if not llm_usage or llm_usage.get("call_count", 0) <= 0:
        return []
    calls = llm_usage["call_count"]
    pt = llm_usage.get("prompt_tokens", 0)
    ct = llm_usage.get("completion_tokens", 0)
    ttft = llm_usage.get("avg_ttft", 0)
    per_model = llm_usage.get("per_model", {})
    model_names = ", ".join(per_model.keys()) if per_model else (llm_model or "unknown")
    return [
        "",
        f"**LLM:** {calls} calls | {_fmt_tokens(pt)} in / {_fmt_tokens(ct)} out | avg TTFT {ttft:.1f}s | {model_names}",
    ]


def _format_tier_timings_block(tier_timings: dict, duration: float | None) -> list[str]:
    """Build the timing breakdown table lines for the log entry."""
    if not tier_timings:
        return []
    lines = ["", "**Timing:**", "", "| Stage | Duration |", "|-------|----------|"]
    for tier_name in ("index", "chunk", "resolve", "compile", "output"):
        if tier_name in tier_timings:
            lines.append(f"| {tier_name} | {_fmt_duration(tier_timings[tier_name])} |")
    if duration is not None:
        lines.append(f"| **total** | **{_fmt_duration(duration)}** |")
    return lines


def generate_stage_log_entry(
    *,
    stage: str,
    duration: float | None = None,
    source_count: int = 0,
    source_names: list[str] | None = None,
    chunk_count: int | None = None,
    health_score: float | None = None,
    rule_summary: str | None = None,
    fixes_applied: int | None = None,
    llm_model: str | None = None,
    llm_usage: dict | None = None,
) -> str:
    """Generate a log entry for non-compile stages (ingest, lint).

    This is a lighter-weight entry that doesn't require a full
    :class:`CompiledWiki` object.
    """
    now = iso_local()

    header_parts: list[str] = []
    if source_count:
        header_parts.append(f"{source_count} source(s)")
    if chunk_count is not None:
        header_parts.append(f"{chunk_count} chunk(s)")
    if health_score is not None:
        header_parts.append(f"score {health_score:.0f}%")
    if duration is not None:
        header_parts.append(_fmt_duration(duration))
    if llm_model:
        header_parts.append(llm_model)

    header_suffix = " | ".join(header_parts)
    lines: list[str] = [f"## [{now}] {stage}"]
    if header_suffix:
        lines[-1] += f" | {header_suffix}"

    if source_names:
        lines.append("")
        lines.extend(_format_sources_table(source_names))

    if fixes_applied is not None and fixes_applied > 0:
        lines.append("")
        lines.append(f"**Fixes applied:** {fixes_applied}")

    if rule_summary:
        lines.append("")
        lines.append(f"**Rules:** {rule_summary}")

    if health_score is not None:
        lines.append("")
        lines.append(f"**Health score:** {health_score:.0f}/100")

    if llm_usage and llm_usage.get("call_count", 0) > 0:
        calls = llm_usage["call_count"]
        pt = llm_usage.get("prompt_tokens", 0)
        ct = llm_usage.get("completion_tokens", 0)
        lines.append("")
        lines.append(f"**LLM:** {calls} calls | {_fmt_tokens(pt)} in / {_fmt_tokens(ct)} out")

    return "\n".join(lines) + "\n"


def _fmt_tokens(n: int) -> str:
    """Format token count with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds >= 3600:
        h = seconds / 3600
        return f"{h:.1f}h"
    if seconds >= 60:
        m = seconds / 60
        return f"{m:.1f}m"
    return f"{seconds:.1f}s"


def _format_sources_table(sources: list[str], titles: dict[str, str] | None = None) -> list[str]:
    """Format a source list as a markdown table.

    No row cap: a comprehensive ingest list is more useful than a tidy log.
    When *titles* is provided (a mapping of full source path → summary text),
    a Title column is added showing the document's summary as a display title.
    """
    if len(sources) <= 5 and not titles:
        return [", ".join(f"`{_short_source(s)}`" for s in sources)]
    if len(sources) <= 5:
        lines = ["", "| # | Source | Title |", "|---|--------|-------|"]
        for i, src in enumerate(sources, 1):
            title = _display_title((titles or {}).get(src, ""))
            lines.append(f"| {i} | {_short_source(src)} | {title} |")
        return lines
    lines = ["", "| # | Source | Title |", "|---|--------|-------|"]
    for i, src in enumerate(sources, 1):
        title = _display_title((titles or {}).get(src, ""))
        lines.append(f"| {i} | {_short_source(src)} | {title} |")
    return lines


def _short_source(path: str) -> str:
    """Strip a source path to just the filename (drop ``sources/`` prefix)."""
    return Path(path).name


def _display_title(text: str, max_chars: int = 80) -> str:
    """Format summary text for table display (no truncation)."""
    return text.strip().replace("\n", " ").replace("|", "\\|")


def prepend_log(log_file: Path, entry: str) -> None:
    """Prepend a raw entry to *log_file* (newest first). Creates the file if missing.

    Low-level helper: *entry* is inserted verbatim above any existing content,
    separated by ``---``. Prefer :func:`append_log_entry` for OKF §9 entries.
    """
    if log_file.exists():
        existing = log_file.read_text(encoding="utf-8")
        if existing.strip():
            entry = entry.rstrip("\n") + "\n\n---\n\n" + existing
    log_file.write_text(entry, encoding="utf-8")


def append_log_entry(log_file: Path, *, action: str, detail: str) -> None:
    """Record one OKF §9 log entry, newest first.

    Entries are date-grouped under ``## YYYY-MM-DD`` headings (per §9), each
    entry a list item with a ``[HH:mm:ss]`` timestamp prefix and a bold
    action (``- [14:23:45] **Query**: \u2026``). Newest date section first;
    newest entry first within a day. The file is created with a
    ``# Bundle Update Log`` header when missing and is never deleted by
    recompiles.
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
        # Newest entry first within the day: insert right after the heading.
        anchor_end = existing.index(date_heading) + len(date_heading)
        rest = existing[anchor_end:].lstrip("\n")
        body = f"{existing[:anchor_end]}\n\n{entry}{rest}"
    else:
        # Newest date section first, directly after the file header.
        rest = existing[len(header):].lstrip("\n")
        body = f"{header}\n\n{date_heading}\n\n{entry}\n{rest}"
    log_file.write_text(body, encoding="utf-8")
