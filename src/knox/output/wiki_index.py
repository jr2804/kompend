"""Auto-generated index, log, and overview pages for the compiled wiki.

N11 generates three complementary pages after every compile:

- ``index.md`` — Table of contents + wiki stats
- ``log.md`` — Append-only activity journal
- ``overview.md`` — LLM-generated thematic summary (opt-in via ``--synthesize``)
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from knox.agents.deps import PipelineAgentDeps
from knox.agents.wiki_index import WikiIndexAgent
from knox.core.cache.store import CacheStore
from knox.core.config import DEFAULTS
from knox.core.config.keys import DISPLAY_WIKI_TOPICS
from knox.core.models import CompiledWiki, MetadataField, WikiField
from knox.core.paths import INDEX_FILE, VaultPaths
from knox.core.templates import get_template_loader
from knox.core.utils.time import display_local, iso_local

_loader = get_template_loader()


def generate_index(wiki: CompiledWiki) -> str:
    """Generate ``index.md`` — TOC + stats for the compiled wiki."""
    lines: list[str] = ["# Wiki Index\n"]

    # Stats summary
    source_count = wiki.metadata.get(MetadataField.SOURCE_COUNT, 0)
    article_count = len(wiki.nodes)
    total_words = sum(len(n.content.split()) for n in wiki.nodes)
    last_compile = display_local("%Y-%m-%d %H:%M")
    duration = wiki.metadata.get(WikiField.DURATION_SECONDS)

    lines.append("## Summary\n")
    lines.append(f"- **Sources:** {source_count}")
    lines.append(f"- **Articles:** {article_count}")
    lines.append(f"- **Total words:** {total_words:,}")
    lines.append(f"- **Last compile:** {last_compile}")
    if duration is not None:
        lines.append(f"- **Duration:** {duration:.1f}s")
    if wiki.quality_score:
        lines.append(f"- **Health score:** {wiki.quality_score:.0f}/100")
    lines.append("")

    # Topic / concept cloud (from edges metadata)
    topics = wiki.metadata.get("topics", [])
    if topics:
        topics_limit = DEFAULTS.get(DISPLAY_WIKI_TOPICS, 30)
        lines.append("## Topics\n")
        lines.append(" ".join(f"`{t}`" for t in sorted(topics)[:topics_limit]))
        lines.append("")

    # Full TOC
    lines.append("## Table of Contents\n")
    for i, node in enumerate(wiki.nodes, 1):
        slug = node.id
        lines.append(f"  {i}. [[{slug}|{node.title}]]")

    return "\n".join(lines)


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


def _fmt_tokens_k(n: int) -> str:
    """Format number with K/M suffix (integer-friendly)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


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


def _short_source(path: str) -> str:
    """Strip a source path to just the filename (drop ``sources/`` prefix)."""
    return Path(path).name


def _display_title(text: str, max_chars: int = 80) -> str:
    """Truncate summary text for table display (not content processing)."""
    text = text.strip().replace("\n", " ").replace("|", "\\|")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


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


def _format_per_source_stats(stats: list[dict[str, str | int]]) -> list[str]:
    """Format per-source stats as a markdown table."""
    if not stats:
        return []
    lines = ["", "| Source | Pages | Chars | ~Tokens |", "|--------|------:|------:|--------:|"]
    for s in stats:
        lines.append(f"| {s['stem']} | {s['pages']} | {_fmt_tokens_k(int(s['chars']))} | {_fmt_tokens_k(int(s['tokens_est']))} |")
    return lines


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
            f"| {_short_source(str(s.get('source', '?')))} | {title} | {dur:.1f}s | {summary_cell} | "
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


def prepend_log(log_file: Path, entry: str) -> None:
    """Prepend a log entry to *log_file* (newest first).

    Creates the file if missing.
    """
    if log_file.exists():
        existing = log_file.read_text(encoding="utf-8")
        if existing.strip():
            entry = entry.rstrip("\n") + "\n\n---\n\n" + existing
    log_file.write_text(entry, encoding="utf-8")


def write_index(
    wiki: CompiledWiki,
    output_dir: Path,
    *,
    base_dir: Path | None = None,
) -> dict[str, Path]:
    """Write ``index.md`` for a compiled wiki.

    The human-readable pipeline log is no longer auto-written; regenerate it
    on demand with ``knox log`` (or ``knox log --write``).

    Returns a dict of ``{name: path}`` for generated files.
    """
    resolved_base = base_dir if base_dir is not None else output_dir
    resolved_base.mkdir(parents=True, exist_ok=True)

    # index.md — entry point at base_dir level
    index_file = resolved_base / INDEX_FILE
    index_content = generate_index(wiki)
    index_file.write_text(index_content, encoding="utf-8")

    return {"index": index_file}


def generate_overview(
    source_texts: list[str],
    agent: WikiIndexAgent,
    *,
    overview_budget: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> str | None:
    """Generate an overview of the wiki from node content.

    All *source_texts* are processed — no content is dropped.  If the total
    exceeds ``overview_budget`` chars, texts are batched and each batch
    produces a partial overview.  The partials are then merged in a final
    synthesis call.

    Returns Markdown text, or ``None`` when there is no content to summarise.
    LLM failures propagate to the caller — overview is optional, so the caller
    catches and reports the real error rather than silently producing nothing.
    """
    if not source_texts:
        return None

    batches: list[str] = []
    current_parts: list[str] = []
    current_len = 0
    for text in source_texts:
        if current_len + len(text) > overview_budget and current_parts:
            batches.append("\n---\n".join(current_parts))
            current_parts = []
            current_len = 0
        current_parts.append(text)
        current_len += len(text)
    if current_parts:
        batches.append("\n---\n".join(current_parts))

    total_steps = len(batches) + (1 if len(batches) > 1 else 0)
    if progress_callback:
        progress_callback(0, total_steps)

    partials: list[str] = []
    for idx, batch_text in enumerate(batches):
        prompt = _loader.render("output/synthesize.md.j2", wiki_content=batch_text)
        deps = PipelineAgentDeps(text=prompt, source_id="overview", timeout=agent._timeout)
        try:
            result = agent.run(deps)
            if result and isinstance(result, dict):
                content = result.get("content", "") or str(result.get("text", ""))
                if content:
                    partials.append(content)
        except Exception as exc:
            logger.warning("Overview synthesis batch %d failed: %s", idx, exc)
        if progress_callback:
            progress_callback(idx + 1, total_steps)

    if not partials:
        return None

    if len(partials) == 1:
        return partials[0]

    # Merge partial overviews. A merge failure propagates to the caller,
    # which reports the real error — concatenating partials silently would
    # hide a real LLM failure behind a degraded overview.
    combined = "\n\n".join(partials)
    prompt = _loader.render("output/synthesize.md.j2", wiki_content=combined)
    deps = PipelineAgentDeps(text=prompt, source_id="overview-merge", timeout=agent._timeout)
    try:
        result = agent.run(deps)
        if result and isinstance(result, dict):
            content = result.get("content", "") or str(result.get("text", ""))
            if content:
                if progress_callback:
                    progress_callback(total_steps, total_steps)
                return content
    except Exception as exc:
        logger.warning("Overview merge failed: %s", exc)
    return None


def write_overview(
    wiki: CompiledWiki,
    base_dir: Path,
    agent: WikiIndexAgent,
    *,
    overview_budget: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path | None:
    """Write ``overview.md`` (LLM-generated). Returns the path or None."""
    source_texts = [n.content for n in wiki.nodes if n.content]

    # Overview cache (inside .knox)
    cache_store = CacheStore(VaultPaths(base_dir).cache_dir)
    text_block = "\n".join(sorted(source_texts))
    overview_deps = {
        "source_texts": hashlib.sha256(text_block.encode()).hexdigest(),
        "model": model,
        "overview_budget": str(overview_budget),
    }
    cached = cache_store.get("overview", overview_deps)
    if cached is not None:
        overview_path = base_dir / "overview.md"
        overview_path.write_text(cached, encoding="utf-8")
        return overview_path

    content = generate_overview(source_texts, agent, overview_budget=overview_budget, progress_callback=progress_callback)
    if not content:
        return None

    cache_store.set("overview", overview_deps, f"# Wiki Overview\n\n{content}\n")

    overview_path = base_dir / "overview.md"
    overview_path.write_text(f"# Wiki Overview\n\n{content}\n", encoding="utf-8")
    return overview_path
