"""Auto-generated index, log, and overview pages for the compiled wiki.

N11 generates three complementary pages after every compile:

- ``index.md`` — Table of contents + wiki stats
- ``log.md`` — Append-only activity journal
- ``overview.md`` — LLM-generated thematic summary (opt-in via ``--synthesize``)
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from knox.core.cache_store import CacheStore
from knox.core.models import CompiledWiki
from knox.core.paths import CACHE_SUBDIR, INDEX_FILE, KNOX_DIR
from knox.core.template_loader import get_template_loader
from knox.llm.client import LLMClient

_loader = get_template_loader()


def generate_index(wiki: CompiledWiki) -> str:
    """Generate ``index.md`` — TOC + stats for the compiled wiki."""
    lines: list[str] = ["# Wiki Index\n"]

    # Stats summary
    source_count = wiki.metadata.get("source_count", 0)
    article_count = len(wiki.nodes)
    total_words = sum(len(n.content.split()) for n in wiki.nodes)
    last_compile = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    duration = wiki.metadata.get("duration_seconds")

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
        lines.append("## Topics\n")
        lines.append(" ".join(f"`{t}`" for t in sorted(topics)[:30]))
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


def _sources_for_table(wiki: CompiledWiki) -> list[tuple[str, int, int]]:
    """Group pages by source stem → (stem, page_count, word_count)."""
    source_map: dict[str, list[int]] = {}
    for node in wiki.nodes:
        sc = node.source_chunks or []
        if not sc:
            continue
        stem = sc[0].split("/")[0] if "/" in sc[0] else sc[0].rsplit("_", 1)[0] if "_" in sc[0] else sc[0]
        words = len(node.content.split())
        entry = source_map.setdefault(stem, [0, 0])
        entry[0] += 1
        entry[1] += words
    return [(stem, pages, words) for stem, (pages, words) in sorted(source_map.items(), key=lambda x: -x[1][0])]


def _format_sources_table(sources: list[str], max_rows: int = 12) -> list[str]:
    """Format a source list as a markdown table or inline list."""
    if len(sources) <= 5:
        return [", ".join(f"`{s}`" for s in sources)]
    lines = ["", "| # | Source |", "|---|--------|"]
    for i, src in enumerate(sources[:max_rows], 1):
        lines.append(f"| {i} | {src} |")
    if len(sources) > max_rows:
        lines.append(f"| ... | _{len(sources) - max_rows} more_ |")
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


def generate_log_entry(wiki: CompiledWiki, *, stage: str = "compile") -> str:
    """Generate a rich log entry for ``log.md``.

    Parameters
    ----------
    wiki:
        The compiled wiki (may be empty for non-compile stages).
    stage:
        Pipeline stage identifier: ``"compile"``, ``"ingest"``, ``"lint"``, etc.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    source_count = wiki.metadata.get("source_count", 0)
    article_count = len(wiki.nodes)
    duration = wiki.metadata.get("duration_seconds")
    llm_model = wiki.metadata.get("llm_model")
    tier_timings = wiki.metadata.get("tier_timings", {})
    llm_usage = wiki.metadata.get("llm_usage", {})

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

    source_names = wiki.metadata.get("source_names", [])
    if source_names:
        lines.append("")
        lines.extend(_format_sources_table(source_names))

    # Per-document stats
    if wiki.nodes:
        total_words = sum(len(n.content.split()) for n in wiki.nodes)
        avg_words = total_words // article_count if article_count else 0
        lines.append("")
        lines.append(f"**Words:** {total_words:,} total (~{avg_words:,}/page)")

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
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

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


def prepend_log(log_path: Path, entry: str) -> None:
    """Prepend a log entry to *log_path* (newest first).

    Creates the file if missing.
    """
    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")
        if existing.strip():
            entry = entry.rstrip("\n") + "\n\n---\n\n" + existing
    log_path.write_text(entry, encoding="utf-8")


def write_index_and_log(
    wiki: CompiledWiki,
    output_dir: Path,
    *,
    base_dir: Path | None = None,
) -> dict[str, Path]:
    """Write ``index.md`` and ``log.md`` for a compiled wiki.

    Returns a dict of ``{name: path}`` for generated files.
    """
    resolved_base = base_dir if base_dir is not None else output_dir
    resolved_base.mkdir(parents=True, exist_ok=True)

    # index.md — entry point at base_dir level
    index_path = resolved_base / INDEX_FILE
    index_content = generate_index(wiki)
    index_path.write_text(index_content, encoding="utf-8")

    # log.md (prepended, newest first)
    log_path = resolved_base / "log.md"
    log_entry = generate_log_entry(wiki, stage="compile")
    prepend_log(log_path, log_entry)

    return {"index": index_path, "log": log_path}


def generate_overview(
    source_texts: list[str],
    model: str,
    *,
    overview_budget: int,
    timeout: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> str | None:
    """Generate an overview of the wiki from node content.

    All *source_texts* are processed — no content is dropped.  If the total
    exceeds ``overview_budget`` chars, texts are batched and each batch
    produces a partial overview.  The partials are then merged in a final
    synthesis call.

    Returns Markdown text, or ``None`` when the LLM is unavailable.
    """
    if not source_texts:
        return None

    try:
        client = LLMClient(default_model=model, fallback_chain=[], max_retries_per_model=1, timeout=timeout)
    except Exception:  # noqa: BLE001
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
        try:
            result = client.complete(
                role="synthesize",
                messages=[{"role": "user", "content": prompt}],
            )
            if result.content:
                partials.append(result.content)
        except Exception:  # noqa: BLE001, S110
            pass
        if progress_callback:
            progress_callback(idx + 1, total_steps)

    if not partials:
        return None

    if len(partials) == 1:
        return partials[0]

    # Merge partial overviews.
    combined = "\n\n".join(partials)
    prompt = _loader.render("output/synthesize.md.j2", wiki_content=combined)
    try:
        result = client.complete(
            role="synthesize",
            messages=[{"role": "user", "content": prompt}],
        )
        if progress_callback:
            progress_callback(total_steps, total_steps)
        return result.content
    except Exception:  # noqa: BLE001
        # Return concatenated partials if the merge call fails.
        if progress_callback:
            progress_callback(total_steps, total_steps)
        return combined


def write_overview(
    wiki: CompiledWiki,
    base_dir: Path,
    model: str,
    *,
    overview_budget: int,
    timeout: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path | None:
    """Write ``overview.md`` (LLM-generated). Returns the path or None."""
    source_texts = [n.content for n in wiki.nodes if n.content]

    # Overview cache (inside .knox)
    cache_store = CacheStore(base_dir / KNOX_DIR / CACHE_SUBDIR)
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

    content = generate_overview(source_texts, model=model, overview_budget=overview_budget, timeout=timeout, progress_callback=progress_callback)
    if not content:
        return None

    cache_store.set("overview", overview_deps, f"# Wiki Overview\n\n{content}\n")

    overview_path = base_dir / "overview.md"
    overview_path.write_text(f"# Wiki Overview\n\n{content}\n", encoding="utf-8")
    return overview_path
