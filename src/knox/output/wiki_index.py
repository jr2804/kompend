"""Auto-generated index, log, and overview pages for the compiled wiki.

N11 generates three complementary pages after every compile:

- ``index.md`` — Table of contents + wiki stats
- ``log.md`` — Append-only activity journal
- ``overview.md`` — LLM-generated thematic summary (opt-in via ``--synthesize``)
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from knox.core.cache_store import CacheStore
from knox.core.models import CompiledWiki
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

    lines.append("## Summary\n")
    lines.append(f"- **Sources:** {source_count}")
    lines.append(f"- **Articles:** {article_count}")
    lines.append(f"- **Total words:** {total_words:,}")
    lines.append(f"- **Last compile:** {last_compile}")
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


def generate_log_entry(wiki: CompiledWiki) -> str:
    """Generate a single log entry for ``log.md``.

    Returns a Markdown section with timestamp, source/article counts,
    and lists of created/updated pages.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    source_count = wiki.metadata.get("source_count", 0)
    article_count = len(wiki.nodes)

    lines: list[str] = [
        f"## [{now}] compile | {source_count} source(s) → {article_count} page(s)",
    ]

    sources = wiki.metadata.get("source_names", [])
    if sources:
        lines.append(f"- Sources: {', '.join(sources)}")

    # List all pages as wikilinks
    if wiki.nodes:
        page_links = [f"[[{n.title}]]" for n in wiki.nodes[:20]]
        lines.append(f"- Created: {', '.join(page_links)}")
        if len(wiki.nodes) > 20:
            lines.append(f"- ... and {len(wiki.nodes) - 20} more")

    return "\n".join(lines) + "\n"


def append_log(log_path: Path, entry: str) -> None:
    """Append a log entry to *log_path* (create if missing).

    Preserves existing content and adds the new entry below.
    """
    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")
        if existing.strip():
            entry = existing.rstrip("\n") + "\n\n" + entry
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
    index_path = resolved_base / "index.md"
    index_content = generate_index(wiki)
    index_path.write_text(index_content, encoding="utf-8")

    # log.md (appended, never overwritten)
    log_path = resolved_base / "log.md"
    log_entry = generate_log_entry(wiki)
    append_log(log_path, log_entry)

    return {"index": index_path, "log": log_path}



def generate_overview(source_texts: list[str], model: str, overview_budget: int = 4000, timeout: float | None = None) -> str | None:
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

    # Build budget-sized batches of source text.
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

    # Generate a partial overview from each batch.
    partials: list[str] = []
    for batch_text in batches:
        prompt = _loader.render("output/synthesize.md.j2", wiki_content=batch_text)
        try:
            result = client.complete(
                role="synthesize",
                messages=[{"role": "user", "content": prompt}],
            )
            if result.content:
                partials.append(result.content)
        except Exception:  # noqa: BLE001
            continue

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
        return result.content
    except Exception:  # noqa: BLE001
        # Return concatenated partials if the merge call fails.
        return combined


def write_overview(
    wiki: CompiledWiki,
    base_dir: Path,
    model: str,
    overview_budget: int = 4000,
    timeout: float | None = None,
) -> Path | None:
    """Write ``overview.md`` (LLM-generated). Returns the path or None."""
    source_texts = [n.content for n in wiki.nodes if n.content]

    # Overview cache (inside .knox)
    cache_store = CacheStore(base_dir / ".knox" / "cache")
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

    content = generate_overview(source_texts, model=model, overview_budget=overview_budget, timeout=timeout)
    if not content:
        return None

    cache_store.set("overview", overview_deps, f"# Wiki Overview\n\n{content}\n")

    overview_path = base_dir / "overview.md"
    overview_path.write_text(f"# Wiki Overview\n\n{content}\n", encoding="utf-8")
    return overview_path
