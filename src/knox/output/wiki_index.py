"""Auto-generated index, log, and overview pages for the compiled wiki.

N11 generates three complementary pages after every compile:

- ``index.md`` — Table of contents + wiki stats
- ``log.md`` — Append-only activity journal
- ``overview.md`` — LLM-generated thematic summary (opt-in via ``--synthesize``)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knox.core.models import CompiledWiki


def generate_index(wiki: CompiledWiki) -> str:
    """Generate ``index.md`` — TOC + stats for the compiled wiki."""
    lines: list[str] = ["# Wiki Index\n"]

    # Stats summary
    source_count = wiki.metadata.get("source_count", 0)
    article_count = len(wiki.nodes)
    total_words = sum(len(n.content.split()) for n in wiki.nodes)
    last_compile = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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
) -> dict[str, Path]:
    """Write ``index.md`` and ``log.md`` for a compiled wiki.

    Returns a dict of ``{name: path}`` for generated files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # index.md (regenerated each compile)
    index_path = output_dir / "index.md"
    index_content = generate_index(wiki)
    index_path.write_text(index_content, encoding="utf-8")

    # log.md (appended, never overwritten)
    log_path = output_dir / "log.md"
    log_entry = generate_log_entry(wiki)
    append_log(log_path, log_entry)

    return {"index": index_path, "log": log_path}


def generate_overview(source_texts: list[str], model: str = "llama3.2") -> str | None:
    """Generate ``overview.md`` via LLM (opt-in).

    Returns the Markdown overview text, or ``None`` if unavailable.
    """
    # This is a stub — requires LLM client which may not be wired yet.
    # The actual implementation will use LLMClient.complete() with a
    # synthesis prompt.
    return None


def write_overview(
    wiki: CompiledWiki,
    output_dir: Path,
    model: str = "llama3.2",
) -> Path | None:
    """Write ``overview.md`` (LLM-generated). Returns the path or None.

    Requires Q12/Q13 (LLM client) to be wired. Returns None if the
    LLM client is unavailable.
    """
    try:
        from knox.llm.client import LLMClient  # noqa: PLC0415
    except ImportError:
        return None

    client = LLMClient()

    source_texts = [n.content for n in wiki.nodes if n.content]
    if not source_texts:
        return None

    prompt = (
        "You are synthesizing a concise thematic overview of a knowledge wiki. "
        "Write 2-3 paragraphs covering:\n"
        "- The main domain and key themes\n"
        "- Prominent entities and concepts\n"
        "- Notable relationships or open questions\n\n"
        "Wiki content:\n\n"
        + "\n---\n".join(text[:2000] for text in source_texts[:10])
    )

    try:
        result = client.complete(
            role="synthesize",
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        content = result.content if hasattr(result, "content") else str(result)
        overview_path = output_dir / "overview.md"
        overview_path.write_text(f"# Wiki Overview\n\n{content}\n", encoding="utf-8")
        return overview_path
    except Exception:
        return None
