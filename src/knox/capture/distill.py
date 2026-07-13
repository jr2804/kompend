"""Distill captured notes into coherent wiki pages.

Reads raw captures from ``sources/_captured/``, groups them by tag, and uses an
LLM to synthesize each group into a single well-structured markdown page written
to ``sources/_distilled/``.  When no LLM client is provided a deterministic
concatenation fallback is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from knox.agents.deps import PipelineAgentDeps
from knox.agents.distill import DistillAgent
from knox.capture import TIMESTAMP_FORMAT
from knox.capture._utils import make_slug, make_title
from knox.core.paths import CAPTURED_SUBDIR, DISTILLED_SUBDIR
from knox.core.templates import get_template_loader
from knox.core.utils.time import display_local
from knox.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CapturedNote:
    """A single parsed capture file."""

    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    source_file: Path | None = None
    captured_at: str = ""


def distill(
    sources_dir: Path,
    *,
    distill_agent: DistillAgent | None = None,
) -> list[Path]:
    """Distill all captured notes into synthesized wiki pages.

    Args:
        sources_dir: Root sources directory (``sources/``).
        llm_client: Optional LLM client for synthesis.  When ``None``, notes are
            merged deterministically without LLM call.

    Returns:
        List of paths to the written distilled pages.
    """
    captured_dir = sources_dir / CAPTURED_SUBDIR
    if not captured_dir.is_dir():
        logger.warning("No captured notes found at %s", captured_dir)
        return []

    notes = _load_captured_notes(captured_dir)
    if not notes:
        logger.info("No captured notes to distill")
        return []

    output_dir = sources_dir / DISTILLED_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = _group_by_tag(notes)
    timestamp = display_local(TIMESTAMP_FORMAT)
    written: list[Path] = []

    for tag, group_notes in groups.items():
        page = _synthesize_with_agent(tag, group_notes, distill_agent) if distill_agent is not None else _synthesize_fallback(tag, group_notes)
        slug = make_slug(page["title"])
        source_files = [n.source_file.name if n.source_file else "" for n in group_notes]
        filepath = output_dir / f"distilled-{timestamp}-{slug}.md"
        _write_page(filepath, page, source_files, timestamp)
        logger.info("Distilled %d notes -> %s", len(group_notes), filepath)
        written.append(filepath)

    return written


# ── Parsing ─────────────────────────────────────────────────────────────


def _load_captured_notes(captured_dir: Path) -> list[CapturedNote]:
    """Read and parse all ``.md`` files from the captured directory."""
    notes: list[CapturedNote] = []
    for filepath in sorted(captured_dir.glob("*.md")):
        text = filepath.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)
        title = meta.get("title") or make_title(body)
        tags_raw = meta.get("tags") or []
        tags = [str(t) for t in tags_raw] if isinstance(tags_raw, list) else []
        captured_at = str(meta.get("captured_at") or "")
        notes.append(
            CapturedNote(
                title=str(title),
                content=body.strip(),
                tags=tags,
                source_file=filepath,
                captured_at=captured_at,
            )
        )
    return notes


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from markdown body.

    Returns ``(metadata_dict, body)``.  If no frontmatter is present, returns
    ``(empty_dict, full_text)``.
    """
    if not text.startswith("---\n"):
        return {}, text
    parts = text[4:].split("\n---\n", 1)
    if len(parts) != 2:
        return {}, text
    try:
        meta = yaml.safe_load(parts[0]) or {}
    except yaml.YAMLError:
        return {}, text
    return meta, parts[1]


def _group_by_tag(notes: list[CapturedNote]) -> dict[str, list[CapturedNote]]:
    """Group notes by their first tag.  Untagged notes go to ``general``."""
    groups: dict[str, list[CapturedNote]] = {}
    for note in notes:
        key = note.tags[0] if note.tags else "general"
        groups.setdefault(key, []).append(note)
    return groups


# ── Synthesis ───────────────────────────────────────────────────────────


def _synthesize_with_agent(
    tag: str,
    notes: list[CapturedNote],
    distill_agent: DistillAgent,
) -> dict:
    """Use the DistillAgent to synthesize a group of notes into a single page."""
    loader = get_template_loader()
    notes_data = [
        {
            "title": n.title,
            "content": n.content,
            "source": n.source_file.name if n.source_file else "unknown",
        }
        for n in notes
    ]
    loader.render(
        "capture/distill.md.j2",
        notes=notes_data,
        group_tag=tag,
    )
    user_prompt = f"Synthesize the {len(notes)} notes tagged '{tag}' into one wiki page."
    deps = PipelineAgentDeps(
        text=user_prompt,
        source_id=f"distill-{tag}",
        timeout=distill_agent._timeout,
    )

    try:
        result = distill_agent.run(deps)
    except Exception:
        logger.warning("DistillAgent failed for tag '%s', using fallback", tag)
        return _synthesize_fallback(tag, notes)

    return {
        "title": result.title or tag.replace("-", " ").replace("_", " ").title(),
        "content": result.content or "",
        "tags": result.tags or [tag],
    }


def _synthesize_fallback(tag: str, notes: list[CapturedNote]) -> dict:
    """Deterministically merge notes without an LLM call."""
    sections = []
    for note in notes:
        sections.append(f"## {note.title}\n\n{note.content}\n")
    body = "\n".join(sections)
    return {
        "title": tag.replace("-", " ").replace("_", " ").title(),
        "content": body.strip(),
        "tags": [tag],
    }


# ── Output ──────────────────────────────────────────────────────────────


def _write_page(
    filepath: Path,
    page: dict,
    source_files: list[str],
    timestamp: str,
) -> None:
    """Write a distilled page with frontmatter."""
    tag_lines = "\n".join(f"  - {t}" for t in page["tags"])
    source_lines = "\n".join(f"  - {s}" for s in source_files)
    frontmatter = (
        f"---\n"
        f"type: distilled\n"
        f"title: {page['title']}\n"
        f"distilled_at: {timestamp}\n"
        f"tags:\n{tag_lines}\n"
        f"synthesized_from:\n{source_lines}\n"
        f"---\n\n"
        f"# {page['title']}\n\n"
        f"{page['content'].strip()}\n"
    )
    filepath.write_text(frontmatter, encoding="utf-8")
