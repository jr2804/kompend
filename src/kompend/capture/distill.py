"""Distill captured notes into coherent wiki pages.

Reads raw captures from ``sources/_captured/``, groups them by tag, and uses an
LLM to synthesize each group into a single well-structured markdown page written
to ``sources/_distilled/``.  When no LLM client is provided a deterministic
concatenation fallback is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import yaml

from kompend.agents.deps import DistillDeps
from kompend.agents.distill import DistillAgent
from kompend.agents.distill_agentic import AgenticDistillAgent
from kompend.agents.types import DistillAgentDeps
from kompend.capture import TIMESTAMP_FORMAT
from kompend.capture._utils import format_sources_markdown, make_slug, make_title
from kompend.core.paths import ARCHIVE_SUBDIR, CAPTURED_SUBDIR, DISTILLED_SUBDIR, LEARNED_SUBDIR
from kompend.core.utils.time import display_local
from kompend.logging import get_logger
from kompend.output.frontmatter import okf_frontmatter_yaml, source_frontmatter

logger = get_logger(__name__)


@dataclass
class CapturedNote:
    """A single parsed capture file."""

    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    source_file: Path | None = None
    captured_at: str = ""


class DistilledPage(TypedDict):
    """One synthesized distillation page, as written to frontmatter + body."""

    title: str
    content: str
    tags: list[str]


def distill(
    sources_dir: Path,
    *,
    distill_agent: DistillAgent | AgenticDistillAgent | None = None,
    archive: bool = True,
) -> list[Path]:
    """Distill all captured notes into synthesized wiki pages.

    Args:
        sources_dir: Root sources directory (``sources/``).
        distill_agent: Optional agent for synthesis.  When ``None``, notes are
            merged deterministically without LLM call.
        archive: When ``True`` (default), the source notes that were distilled
            are moved to ``sources/_archive/<kind>/`` after a successful
            write. Pass ``--no-archive`` to keep them in place.

    Returns:
        List of paths to the written distilled pages.
    """
    captured_dir = sources_dir / CAPTURED_SUBDIR
    learned_dir = sources_dir / LEARNED_SUBDIR
    notes = _load_captured_notes(captured_dir) + _load_captured_notes(learned_dir)
    if not notes:
        logger.info("No captured notes to distill")
        return []

    output_dir = sources_dir / DISTILLED_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = _group_by_tag(notes)
    timestamp = display_local(TIMESTAMP_FORMAT)
    written: list[Path] = []

    if distill_agent is not None and isinstance(distill_agent, AgenticDistillAgent):
        written = _distill_agentic(groups, notes, distill_agent, timestamp, output_dir)
    else:
        for tag, group_notes in groups.items():
            page = _synthesize_with_agent(tag, group_notes, distill_agent) if distill_agent is not None else _synthesize_fallback(tag, group_notes)
            slug = make_slug(page["title"])
            source_files = [n.source_file.name if n.source_file else "" for n in group_notes]
            filepath = _distilled_path(output_dir, slug, timestamp)
            _write_page(filepath, page, source_files, timestamp)
            logger.info("Distilled %d notes -> %s", len(group_notes), filepath)
            written.append(filepath)

    if archive and written:
        _archive_notes(notes, sources_dir)

    format_sources_markdown(sources_dir)
    return written


def _archive_notes(notes: list[CapturedNote], sources_dir: Path) -> None:
    """Move distilled source notes to ``sources/_archive/<kind>/``.

    The kind subdir mirrors the source zone (``capture`` / ``learn``) so
    archived files stay identifiable. Files are moved, not copied — the
    distilled page is now the canonical home of the content.
    """
    archive_root = sources_dir / ARCHIVE_SUBDIR
    for note in notes:
        if note.source_file is None or not note.source_file.exists():
            continue
        kind = "capture" if CAPTURED_SUBDIR in note.source_file.parts else "learn"
        dest_dir = archive_root / kind
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / note.source_file.name
        if dest.exists():
            dest.unlink()
        note.source_file.rename(dest)
        logger.info("Archived %s -> %s", note.source_file.name, dest)


# ── Parsing ─────────────────────────────────────────────────────────────


def _load_captured_notes(captured_dir: Path) -> list[CapturedNote]:
    """Read and parse all ``.md`` files from the captured directory."""
    notes: list[CapturedNote] = []
    for filepath in sorted(captured_dir.glob("*.md")):
        text = filepath.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)
        title = meta.get("title") or make_title(body)
        tags_raw = meta.get("tags") or []  # nocheck: frontmatter dict, not wiki metadata
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


def _distill_agentic(
    groups: dict[str, list[CapturedNote]],
    notes: list[CapturedNote],
    distill_agent: AgenticDistillAgent,
    timestamp: str,
    output_dir: Path,
) -> list[Path]:
    """Distill all groups in a single agentic session."""
    groups_data = [
        {"tag": tag, "notes": [{"title": n.title, "content": n.content, "source": n.source_file.name if n.source_file else "unknown"} for n in note_list]}
        for tag, note_list in groups.items()
    ]
    deps = DistillAgentDeps(groups=groups_data, source_id="distill-all", timeout=distill_agent.timeout)

    try:
        distill_agent.run(deps)
    except Exception as exc:
        logger.warning("AgenticDistillAgent failed: %s, falling back to per-group", exc)
        return _distill_fallback_per_group(groups, timestamp, output_dir)

    written = []
    for tag, note_list in groups.items():
        page_data = deps.pages.get(tag)
        if page_data is None:
            page = _synthesize_fallback(tag, note_list)
        else:
            raw_tags = page_data.get("tags")  # nocheck: OKF frontmatter key, not wiki metadata
            page = DistilledPage(
                title=str(page_data.get("title") or tag.replace("-", " ").replace("_", " ").title()),
                content=str(page_data.get("content") or ""),
                tags=[str(t) for t in raw_tags] if isinstance(raw_tags, list) else [tag],
            )
        slug = make_slug(page["title"])
        source_files = [n.source_file.name if n.source_file else "" for n in note_list]
        filepath = _distilled_path(output_dir, slug, timestamp)
        _write_page(filepath, page, source_files, timestamp)
        logger.info("Distilled %d notes -> %s", len(note_list), filepath)
        written.append(filepath)
    return written


def _distill_fallback_per_group(
    groups: dict[str, list[CapturedNote]],
    timestamp: str,
    output_dir: Path,
) -> list[Path]:
    """Deterministically merge every group after the agentic session failed.

    The agentic agent's contract is one multi-group session (``groups``/
    ``pages`` deps); it has no per-group entry point, so there is no second
    LLM pass to fall back to. Passing it to ``_synthesize_with_agent`` — which
    builds the *structured-output* agent's deps — only ever rendered a broken
    prompt and landed here anyway, via the fallback it already returned.
    """
    written: list[Path] = []
    for tag, group_notes in groups.items():
        page = _synthesize_fallback(tag, group_notes)
        slug = make_slug(page["title"])
        source_files = [n.source_file.name if n.source_file else "" for n in group_notes]
        filepath = _distilled_path(output_dir, slug, timestamp)
        _write_page(filepath, page, source_files, timestamp)
        logger.info("Distilled %d notes -> %s", len(group_notes), filepath)
        written.append(filepath)
    return written


def _distilled_path(output_dir: Path, slug: str, timestamp: str) -> Path:
    """Return the distilled page path for *slug*.

    If a page for the same slug already exists (a previous distill run),
    that file is reused — the group's page is *updated* in place instead of
    accumulating a new timestamped file per run. This is what keeps repeated
    ``kompend distill`` runs from piling up near-duplicate pages.
    """
    existing = sorted(output_dir.glob(f"distilled-*-{slug}.md"))
    if existing:
        return existing[0]
    return output_dir / f"distilled-{timestamp}-{slug}.md"


# ── Synthesis ───────────────────────────────────────────────────────────


def _synthesize_with_agent(
    tag: str,
    notes: list[CapturedNote],
    distill_agent: DistillAgent,
) -> DistilledPage:
    """Use the DistillAgent to synthesize a group of notes into a single page."""
    notes_data = [
        {
            "title": n.title,
            "content": n.content,
            "source": n.source_file.name if n.source_file else "unknown",
        }
        for n in notes
    ]
    user_prompt = f"Synthesize the {len(notes)} notes tagged '{tag}' into one wiki page."
    deps = DistillDeps(
        notes=notes_data,
        group_tag=tag,
        text=user_prompt,
        source_id=f"distill-{tag}",
        timeout=distill_agent.timeout,
    )

    try:
        result = distill_agent.run(deps)
    except Exception:
        logger.warning("DistillAgent failed for tag '%s', using fallback", tag)
        return _synthesize_fallback(tag, notes)

    return {
        "title": result.title or tag.replace("-", " ").replace("_", " ").title(),
        "content": result.content or "",
        "tags": result.tags or [tag],  # nocheck: frontmatter dict, not wiki metadata
    }


def _synthesize_fallback(tag: str, notes: list[CapturedNote]) -> DistilledPage:
    """Deterministically merge notes without an LLM call."""
    sections = []
    for note in notes:
        sections.append(f"## {note.title}\n\n{note.content}\n")
    body = "\n".join(sections)
    return {
        "title": tag.replace("-", " ").replace("_", " ").title(),
        "content": body.strip(),
        "tags": [tag],  # nocheck: frontmatter dict, not wiki metadata
    }


# ── Output ──────────────────────────────────────────────────────────────


def _write_page(
    filepath: Path,
    page: DistilledPage,
    source_files: list[str],
    timestamp: str,
) -> None:
    """Write a distilled page with frontmatter."""
    frontmatter = okf_frontmatter_yaml(
        source_frontmatter(
            page_type="distilled",
            title=page["title"],
            timestamp=timestamp,
            timestamp_key="distilled_at",
            tags=list(page["tags"]),  # nocheck: frontmatter dict, not wiki metadata
            extra={"synthesized_from": list(source_files)},
        ),
    )
    filepath.write_text(
        f"{frontmatter}# {page['title']}\n\n{page['content'].strip()}\n",
        encoding="utf-8",
    )
