from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kompend.agents.capture import CaptureAgent
from kompend.agents.deps import PipelineAgentDeps
from kompend.capture import TIMESTAMP_FORMAT
from kompend.capture._utils import format_sources_markdown, make_slug, make_title
from kompend.core.paths import CAPTURED_SUBDIR
from kompend.core.utils.time import display_local
from kompend.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CapturedItem:
    kind: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)


def capture(
    text: str,
    *,
    sources_dir: Path,
    capture_agent: CaptureAgent | None = None,
    tags: list[str] | None = None,
    attributes: dict[str, str] | None = None,
) -> list[Path]:
    sources_dir.mkdir(parents=True, exist_ok=True)
    captured_dir = sources_dir / CAPTURED_SUBDIR
    captured_dir.mkdir(parents=True, exist_ok=True)

    if not text.strip():
        raise ValueError("capture input is empty. Provide text with something to extract.")

    user_tags = list(tags or [])
    items = (
        _extract_with_agent(text, capture_agent, user_tags)
        if capture_agent is not None
        else [CapturedItem(kind="note", title=make_title(text), content=text, tags=user_tags)]
    )

    paths: list[Path] = []
    timestamp = display_local(TIMESTAMP_FORMAT)
    for i, item in enumerate(items):
        slug = make_slug(item.title)
        filename = f"capture-{timestamp}-{i:02d}-{slug}.md"
        filepath = captured_dir / filename

        tag_lines = "\n".join(f"  - {t}" for t in item.tags)
        attr_lines = "\n".join(f"  {k}: {v}" for k, v in (attributes or {}).items())
        frontmatter = f"""---
type: {item.kind}
title: {item.title}
captured_at: {timestamp}
tags:
{tag_lines}
attributes:
{attr_lines}
source: capture
---

# {item.title}

{item.content.strip()}
"""
        filepath.write_text(frontmatter, encoding="utf-8")
        logger.info("Captured %s -> %s", item.kind, filepath)
        paths.append(filepath)

    format_sources_markdown(sources_dir)
    return paths


def _extract_with_agent(text: str, capture_agent: CaptureAgent, user_tags: list[str]) -> list[CapturedItem]:
    """Use the CaptureAgent to extract structured items from text.

    Returns an empty list when the agent refuses the input (nonsense,
    nothing extractable) — the caller warns and writes nothing. Falls back
    to a raw note only when the agent itself fails (not when it refuses).
    """
    tag_hint = f"\n\nThe user suggests these tags as a baseline (add more if relevant, keep these): {user_tags}." if user_tags else ""
    user_prompt = f"Extract knowledge items from this text:{tag_hint}\n\n{text}"
    deps = PipelineAgentDeps(text=user_prompt, source_id="capture", timeout=capture_agent.timeout)

    try:
        result = capture_agent.run(deps)
    except Exception:
        logger.warning("CaptureAgent failed, falling back to raw capture")
        return [CapturedItem(kind="note", title=make_title(text), content=text)]

    if result.refused:
        reason = result.refusal_reason or "the input contains nothing extractable"
        logger.warning("CaptureAgent refused input: %s", reason)
        return []

    items: list[CapturedItem] = []
    for raw in result.items:
        items.append(
            CapturedItem(
                kind=raw.kind or "note",
                title=raw.title or make_title(text),
                content=raw.content or text,
                tags=_merge_tags(user_tags, raw.tags),
            )
        )

    if not items:
        logger.warning("CaptureAgent returned no items for the input")
    return items


def _merge_tags(user_tags: list[str], llm_tags: list[str]) -> list[str]:
    """User tags first, then LLM tags, deduplicated (case-insensitive), order preserved."""
    merged: list[str] = []
    seen: set[str] = set()
    for t in [*user_tags, *llm_tags]:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            merged.append(t)
    return merged
