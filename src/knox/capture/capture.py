from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from knox.agents.capture import CaptureAgent
from knox.agents.deps import PipelineAgentDeps
from knox.capture import TIMESTAMP_FORMAT
from knox.capture._utils import make_slug, make_title
from knox.core.paths import CAPTURED_SUBDIR
from knox.core.utils.time import display_local
from knox.logging import get_logger

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
) -> list[Path]:
    sources_dir.mkdir(parents=True, exist_ok=True)
    captured_dir = sources_dir / CAPTURED_SUBDIR
    captured_dir.mkdir(parents=True, exist_ok=True)

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
        frontmatter = f"""---
type: {item.kind}
title: {item.title}
captured_at: {timestamp}
tags:
{tag_lines}
source: capture
---

# {item.title}

{item.content.strip()}
"""
        filepath.write_text(frontmatter, encoding="utf-8")
        logger.info("Captured %s -> %s", item.kind, filepath)
        paths.append(filepath)

    return paths


def _extract_with_agent(text: str, capture_agent: CaptureAgent, user_tags: list[str]) -> list[CapturedItem]:
    """Use the CaptureAgent to extract structured items from text."""
    tag_hint = f"\n\nThe user suggests these tags as a baseline (add more if relevant, keep these): {user_tags}." if user_tags else ""
    user_prompt = f"Extract knowledge items from this text:{tag_hint}\n\n{text}"
    deps = PipelineAgentDeps(text=user_prompt, source_id="capture", timeout=capture_agent._timeout)

    try:
        result = capture_agent.run(deps)
    except Exception:
        logger.warning("CaptureAgent failed, falling back to raw capture")
        return [CapturedItem(kind="note", title=make_title(text), content=text)]

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

    return items if items else [CapturedItem(kind="note", title=make_title(text), content=text, tags=user_tags)]


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
