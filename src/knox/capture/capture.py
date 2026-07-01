from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from knox.capture import TIMESTAMP_FORMAT
from knox.capture._utils import make_slug, make_title
from knox.core.template_loader import get_template_loader
from knox.llm.client import JSON_RESPONSE_FORMAT, LLMClient
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
    llm_client: LLMClient | None = None,
    tags: list[str] | None = None,
) -> list[Path]:
    sources_dir.mkdir(parents=True, exist_ok=True)
    captured_dir = sources_dir / "_captured"
    captured_dir.mkdir(parents=True, exist_ok=True)

    user_tags = list(tags or [])
    items = (
        _extract_with_llm(text, llm_client, user_tags)
        if llm_client is not None
        else [CapturedItem(kind="note", title=make_title(text), content=text, tags=user_tags)]
    )

    paths: list[Path] = []
    timestamp = datetime.now(UTC).strftime(TIMESTAMP_FORMAT)
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


def _extract_with_llm(text: str, llm_client: LLMClient, user_tags: list[str]) -> list[CapturedItem]:
    loader = get_template_loader()
    system_prompt = loader.render("capture/capture_system.md.j2")
    tag_hint = f"\n\nThe user suggests these tags as a baseline (add more if relevant, keep these): {user_tags}." if user_tags else ""
    user_prompt = f"Extract knowledge items from this text:{tag_hint}\n\n{text}"

    result = llm_client.complete(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        role="capture",
        response_format=JSON_RESPONSE_FORMAT,
        temperature=0.0,
    )

    if result.json_data is None:
        logger.warning("LLM returned no JSON, falling back to raw capture")
        return [CapturedItem(kind="note", title=make_title(text), content=text)]

    items_raw = result.json_data.get("items", [])
    if not isinstance(items_raw, list):
        return [CapturedItem(kind="note", title=make_title(text), content=text)]

    items: list[CapturedItem] = []
    for raw in items_raw:
        if not isinstance(raw, dict):
            continue
        kind = raw.get("kind", "note")
        title = raw.get("title", make_title(text))
        content = raw.get("content", text)
        tags = raw.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        items.append(CapturedItem(kind=kind, title=title, content=content, tags=_merge_tags(user_tags, tags)))

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
