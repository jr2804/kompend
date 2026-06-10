from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from slugify import slugify as _slugify_impl

logger = logging.getLogger(__name__)


def learn(
    text: str,
    *,
    sources_dir: Path,
    tags: list[str] | None = None,
) -> Path:
    sources_dir.mkdir(parents=True, exist_ok=True)
    captured_dir = sources_dir / "_captured"
    captured_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = _make_slug(text)
    filename = f"learn-{timestamp}-{slug}.md"
    filepath = captured_dir / filename

    tag_lines = "\n".join(f"  - {t}" for t in (tags or []))
    frontmatter = f"""---
type: learn
captured_at: {timestamp}
tags:
{tag_lines}
source: manual
---

{text.strip()}
"""
    filepath.write_text(frontmatter, encoding="utf-8")
    logger.info("Learned fact -> %s", filepath)
    return filepath


def _make_slug(text: str, max_len: int = 40) -> str:
    return _slugify_impl(text, max_length=max_len)[:max_len].strip("-")
