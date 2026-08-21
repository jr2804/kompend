from __future__ import annotations

from pathlib import Path

from knox.capture import TIMESTAMP_FORMAT
from knox.capture._utils import format_sources_markdown, make_slug
from knox.core.paths import CAPTURED_SUBDIR
from knox.core.utils.time import display_local
from knox.logging import get_logger

logger = get_logger(__name__)


def learn(
    text: str,
    *,
    sources_dir: Path,
    tags: list[str] | None = None,
) -> Path:
    sources_dir.mkdir(parents=True, exist_ok=True)
    captured_dir = sources_dir / CAPTURED_SUBDIR
    captured_dir.mkdir(parents=True, exist_ok=True)

    timestamp = display_local(TIMESTAMP_FORMAT)
    slug = make_slug(text)
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
    format_sources_markdown(sources_dir)
    return filepath
