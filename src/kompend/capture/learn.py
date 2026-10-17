from __future__ import annotations

from pathlib import Path

from kompend.capture import TIMESTAMP_FORMAT
from kompend.capture._utils import format_sources_markdown, make_slug
from kompend.core.paths import LEARNED_SUBDIR
from kompend.core.utils.time import display_local
from kompend.logging import get_logger
from kompend.output.frontmatter import okf_frontmatter_yaml, source_frontmatter

logger = get_logger(__name__)


def learn(
    text: str,
    *,
    sources_dir: Path,
    tags: list[str] | None = None,
    attributes: dict[str, str] | None = None,
) -> Path:
    sources_dir.mkdir(parents=True, exist_ok=True)
    learned_dir = sources_dir / LEARNED_SUBDIR
    learned_dir.mkdir(parents=True, exist_ok=True)

    timestamp = display_local(TIMESTAMP_FORMAT)
    slug = make_slug(text)
    filename = f"learn-{timestamp}-{slug}.md"
    filepath = learned_dir / filename

    frontmatter = okf_frontmatter_yaml(
        source_frontmatter(
            page_type="learn",
            timestamp=timestamp,
            timestamp_key="captured_at",
            tags=list(tags or []),
            extra={
                "attributes": dict(attributes or {}),  # nocheck: frontmatter dict, not wiki metadata
                "source": "manual",  # nocheck: frontmatter dict, not wiki metadata
            },
        ),
    )

    filepath.write_text(f"{frontmatter}{text.strip()}\n", encoding="utf-8")
    logger.info("Learned fact -> %s", filepath)
    format_sources_markdown(sources_dir)
    return filepath
