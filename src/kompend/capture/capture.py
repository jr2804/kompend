from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kompend.agents.capture import CaptureAgent
from kompend.agents.deps import PipelineAgentDeps
from kompend.capture import TIMESTAMP_FORMAT
from kompend.capture._utils import format_sources_markdown, make_slug, make_title
from kompend.capture.kinds import CAPTURE_SOURCE_ID, CaptureKind, SourceZone
from kompend.core.models.enums import MetadataField
from kompend.core.paths import CAPTURED_SUBDIR
from kompend.core.text.pipes import escape_prose_pipes
from kompend.core.utils.exceptions import LLMOutputError
from kompend.core.utils.time import display_local
from kompend.logging import get_logger
from kompend.output.frontmatter import okf_frontmatter_yaml, source_frontmatter

logger = get_logger(__name__)


@dataclass
class CapturedItem:
    kind: CaptureKind
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
        else [CapturedItem(kind=CaptureKind.NOTE, title=make_title(text), content=text, tags=user_tags)]
    )

    paths: list[Path] = []
    timestamp = display_local(TIMESTAMP_FORMAT)
    for i, item in enumerate(items):
        slug = make_slug(item.title)
        filename = f"capture-{timestamp}-{i:02d}-{slug}.md"
        filepath = captured_dir / filename

        frontmatter = okf_frontmatter_yaml(
            source_frontmatter(
                page_type=item.kind,
                title=item.title,
                timestamp=timestamp,
                timestamp_key=MetadataField.CAPTURED_AT,
                tags=list(item.tags),
                extra={
                    "attributes": dict(attributes or {}),  # nocheck: frontmatter dict, not wiki metadata
                    MetadataField.SOURCE: SourceZone.CAPTURE,
                },
            ),
        )

        content = escape_prose_pipes(item.content.strip())
        filepath.write_text(f"{frontmatter}# {item.title}\n\n{content}\n", encoding="utf-8")
        logger.info("Captured %s -> %s", item.kind, filepath)
        paths.append(filepath)

    format_sources_markdown(sources_dir)
    return paths


def _extract_with_agent(text: str, capture_agent: CaptureAgent, user_tags: list[str]) -> list[CapturedItem]:
    """Use the CaptureAgent to extract structured items from text.

    Returns an empty list when the agent refuses the input (nonsense,
    nothing extractable) — the caller warns and writes nothing. Falls back
    to a raw note only when the agent's own LLM work fails
    (:class:`~kompend.core.utils.exceptions.LLMOutputError`): the text is
    still worth keeping, and the caller's tags ride along so they are not
    silently lost. Any other exception is a bug in the capture path, not a
    degraded capture, and escapes.
    """
    tag_hint = f"\n\nThe user suggests these tags as a baseline (add more if relevant, keep these): {user_tags}." if user_tags else ""
    user_prompt = f"Extract knowledge items from this text:{tag_hint}\n\n{text}"
    deps = PipelineAgentDeps(text=user_prompt, source_id=CAPTURE_SOURCE_ID)

    try:
        result = capture_agent.run(deps)
    except LLMOutputError as exc:
        logger.warning("CaptureAgent failed, keeping the text as one raw note: %s", exc, exc_info=True)
        return [CapturedItem(kind=CaptureKind.NOTE, title=make_title(text), content=text, tags=list(user_tags))]

    if result.refused:
        reason = result.refusal_reason or "the input contains nothing extractable"
        logger.warning("CaptureAgent refused input: %s", reason)
        return []

    items: list[CapturedItem] = []
    for raw in result.items:
        items.append(
            CapturedItem(
                kind=CaptureKind.of(raw.kind),
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
