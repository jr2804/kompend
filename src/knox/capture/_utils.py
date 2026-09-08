"""Shared utilities for the capture subsystem."""

from __future__ import annotations

from pathlib import Path

from slugify import slugify as _slugify_impl

from knox.core.config.rumdl import resolve_rumdl_config
from knox.core.paths import CAPTURED_SUBDIR, DISTILLED_SUBDIR, LEARNED_SUBDIR
from knox.output.renderer import format_output

MAX_TITLE_LENGTH: int = 60
MAX_SLUG_LENGTH: int = 40
MAX_TITLE_WORDS: int = 8


def make_title(text: str, max_len: int = MAX_TITLE_LENGTH) -> str:
    """Build a human-readable title from the first few words of *text*."""
    words = text.strip().split()[:MAX_TITLE_WORDS]
    title = " ".join(words)
    return title[:max_len].rstrip("-")


def make_slug(text: str, max_len: int = MAX_SLUG_LENGTH) -> str:
    """Build a URL-safe slug from *text*."""
    return _slugify_impl(text, max_length=max_len)[:max_len].strip("-")


def format_sources_markdown(sources_dir: Path) -> None:
    """Run rumdl fmt over captured/distilled markdown.

    Single source of truth with the compile path: reuses the same
    :func:`knox.output.renderer.format_output` + ``resolve_rumdl_config``
    pair that ``knox lint format`` and post-compile use. Best-effort —
    skips silently when rumdl is not installed; never raises.
    """
    rumdl_config = resolve_rumdl_config(sources_dir.parent)
    for sub in (CAPTURED_SUBDIR, LEARNED_SUBDIR, DISTILLED_SUBDIR):
        directory = sources_dir / sub
        if directory.is_dir():
            format_output(directory, rumdl_config=rumdl_config)
