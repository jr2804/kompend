"""Shared utilities for the capture subsystem."""

from __future__ import annotations

from slugify import slugify as _slugify_impl

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
