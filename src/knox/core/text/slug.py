"""Canonical slug helper — the single wrapper over ``python-slugify``.

Centralises the dependency (and its alias) so every caller slugs text the same
way. ``default`` supplies the fallback for empty input; callers with a
different slug *rule* (e.g. the regex sluggers in ``agents.generation`` /
``organize.categorize``) deliberately do not use this, because their output
diverges from python-slugify's.
"""

from __future__ import annotations

from slugify import slugify as _slugify_impl


def slugify(text: str, *, default: str = "", max_length: int = 0) -> str:
    """Slugify *text* via python-slugify; return *default* when the result is empty.

    ``max_length=0`` (the default) means no length cap, matching python-slugify.
    """
    return _slugify_impl(text, max_length=max_length) or default
