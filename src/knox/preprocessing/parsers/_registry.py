"""Parser registry — tracks all registered parsers.

Used by ``knox list-parsers`` and ``ParserRouter`` for parser discovery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knox.logging import get_logger
from knox.preprocessing.parsers.base import ParserRouter
from knox.preprocessing.parsers.docling_parser import DoclingParser
from knox.preprocessing.parsers.pandoc_parser import PandocParser

logger = get_logger(__name__)

# In-memory registry of all parsers
_registry: dict[str, dict[str, Any]] = {}


def get_registered_parsers() -> list[dict[str, Any]]:
    """Return all registered parsers as a list of dicts."""
    return list(_registry.values())


def get_all_extensions() -> set[str]:
    """Return the union of all registered parsers' supported extensions."""
    extensions: set[str] = set()
    for entry in _registry.values():
        for ext in entry.get("extensions", []):
            extensions.add(ext.lower())
    return extensions


def _register(name: str, parser_type: str, extensions: list[str]) -> None:
    """Register a parser in the global registry. Idempotent."""
    if name not in _registry:
        _registry[name] = {
            "name": name,
            "type": parser_type,
            "extensions": extensions,
            "metadata": {},
        }


def build_default_router(images_dir: Path | None = None) -> ParserRouter:
    """Construct a :class:`ParserRouter` with all built-in parsers registered.

    This is the single factory for creating a ready-to-use router.
    The CLI and any programmatic user should call this instead of
    manually constructing and registering parsers.

    Args:
        images_dir: When provided, extracted images go under this directory
            instead of next to each source file. Typically ``<base>/.knox/cache/images``.
    """
    router = ParserRouter()
    docling = DoclingParser(images_dir=images_dir)
    router.register(docling)
    _register("DoclingParser", "built-in", list(docling.supported_extensions))

    pandoc = PandocParser()
    router.register(pandoc)
    _register("PandocParser", "built-in", list(pandoc.supported_extensions))
    router.set_text_fallback(pandoc)

    return router
