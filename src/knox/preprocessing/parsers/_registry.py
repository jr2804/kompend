"""Parser registry — tracks all registered parsers.

Used by ``knox list-parsers`` and ``ParserRouter`` for parser discovery.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from knox.preprocessing.parsers.base import ParserRouter

try:
    from knox.preprocessing.parsers.docling_parser import DoclingParser
except ImportError:
    DoclingParser = None  # ty: ignore

try:
    from knox.preprocessing.parsers.pandoc_parser import PandocParser
except ImportError:
    PandocParser = None  # ty: ignore

logger = logging.getLogger(__name__)

# In-memory registry of all parsers
_registry: dict[str, dict[str, Any]] = {}


def register_parser(
    name: str,
    parser_type: str,  # "built-in", "external", "http"
    extensions: list[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Register a parser in the global registry.

    Idempotent — re-registering the same name is a no-op.

    Args:
        name: Parser name (class name or command name).
        parser_type: ``"built-in"``, ``"external"``, or ``"http"``.
        extensions: List of supported file extensions (with dot, e.g. ``[".xyz"]``).
        metadata: Optional extra info (command path, URL, etc.).
    """
    if name not in _registry:
        _registry[name] = {
            "name": name,
            "type": parser_type,
            "extensions": extensions,
            "metadata": metadata or {},
        }


def get_registered_parsers() -> list[dict[str, Any]]:
    """Return all registered parsers as a list of dicts."""
    return list(_registry.values())


def get_parser_names() -> list[str]:
    """Return names of all registered parsers."""
    return list(_registry.keys())


def clear_registry() -> None:
    """Clear the registry (useful for testing)."""
    _registry.clear()


def get_all_extensions() -> set[str]:
    """Return the union of all registered parsers' supported extensions."""
    extensions: set[str] = set()
    for entry in _registry.values():
        for ext in entry.get("extensions", []):
            extensions.add(ext.lower())
    return extensions


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

    if DoclingParser is not None:
        try:
            docling = DoclingParser(images_dir=images_dir)
            router.register(docling)
            register_parser("DoclingParser", "built-in", list(docling.supported_extensions))
        except (ImportError, OSError) as exc:
            logger.debug("DoclingParser not available: %s", exc, exc_info=True)

    if PandocParser is not None:
        try:
            pandoc = PandocParser()
            router.register(pandoc)
            register_parser("PandocParser", "built-in", list(pandoc.supported_extensions))
            # PandocParser also serves as the binaryornot text fallback.
            router.set_text_fallback(pandoc)
        except (ImportError, OSError) as exc:
            logger.debug("PandocParser not available: %s", exc, exc_info=True)

    return router
