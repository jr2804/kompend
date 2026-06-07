"""Parser registry — tracks all registered parsers (built-in + external + HTTP).

Used by ``knox list-parsers`` and ``ParserRouter`` for parser discovery.
"""

from __future__ import annotations

from typing import Any

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


def build_default_router() -> ParserRouter:  # type: ignore[name-defined]  # noqa: F821
    """Construct a :class:`ParserRouter` with all built-in parsers registered.

    This is the single factory for creating a ready-to-use router.
    The CLI and any programmatic user should call this instead of
    manually constructing and registering parsers.
    """
    from knox.preprocessing.parsers.base import ParserRouter  # noqa: PLC0415

    router = ParserRouter()

    try:
        from knox.preprocessing.parsers.docling_parser import DoclingParser  # noqa: PLC0415

        docling = DoclingParser()
        router.register(docling)
        register_parser("DoclingParser", "built-in", list(docling.supported_extensions))
    except Exception as exc:
        logger.debug("DoclingParser not available: %s", exc)

    try:
        from knox.preprocessing.parsers.pdfplumber_parser import PdfPlumberParser  # noqa: PLC0415

        pdfplumber = PdfPlumberParser()
        router.register(pdfplumber)
        register_parser("PdfPlumberParser", "built-in", list(pdfplumber.supported_extensions))
    except Exception as exc:
        logger.debug("PdfPlumberParser not available: %s", exc)

    try:
        from knox.preprocessing.parsers.pandoc_parser import PandocParser  # noqa: PLC0415

        pandoc = PandocParser()
        router.register(pandoc)
        register_parser("PandocParser", "built-in", list(pandoc.supported_extensions))
    except Exception as exc:
        logger.debug("PandocParser not available: %s", exc)

    return router
