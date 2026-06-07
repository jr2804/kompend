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
