"""Parser registry — pluggable ingestion backends.

Each backend is a :class:`BaseParser` subclass plus a factory callable that
builds an instance from :class:`ParserBuildOptions`. Backends self-register at
importing the ``knox.preprocessing.parsers`` package triggers registration.

Used by ``knox list-parsers``, ``ParserRouter`` discovery, and
:func:`build_default_router` (the single factory for a ready-to-use router).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knox.core.config import PdfConfig
from knox.logging import get_logger
from knox.preprocessing.parsers.base import (
    IMAGE_SCALE_DEFAULT,
    BaseParser,
    ParserRouter,
)

logger = get_logger(__name__)

# Metadata powering ``knox list-parsers`` (name/type/extensions/figure_enrichment).
_registry: dict[str, dict[str, Any]] = {}

# Selectable ingestion backends: name → factory(options) -> BaseParser.
PARSER_BACKENDS: dict[str, Callable[[ParserBuildOptions], BaseParser]] = {}


@dataclass(frozen=True)
class ParserBuildOptions:
    """Inputs needed to construct a parser instance.

    Passed to every backend factory so each parser gets the same build context.
    """

    images_dir: Path | None = None
    image_mode: str = "file"
    images_scale: float = IMAGE_SCALE_DEFAULT
    pdf_config: PdfConfig | None = None


def register_backend(
    name: str,
    factory: Callable[[ParserBuildOptions], BaseParser],
    *,
    parser_cls: type[BaseParser],
    parser_type: str = "built-in",
) -> None:
    """Register an ingestion backend.

    Args:
        name: Backend identifier (e.g. ``"docling"``, ``"liteparse"``).
        factory: Builds a parser instance from :class:`ParserBuildOptions`.
        parser_cls: The parser class — ``supported_extensions`` and
            ``figure_enrichment`` are read off it as the single source of truth
            for the registry metadata.
        parser_type: Display category for ``list-parsers``.
    """
    PARSER_BACKENDS[name] = factory
    _register(
        name,
        parser_type,
        [str(ext) for ext in parser_cls.supported_extensions],
        parser_cls.figure_enrichment,
    )


def get_backend(name: str) -> Callable[[ParserBuildOptions], BaseParser]:
    """Return the factory for *name*. Raises ``ValueError`` when unknown."""
    try:
        return PARSER_BACKENDS[name]
    except KeyError:
        available = ", ".join(sorted(PARSER_BACKENDS)) or "(none)"
        raise ValueError(
            f"Unknown parser backend {name!r}. Available: {available}.",
        ) from None


def get_registered_parsers() -> list[dict[str, Any]]:
    """Return all registered parsers as a list of metadata dicts."""
    return list(_registry.values())


def get_all_extensions() -> set[str]:
    """Return the union of all registered parsers' supported extensions."""
    extensions: set[str] = set()
    for entry in _registry.values():
        for ext in entry.get("extensions", []):
            extensions.add(ext.lower())
    return extensions


def _register(
    name: str,
    parser_type: str,
    extensions: list[str],
    figure_enrichment: bool = False,
) -> None:
    """Populate the list-parsers metadata registry. Idempotent."""
    if name not in _registry:
        _registry[name] = {
            "name": name,
            "type": parser_type,
            "extensions": extensions,
            "figure_enrichment": figure_enrichment,
            "metadata": {},
        }


def _normalize_backends(names: str | Sequence[str]) -> list[str]:
    """Normalize a backend spec into a deduplicated, order-preserving list.

    Accepts a comma-separated string (``"liteparse,docling"``) or a sequence.
    Whitespace is trimmed, blanks dropped, duplicates removed. Falls back to
    ``["docling"]`` when nothing remains.
    """
    raw: Sequence[str] = names.split(",") if isinstance(names, str) else names
    seen: set[str] = set()
    result: list[str] = []
    for entry in raw:
        cleaned = entry.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result or ["docling"]


def build_default_router(
    options: ParserBuildOptions | None = None,
    *,
    names: str | Sequence[str] = "docling",
) -> ParserRouter:
    """Construct a :class:`ParserRouter` wired with an ordered backend chain.

    Backends are registered in the given priority order; ``pandoc`` is always
    appended last (when not already listed) and set as the universal
    binaryornot text fallback. The first backend whose
    :meth:`BaseParser.can_parse` matches a file wins, so list backends
    fastest-first while including every backend whose formats you need — e.g.
    ``"liteparse,docling"`` prefers the fast liteparse for PDF/DOCX yet keeps
    docling for HTML/EPUB/TEX/MD/CSV that liteparse cannot parse.

    Args:
        options: Build context (image mode/dir/scale, PDF config). Defaults to
            :class:`ParserBuildOptions` when ``None``.
        names: Ordered backend names — a comma-separated string
            (``"liteparse,docling"``) or a sequence. A single name
            (``"docling"``) is accepted for backward compatibility.
    """
    if options is None:
        options = ParserBuildOptions()

    ordered = _normalize_backends(names)
    router = ParserRouter()
    for backend_name in ordered:
        router.register(get_backend(backend_name)(options))

    # Pandoc is always present: legacy-format parser + universal text fallback.
    pandoc = get_backend("pandoc")(options)
    if "pandoc" not in ordered:
        router.register(pandoc)
    router.set_text_fallback(pandoc)

    return router
