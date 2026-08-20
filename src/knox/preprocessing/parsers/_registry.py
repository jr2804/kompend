"""Parser registry — pluggable ingestion backends.

Each backend is a :class:`BaseParser` subclass plus a factory callable that
builds an instance from :class:`ParserBuildOptions`. Backends self-register at
importing the ``knox.preprocessing.parsers`` package triggers registration.

Used by ``knox list-parsers``, ``ParserRouter`` discovery, and
:func:`build_default_router` (the single factory for a ready-to-use router).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from knox.core.config import PdfConfig
from knox.logging import get_logger
from knox.preprocessing.parsers.base import (
    IMAGE_SCALE_DEFAULT,
    BaseParser,
    ParserRouter,
)

# Selectable ingestion backends: name → factory(options) -> BaseParser.
PARSER_BACKENDS: dict[str, Callable[[ParserBuildOptions], BaseParser]] = {}

# Known backend names — always populated so unknown-name detection works even
# when the backend's optional dependency is not installed. Each entry maps to
# a human-readable description shown in error messages.
KNOWN_BACKENDS: dict[str, str] = {
    "docling": "high-accuracy PDF/EPUB/HTML parser (install: uv sync --extra docling)",
    "glm-ocr": "GLM-OCR image-based OCR for PNG/JPG/TIFF/BMP (install: uv pip install glmocr — conflicts with marker on pillow)",
    "liteparse": "fast PDF/office parser (installed by default)",
    "marker": "Marker high-accuracy PDF parser (install: uv pip install marker-pdf — conflicts with glmocr on pillow)",
    "mineru": "layout-aware PDF/Office parser with OCR (install: uv pip install mineru)",
    "pandoc": "legacy text formats — RST, LaTeX, ODT (install: uv sync --extra pandoc)",
    "web": "trafilatura HTML parser for web pages (installed by default)",
}

# Backend name → module path. The backend module's ``register_backend`` call
# runs only when the module is imported, so importing the ``parsers`` package
# itself stays cheap. These are loaded on demand by :func:`_ensure_backend`
# (from ``build_default_router`` / ``get_backend`` / ``get_all_extensions`` /
# ``get_registered_parsers``) — never at CLI ``--help`` import time.
_BACKEND_MODULES: dict[str, str] = {
    "docling": "knox.preprocessing.parsers.docling_parser",
    "glm-ocr": "knox.preprocessing.parsers.glm_ocr_parser",
    "liteparse": "knox.preprocessing.parsers.liteparse_parser",
    "marker": "knox.preprocessing.parsers.marker_parser",
    "mineru": "knox.preprocessing.parsers.mineru_parser",
    "pandoc": "knox.preprocessing.parsers.pandoc_parser",
    "web": "knox.preprocessing.parsers.web_parser",
}

logger = get_logger(__name__)


# Metadata powering ``knox list-parsers`` (name/type/extensions/figure_enrichment).
_registry: dict[str, ParserMetadata] = {}


class ParserMetadata(TypedDict):
    """Schema for one entry in the parser registry (``knox list-parsers``)."""

    name: str
    type: str
    extensions: list[str]
    figure_enrichment: bool
    metadata: dict[str, Any]


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


def get_registered_parsers() -> list[ParserMetadata]:
    """Return all registered parsers as a list of metadata dicts."""
    _ensure_backends(KNOWN_BACKENDS)
    return list(_registry.values())


def get_all_extensions() -> set[str]:
    """Return the union of all registered parsers' supported extensions."""
    _ensure_backends(KNOWN_BACKENDS)
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


def build_default_router(
    options: ParserBuildOptions | None = None,
    *,
    names: str | Sequence[str] = "liteparse",
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
    # Load the requested backends (and the always-on pandoc fallback) before
    # checking availability, so installed backends self-register on demand.
    _ensure_backends(set(ordered) | {"pandoc"})
    # Distinguish unknown backend names (typos) from known-but-uninstalled
    # optional backends. Unknown names raise immediately; uninstalled backends
    # are skipped with a warning so a partial install still works.
    available = set(PARSER_BACKENDS)
    for n in ordered:
        if n not in KNOWN_BACKENDS and n not in available:
            raise ValueError(
                f"Unknown parser backend {n!r}. Known: {', '.join(sorted(KNOWN_BACKENDS))}.",
            )
    skipped = [n for n in ordered if n not in available]
    for n in skipped:
        hint = KNOWN_BACKENDS.get(n, "unknown backend")
        logger.warning(
            "Parser backend %r is not installed; skipping. (%s)",
            n,
            hint,
        )
    installed_order = [n for n in ordered if n in available]

    router = ParserRouter()
    if installed_order:
        for backend_name in installed_order:
            router.register(get_backend(backend_name)(options))

    # Pandoc is always present: legacy-format parser + universal text fallback.
    if "pandoc" in available:
        pandoc = get_backend("pandoc")(options)
        if "pandoc" not in installed_order:
            router.register(pandoc)
        router.set_text_fallback(pandoc)

    return router


def _ensure_backends(names: Iterable[str]) -> None:
    """Ensure every backend in *names* is loaded (installed ones only)."""
    for n in names:
        _ensure_backend(n)


def get_backend(name: str) -> Callable[[ParserBuildOptions], BaseParser]:
    """Return the factory for *name*. Raises ``ValueError`` when unknown."""
    _ensure_backend(name)
    try:
        return PARSER_BACKENDS[name]
    except KeyError:
        available = ", ".join(sorted(PARSER_BACKENDS)) or "(none)"
        raise ValueError(
            f"Unknown parser backend {name!r}. Available: {available}.",
        ) from None


def _ensure_backend(name: str) -> None:
    """Import *name*'s backend module if not yet registered.

    Importing the module runs its ``register_backend`` call, populating
    ``PARSER_BACKENDS``. If the optional dependency is missing, the import
    fails and the backend simply stays unregistered.
    """
    if name in PARSER_BACKENDS:
        return
    module_name = _BACKEND_MODULES.get(name)
    if module_name is None:
        return
    try:
        importlib.import_module(module_name)
    except ImportError:  # pragma: no cover - optional dependency not installed
        return


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
    return result or ["liteparse"]
