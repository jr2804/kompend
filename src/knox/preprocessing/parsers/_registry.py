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

# ── Backend registry ───────────────────────────────────────────────────
PARSER_BACKENDS: dict[str, Callable[..., BaseParser]] = {}

# Every known backend name → its ``uv sync --group <name>`` install group.
# Lets ``build_default_router`` distinguish a typo (unknown name) from an
# uninstalled optional dependency (known name, not in ``PARSER_BACKENDS``).
KNOWN_BACKENDS: dict[str, str] = {
    "liteparse": "default",
    "docling": "docling",
    "pandoc": "pandoc",
    "web": "web",
    "marker": "marker",
    "mineru": "mineru",
    "glm-ocr": "glm-ocr",
}


@dataclass(frozen=True)
class ParserBuildOptions:
    """Inputs needed to construct a parser instance."""

    images_dir: Path | None = None
    image_mode: str = "ref"
    image_scale: float = IMAGE_SCALE_DEFAULT
    pdf_config: PdfConfig | None = None
    extra: dict[str, Any] | None = None


def register_backend(
    name: str,
    factory: Callable[..., BaseParser],
    extensions: Sequence[str] | None = None,
    *,
    is_text_fallback: bool = False,
) -> None:
    """Register a parser backend.

    Parameters
    ----------
    name:
        Canonical name used in config (e.g. ``"docling"``).
    factory:
        Callable returning a configured parser instance.
    extensions:
        File extensions this backend handles natively (with leading dot).
    is_text_fallback:
        If ``True``, this backend is used as the universal text fallback.
    """
    PARSER_BACKENDS[name] = factory
    if extensions is not None:
        for ext in extensions:
            ext = ext.lower()
            if not ext.startswith("."):
                ext = f".{ext}"
            _backend_extensions[name].add(ext)
    if is_text_fallback:
        _text_fallbacks.append(name)
    logger.debug("Registered parser backend: %s", name)


# ── Extension mapping ──────────────────────────────────────────────────
_backend_extensions: dict[str, set[str]] = {}
_text_fallbacks: list[str] = []


def get_extensions_for_backend(name: str) -> set[str]:
    """Return the set of file extensions registered for *name*."""
    return _backend_extensions.get(name, set())


def get_all_extensions() -> set[str]:
    """Return the union of all registered file extensions."""
    extensions: set[str] = set()
    for ext_set in _backend_extensions.values():
        extensions.update(ext_set)
    return extensions


def get_registered_parsers() -> dict[str, dict[str, Any]]:
    """Return metadata for all registered backends."""
    result: dict[str, dict[str, Any]] = {}
    for name, factory in PARSER_BACKENDS.items():
        result[name] = {
            "name": name,
            "factory": factory,
            "extensions": sorted(_backend_extensions.get(name, set())),
            "is_text_fallback": name in _text_fallbacks,
        }
    return result


# ── Router construction ────────────────────────────────────────────────


def build_default_router(
    options: ParserBuildOptions | None = None,
    names: str | Sequence[str] | None = None,
) -> ParserRouter:
    """Build a :class:`ParserRouter` from backend names.

    Uninstalled backends (known name but optional dependency missing) are
    skipped with a warning rather than raising. Genuinely unknown names raise
    ``ValueError``.

    Args:
        options: Build context (image mode/dir/scale, PDF config). Defaults to
            :class:`ParserBuildOptions` when ``None``.
        names: Ordered backend names — a comma-separated string
            (``"liteparse,docling"``) or a sequence.
    """
    if options is None:
        options = ParserBuildOptions()

    ordered = _normalize_backends(names) if names is not None else ["liteparse"]

    # Validate all names first — fail fast on typos.
    for backend_name in ordered:
        if backend_name not in KNOWN_BACKENDS:
            available = ", ".join(sorted(KNOWN_BACKENDS)) or "(none)"
            msg = (
                f"Unknown parser backend {backend_name!r}. "
                f"Available: {available}."
            )
            raise ValueError(msg)

    # Filter to installed backends, warn on missing.
    available = set(PARSER_BACKENDS)
    skipped = [n for n in ordered if n not in available]
    for n in skipped:
        hint = KNOWN_BACKENDS.get(n, "unknown backend")
        logger.warning(
            "Parser backend %r is not installed; skipping. Install with: uv sync --group %s",
            n,
            hint,
        )
    installed_order = [n for n in ordered if n in available]

    router = ParserRouter()
    for backend_name in installed_order:
        router.register(get_backend(backend_name)(options))

    # Pandoc is always present when installed: legacy-format parser + universal
    # text fallback.
    if "pandoc" in available:
        pandoc = get_backend("pandoc")(options)
        if "pandoc" not in installed_order:
            router.register(pandoc)
        router.set_text_fallback(pandoc)

    return router


def get_backend(name: str) -> Callable[[ParserBuildOptions], BaseParser]:
    """Return the factory for *name*. Raises ``ValueError`` when unknown."""
    try:
        return PARSER_BACKENDS[name]
    except KeyError:
        available = ", ".join(sorted(PARSER_BACKENDS)) or "(none)"
        raise ValueError(
            f"Unknown parser backend {name!r}. Available: {available}.",
        ) from None


def _normalize_backends(names: str | Sequence[str]) -> list[str]:
    """Normalize a backend spec into a deduplicated, order-preserving list.

    Accepts a comma-separated string (``"liteparse,docling"``) or a sequence.
    Whitespace is trimmed, blanks dropped, duplicates removed. Falls back to
    ``["liteparse"]`` when nothing remains.
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
