"""Parser registry — pluggable ingestion backends.

Each backend is a :class:`BaseParser` subclass plus a factory callable that
builds an instance from :class:`ParserBuildOptions`. Backends self-register at
importing the ``kompend.preprocessing.parsers`` package triggers registration.

Used by ``kompend list-parsers``, ``ParserRouter`` discovery, and
:func:`build_default_router` (the single factory for a ready-to-use router).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from kompend.core.config import ResolvedConfig
from kompend.core.config.keys import DEFAULT_PARSER
from kompend.preprocessing.parsers.base import (
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
    "parsecraft": "per-page routing across docling, liteparse, pandoc, native and OCR backends (install: uv sync --extra parsecraft)",
}

# Backend name → module path. The backend module's ``register_backend`` call
# runs only when the module is imported, so importing the ``parsers`` package
# itself stays cheap. These are loaded on demand by :func:`_ensure_backend`
# (from ``build_default_router`` / ``get_backend`` / ``get_all_extensions`` /
# ``get_registered_parsers``) — never at CLI ``--help`` import time.
_BACKEND_MODULES: dict[str, str] = {
    "parsecraft": "kompend.preprocessing.parsers.parsecraft_parser",
}


# Metadata powering ``kompend list-parsers`` (name/type/extensions/figure_enrichment).
_registry: dict[str, ParserMetadata] = {}


@dataclass(frozen=True)
class ParserBuildOptions:
    """Inputs needed to construct a parser instance.

    Passed to every backend factory so each parser gets the same build context.
    """

    images_dir: Path | None = None
    image_mode: str = "file"
    images_scale: float = IMAGE_SCALE_DEFAULT
    #: Judge-tier routing for the parsecraft backend. `none` leaves the
    #: deterministic rule table in place. Any non-`none` value is the SDK
    #: provider name (`ollama`, `zen`, `typesafe-ai`).
    parsecraft_judge: str = "none"
    #: Provider-specific model token for the judge (e.g. `parable-tinyjev` for
    #: ollama). Empty when the judge is `none`.
    parsecraft_judge_model: str = ""
    #: Ranking axis: `speed` | `balanced` | `quality`. Maps to
    #: parsecraft's `RoutingPreference` and threads through every judge call.
    parsecraft_preference: str = "balanced"

    @classmethod
    def from_config(cls, cfg: ResolvedConfig, *, images_dir: Path | None = None) -> ParserBuildOptions:
        """Build the parser options implied by a resolved kompend config.

        The single construction point for every ingestion router the pipeline,
        the CLI and the MCP server build. They all share one parse cache, and
        that cache's key does not cover these options, so a site that built a
        narrower set could serve a document produced under different settings:
        ``kompend ingest`` once passed only ``images_dir`` while ``kompend compile``
        also passed the config-derived ``image_mode``.

        ``images_scale`` is not a config key, so every caller takes the field
        default. ``images_dir`` is a parameter because it derives from the vault
        root, which this method does not receive.
        """
        kompend = cfg.kompend
        return cls(
            images_dir=images_dir,
            image_mode=kompend.compile.image_mode,
            parsecraft_judge=kompend.ingest.parsecraft_judge,
            parsecraft_judge_model=kompend.ingest.parsecraft_judge_model,
            parsecraft_preference=kompend.ingest.parsecraft_preference,
        )


class ParserMetadata(TypedDict):
    """Schema for one entry in the parser registry (``kompend list-parsers``)."""

    name: str
    type: str
    extensions: list[str]
    figure_enrichment: bool
    metadata: dict[str, Any]


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


def get_all_extensions(names: Iterable[str]) -> set[str]:
    """Union of the suffixes the named backends can ingest.

    Discovery follows the backend actually configured (``[ingest] parser``),
    not every name the registry knows: a vault should only ingest what its
    parser can read. Unknown or uninstalled names contribute nothing.
    """
    _ensure_backends(names)
    extensions: set[str] = set()
    for name in names:
        entry = _registry.get(name)
        if entry is None:
            continue
        extensions.update(ext.lower() for ext in entry.get("extensions", []))
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
    name: str = DEFAULT_PARSER,
) -> ParserRouter:
    """Construct a :class:`ParserRouter` wired with the single ingestion backend.

    Exactly one backend is registered — the one named by ``[ingest] parser``.
    parsecraft performs its own per-page routing across its sub-backends, so
    kompend keeps no second, competing level of chain ordering or hand-off.

    Args:
        options: Build context handed to the backend factory. Defaults to
            :class:`ParserBuildOptions` when ``None``.
        name: Backend name.

    Raises:
        ValueError: *name* is unknown, or a known backend whose optional
            dependency is not installed. The message names the install command
            — a vault whose configured parser is missing must fail loudly, not
            start with an empty router.
    """
    if options is None:
        options = ParserBuildOptions()

    _ensure_backend(name)
    if name not in PARSER_BACKENDS:
        hint = KNOWN_BACKENDS.get(name)
        if hint is None:
            raise ValueError(f"Unknown parser backend {name!r}. Known: {', '.join(sorted(KNOWN_BACKENDS))}.")
        raise ValueError(f"Parser backend {name!r} is not installed. {hint}")

    router = ParserRouter()
    router.register(get_backend(name)(options))
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
