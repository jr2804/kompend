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
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, TypedDict

from kompend.core.config import ImageMode, ParsecraftEffort, ParsecraftJudge, ParsecraftPreference, ResolvedConfig
from kompend.core.verbosity import is_verbose
from kompend.logging import get_logger
from kompend.preprocessing.parsers.base import (
    IMAGE_SCALE_DEFAULT,
    BaseParser,
    FileExtension,
    ParserRouter,
)


class ParserName(StrEnum):
    """Kompend-side ingestion backend identifiers — the registry vocabulary."""

    PARSECRAFT = "parsecraft"


#: The one ingestion backend kompend ships with. Single source for the registry
#: factory default and discovery — a second literal would let them disagree
#: silently. There is no config key for this: parsecraft routes internally, so
#: there is nothing left for a selector to select between.
DEFAULT_PARSER: Final[ParserName] = ParserName.PARSECRAFT

# Selectable ingestion backends: name → factory(options) -> BaseParser.
PARSER_BACKENDS: dict[ParserName, Callable[[ParserBuildOptions], BaseParser]] = {}

# Known backend names — always populated so unknown-name detection works even
# when the backend's optional dependency is not installed. Each entry maps to
# a human-readable description shown in error messages.
KNOWN_BACKENDS: dict[ParserName, str] = {
    ParserName.PARSECRAFT: "per-page routing across docling, liteparse, pandoc, native and OCR backends",
}

# Backend name → module path. The backend module's ``register_backend`` call
# runs only when the module is imported, so importing the ``parsers`` package
# itself stays cheap. These are loaded on demand by :func:`_ensure_backend`
# (from ``build_default_router`` / ``get_backend`` / ``get_all_extensions`` /
# ``get_registered_parsers``) — never at CLI ``--help`` import time.
_BACKEND_MODULES: dict[ParserName, str] = {
    ParserName.PARSECRAFT: "kompend.preprocessing.parsers.parsecraft_parser",
}

logger = get_logger(__name__)


# Metadata powering ``kompend list-parsers`` (name/type/extensions/figure_enrichment).
_registry: dict[ParserName, ParserMetadata] = {}


@dataclass(frozen=True)
class ParserBuildOptions:
    """Inputs needed to construct a parser instance.

    Passed to every backend factory so each parser gets the same build context.
    """

    images_dir: Path | None = None
    image_mode: ImageMode = ImageMode.FILE
    images_scale: float = IMAGE_SCALE_DEFAULT
    #: Judge-tier routing for the parsecraft backend. `none` leaves the
    #: deterministic rule table in place. Any non-`none` value is the SDK
    #: provider name (`ollama`, `zen`, `typesafe-ai`).
    parsecraft_judge: ParsecraftJudge = ParsecraftJudge.NONE
    #: Provider-specific model token for the judge (e.g. `parable-tinyjev` for
    #: ollama). Empty when the judge is `none`.
    parsecraft_judge_model: str = ""
    #: Ranking axis: `speed` | `balanced` | `quality`. Maps to
    #: parsecraft's `RoutingPreference` and threads through every judge call.
    parsecraft_preference: ParsecraftPreference = ParsecraftPreference.BALANCED
    #: MinerU conversion effort: `flash` | `medium` | `high` | `xhigh`. Maps to
    #: parsecraft's `BackendConfig.options["effort"]` and is injected only when
    #: non-default, because parsecraft hashes the options dict verbatim into the
    #: conversion cache key.
    parsecraft_effort: ParsecraftEffort = ParsecraftEffort.FLASH
    #: Whether parsecraft may lead a page with an OCR sub-backend. `None`
    #: (default) keeps parsecraft's own derivation from the installed extras and
    #: the host's usable engines; `False` bans OCR, `True` permits it.
    parsecraft_allow_ocr: bool | None = None
    #: External engines parsecraft may use, overriding the probed set. Empty
    #: keeps the probe's answer.
    parsecraft_engines: list[str] = field(default_factory=list)
    #: Whether liteparse recovers table structure from a page's vector graphics.
    #: `None` leaves the backend option unset.
    parsecraft_extract_vector_graphics: bool | None = None
    #: Backend ranked first where eligible; a preference, never a guarantee.
    parsecraft_preferred_backend: str | None = None
    #: Backend used exclusively; a hard pin, unreachable pages fail loudly.
    parsecraft_only_backend: str | None = None
    #: Per-backend options (`{backend: {option: value}}`) forwarded verbatim to
    #: parsecraft's `RoutingConstraints.backend_options`. Empty keeps the
    #: adapter's own derived options untouched.
    parsecraft_backend_options: dict[str, dict[str, str | int | float | bool]] = field(default_factory=dict)

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
            parsecraft_effort=kompend.ingest.parsecraft_effort,
            parsecraft_allow_ocr=kompend.ingest.parsecraft_allow_ocr,
            parsecraft_engines=list(kompend.ingest.parsecraft_engines),
            parsecraft_extract_vector_graphics=kompend.ingest.parsecraft_extract_vector_graphics,
            parsecraft_preferred_backend=kompend.ingest.parsecraft_preferred_backend,
            parsecraft_only_backend=kompend.ingest.parsecraft_only_backend,
            parsecraft_backend_options={backend: dict(options) for backend, options in kompend.ingest.parsecraft_backend_options.items()},
        )


class ParserMetadata(TypedDict):
    """Schema for one entry in the parser registry (``kompend list-parsers``)."""

    name: str
    type: str
    extensions: list[str]
    figure_enrichment: bool
    metadata: dict[str, Any]


def register_backend(
    name: ParserName,
    factory: Callable[[ParserBuildOptions], BaseParser],
    *,
    parser_cls: type[BaseParser],
    parser_type: str = "built-in",
) -> None:
    """Register an ingestion backend.

    Args:
        name: Backend identifier (a :class:`ParserName` member).
        factory: Builds a parser instance from :class:`ParserBuildOptions`.
        parser_cls: The parser class — ``supported_extensions`` and
            ``figure_enrichment`` are read off it as the single source of truth
            for the registry metadata.
        parser_type: Display category for ``list-parsers``.

    Raises:
        ValueError: *name* is not a :class:`ParserName` member — the registry
            vocabulary is closed; a new backend adds a member first.
    """
    key = name if isinstance(name, ParserName) else ParserName(name)
    PARSER_BACKENDS[key] = factory
    _register(
        key,
        parser_type,
        [str(ext) for ext in parser_cls.supported_extensions],
        parser_cls.figure_enrichment,
    )


def get_registered_parsers() -> list[ParserMetadata]:
    """Return all registered parsers as a list of metadata dicts."""
    _ensure_backends(KNOWN_BACKENDS)
    return list(_registry.values())


def get_supported_extensions() -> set[str]:
    """The file extensions a vault can ingest.

    Merges the canonical :class:`~kompend.preprocessing.parsers.base.FileExtension`
    table with the suffixes the ingestion backend declares. Discovery, the
    status report and the parsers themselves must agree on this set.
    """
    extensions: set[str] = set(FileExtension)
    try:
        extensions.update(get_all_extensions([DEFAULT_PARSER]))
    except Exception as exc:
        logger.debug("Could not load external parser extensions: %s", exc, exc_info=is_verbose())
    return extensions


def get_all_extensions(names: Iterable[ParserName | str]) -> set[str]:
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
    name: ParserName,
    parser_type: str,
    extensions: list[str],
    figure_enrichment: bool = False,
) -> None:
    """Populate the list-parsers metadata registry. Idempotent."""
    if name not in _registry:
        _registry[name] = {
            "name": str(name),
            "type": parser_type,
            "extensions": extensions,
            "figure_enrichment": figure_enrichment,
            "metadata": {},
        }


def build_default_router(
    options: ParserBuildOptions | None = None,
    *,
    name: ParserName | str = DEFAULT_PARSER,
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
            known = ", ".join(sorted(str(k) for k in KNOWN_BACKENDS))
            raise ValueError(f"Unknown parser backend {str(name)!r}. Known: {known}.")
        raise ValueError(f"Parser backend {str(name)!r} is not installed. {hint}")

    router = ParserRouter()
    router.register(get_backend(name)(options))
    return router


def _ensure_backends(names: Iterable[ParserName | str]) -> None:
    """Ensure every backend in *names* is loaded (installed ones only)."""
    for n in names:
        _ensure_backend(n)


def get_backend(name: ParserName | str) -> Callable[[ParserBuildOptions], BaseParser]:
    """Return the factory for *name*. Raises ``ValueError`` when unknown."""
    _ensure_backend(name)
    key: ParserName | None
    try:
        # Coerce raw-string callers onto the enum; members pass through.
        key = ParserName(name)
    except ValueError:
        key = None
    if key is None or key not in PARSER_BACKENDS:
        available = ", ".join(sorted(str(k) for k in PARSER_BACKENDS)) or "(none)"
        raise ValueError(
            f"Unknown parser backend {str(name)!r}. Available: {available}.",
        ) from None
    return PARSER_BACKENDS[key]


def _ensure_backend(name: ParserName | str) -> None:
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
