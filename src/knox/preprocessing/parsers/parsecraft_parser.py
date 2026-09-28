"""parsecraft adapter — routes KNOX parsing through the external
`parsecraft <https://pypi.org/project/parsecraft/>`_ document-intelligence
pipeline.

KNOX consumes parsecraft's **public API only** (``parsecraft.pipeline`` /
``parsecraft.backends`` / IR). The dependency is pinned and installed as a
wheel — never the parsecraft checkout. See
``.agents/plans/parsecraft-integration.md`` for the integration contract and
the rationale for each mapping below.

Mapping summary:

- KNOX file → parsecraft MIME via ``media_type_for``; the MIME feeds
  ``RoutingConstraints.formats`` and analyzer selection.
- ``analyze_source`` → ``pipeline.execute`` → IR (``DocumentResult`` →
  ``PageResult.blocks``) → KNOX ``ParsedDocument`` / ``DocumentElement``.
- parsecraft's four error buckets → KNOX parse failures with the cause
  preserved. Never emit placeholder pages as content: on total exhaustion
  parsecraft returns diagnostic-carrying pages, not text.
- Route degradation (OCR → native) arrives as ``QualitySignal`` entries in
  ``DocumentResult.quality`` and is forwarded as KNOX metadata so it survives
  the parse cache.
"""

from __future__ import annotations

from pathlib import Path

from parsecraft.backends import BackendRegistry, default_registry
from parsecraft.backends.protocol import SourceDocument
from parsecraft.environment import (
    EnvironmentInfo,
    constraints_from_environment,
    probe_environment,
)
from parsecraft.ir.models import DocumentResult, StructuredChunk
from parsecraft.pipeline import (
    ALL_PASSES_FAILED_CODE,
    MEDIA_TYPES,
    PageGroup,
    UnsupportedSourceError,
    analyze_source,
    execute,
    media_type_for,
)

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.logging import get_logger
from knox.preprocessing.parsers._registry import (
    ParserBuildOptions,
    register_backend,
)

#: parsecraft chunk kind → KNOX element type. Kinds without a KNOX equivalent
#: (header/footer/quote/unknown/caption) map to TEXT — they carry content and
#: dropping them would lose data.
_CHUNK_KIND_TO_ELEMENT: dict[str, ElementType] = {
    "heading": ElementType.HEADING,
    "paragraph": ElementType.TEXT,
    "list": ElementType.LIST_ITEM,
    "table": ElementType.TABLE,
    "formula": ElementType.FORMULA,
    "code": ElementType.CODE,
    "figure": ElementType.IMAGE,
    "caption": ElementType.TEXT,
    "header": ElementType.TEXT,
    "footer": ElementType.TEXT,
    "quote": ElementType.TEXT,
    "unknown": ElementType.TEXT,
}

logger = get_logger(__name__)


class ParsecraftParser:
    """KNOX Tier-1 parser backed by the parsecraft pipeline.

    Deliberately not a :class:`~knox.preprocessing.parsers.base.BaseParser`
    subclass: it carries parsecraft-specific constructor state (the backend
    registry) that the KNOX base has no business knowing about. The registry
    binds it through an explicit factory, and ``supported_extensions`` /
    ``can_parse`` / ``parse`` satisfy the structural surface the router needs.
    """

    #: Derived from parsecraft's own capability table (``MEDIA_TYPES``), not
    #: hardcoded — discovery and routing cannot disagree.
    supported_extensions: set[str] = set(MEDIA_TYPES)

    figure_enrichment = True

    def __init__(self, registry: BackendRegistry | None = None) -> None:
        self._registry = registry
        self._environment: EnvironmentInfo | None = None

    def can_parse(self, source_file: Path) -> bool:  # noqa: PLR6301 — parser-instance surface
        """True when parsecraft's classifier knows the file's suffix."""
        try:
            media_type_for(source_file)
        except UnsupportedSourceError:
            return False
        return True

    def parse(self, source_file: Path) -> ParsedDocument:
        """Parse *source_file* through parsecraft and map the IR to KNOX."""
        registry = self._registry if self._registry is not None else default_registry
        media_type = media_type_for(source_file)
        # probe_environment shells out (nvidia-smi + find_spec sweep); the
        # answer cannot change mid-process, so compute it once per instance —
        # hundreds of vault files must not mean hundreds of subprocess spawns.
        if self._environment is None:
            self._environment = probe_environment()
        constraints = constraints_from_environment(self._environment, formats={media_type})

        source = SourceDocument(uri=source_file.resolve().as_uri())
        analysis = analyze_source(
            source,
            registry,
            media_type=media_type,
            installed_extras=constraints.installed_extras,
        )
        result = execute(analysis, registry, constraints, source)

        document = result.document
        elements = _map_document(document, source_file)

        return ParsedDocument(
            source_file=source_file,
            elements=elements,
            title=document.metadata.title or source_file.stem,
            author="",
            page_count=len(document.pages),
            metadata={
                "parser": "parsecraft",
                "backends": _winning_backends(result.groups),
                "quality_signals": [signal.name for signal in document.quality],
                "degraded_pages": sorted(
                    signal.page_number for signal in document.quality if signal.name.startswith("degraded-") and signal.page_number is not None
                ),
            },
        )


class ParsecraftParseError(RuntimeError):
    """parsecraft produced no usable content for a source file.

    Raised on total exhaustion (every pass failed for every page) or when the
    IR carries no content blocks. Dependency problems are NOT folded in here:
    parsecraft raises its own typed errors (``DependencyUnavailableError``,
    ``NoAnalyzerError``) and those propagate as-is so callers can tell
    "install an extra" from "bad file".
    """

    def __init__(self, source_file: Path, document: DocumentResult) -> None:
        failures = [
            f"p{page.page_number}: {diagnostic.message}"
            for page in document.pages
            for diagnostic in page.diagnostics
            if diagnostic.code == ALL_PASSES_FAILED_CODE
        ]
        degradations = [f"p{signal.page_number} {signal.name}: {signal.detail}" for signal in document.quality if signal.name.startswith("degraded-")]
        message = "; ".join(failures or degradations) or "all parsecraft passes failed"
        super().__init__(f"{source_file.name}: {message}")


def _map_document(document: DocumentResult, source_file: Path) -> list[DocumentElement]:
    """Map parsecraft IR pages/chunks to KNOX elements, order + pages preserved.

    Raises :class:`ParsecraftParseError` when execution genuinely exhausted —
    discriminated by execution evidence, NOT by ``winner is None`` (a parsecraft
    cache hit also yields ``winner=None`` with plan-shaped ``attempts == []``
    groups and a perfectly valid document).
    """
    exhausted = {page.page_number for page in document.pages if any(diagnostic.code == ALL_PASSES_FAILED_CODE for diagnostic in page.diagnostics)}
    if exhausted and exhausted == {page.page_number for page in document.pages}:
        raise ParsecraftParseError(source_file, document)

    elements: list[DocumentElement] = []
    for page in document.pages:
        for chunk in page.blocks:
            element = _chunk_to_element(chunk, source_file, len(elements))
            if element is not None:
                elements.append(element)
    if not elements:
        # A valid parsecraft document always carries at least one block; zero
        # content is a failure regardless of whether diagnostics were attached.
        raise ParsecraftParseError(source_file, document)
    return elements


def _chunk_to_element(
    chunk: StructuredChunk,
    source_file: Path,
    reading_order: int,
) -> DocumentElement | None:
    """One IR chunk → one KNOX element, or ``None`` when it carries no content."""
    element_type = _CHUNK_KIND_TO_ELEMENT.get(chunk.kind, ElementType.TEXT)
    content = _chunk_content(chunk)
    if not content:
        return None
    return DocumentElement(
        type=element_type,
        content=content,
        page=chunk.page_number,
        reading_order=reading_order,
        heading_level=_heading_level(chunk),
        metadata={
            "parsecraft_kind": chunk.kind.value,
            "confidence": chunk.confidence,
        },
        source_file=source_file,
    )


def _chunk_content(chunk: StructuredChunk) -> str:
    """Chunk content in KNOX's element shape.

    Tables arrive as Markdown text: parsecraft's IR has no structured row
    model yet (upstream gap), so KNOX consumes the Markdown table — its own
    markdown converter restores the grid downstream.
    """
    return chunk.content if isinstance(chunk.content, str) else ""


def _heading_level(chunk: StructuredChunk) -> int | None:
    """Heading depth for heading chunks.

    parsecraft carries the level in metadata as a string (``"1"``, ``"2"``).
    """
    if chunk.kind != "heading":
        return None
    level = chunk.metadata.get("heading_level")
    if level is None:
        return None
    try:
        return int(level)
    except ValueError:
        return None


def _winning_backends(groups: list[PageGroup]) -> str:
    """Comma-separated backends that won page groups — parse provenance."""
    names = {group.winner for group in groups if group.winner is not None}
    return ",".join(sorted(names))


def _factory(options: ParserBuildOptions) -> ParsecraftParser:  # noqa: ARG001 — uniform backend signature
    return ParsecraftParser()


register_backend("parsecraft", _factory, parser_cls=ParsecraftParser)  # ty: ignore[invalid-argument-type]
