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
    PipelineResult,
    UnsupportedSourceError,
    analyze_source,
    execute,
    media_type_for,
)
from parsecraft.pipeline.analysis import AnalysisResult
from parsecraft.routing import NoEligibleBackendError, RoutingConstraints, RoutingError
from parsecraft.routing.judge import DeterministicJudge, RoutingJudge
from parsecraft.routing.judge_providers import JudgeError, resolve_judge
from parsecraft.routing.models import RoutingPreference

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


#: Characters of a judge/SDK error message kept in a log line. The SDK wraps
#: provider errors with the full HTTP exchange; the first clause names the
#: cause and the rest is noise in a per-file warning.
_JUDGE_ERROR_EXCERPT_CHARS = 160


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

    def __init__(
        self,
        registry: BackendRegistry | None = None,
        *,
        parsecraft_judge: str = "none",
        parsecraft_judge_model: str = "",
        parsecraft_preference: str = "balanced",
    ) -> None:
        self._registry = registry
        self._environment: EnvironmentInfo | None = None
        #: Routing inputs — the same three keys the config exposes under
        #: ``[ingest] parsecraft_*``. ``"none"`` keeps the deterministic rule
        #: table; any other value resolves to a real :class:`RoutingJudge` once
        #: ``parse()`` runs (and only then, so a misconfigured provider never
        #: costs anything at CLI ``--help`` time).
        self._judge_name = parsecraft_judge
        self._judge_model = parsecraft_judge_model
        self._judge_preference = parsecraft_preference
        self._judge: RoutingJudge | None = None
        self._judge_fallback: RoutingJudge | None = None  # DeterministicJudge after a real judge failed
        self._judge_degraded = False  # live judge dropped this run (warn once, no retry)

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
        result = self._execute_with_judge(analysis, registry, constraints, source)

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

    def cache_identity(self) -> str:
        """Routing inputs that change this backend's output without changing its class.

        The router folds this into :meth:`ParserRouter.fingerprint`, so the parse
        cache invalidates when the judge tier changes rather than silently
        serving pages routed under the previous configuration. Returns an empty
        string when the deterministic rule table is in use — the default
        configuration's cache key must not shift.
        """
        if self._judge_name in ("", "none"):
            return ""
        return f"parsecraft_judge:{self._judge_name}/{self._judge_model}:{self._judge_preference}"

    def _execute_with_judge(
        self,
        analysis: AnalysisResult,
        registry: BackendRegistry,
        constraints: RoutingConstraints,
        source: SourceDocument,
    ) -> PipelineResult:
        """Run ``execute`` with the configured judge, degrading to the rule table once.

        ``resolve_judge`` validates the *spec*, never the *endpoint*: the cloud
        providers accept any model token and only fail on the first ``rank``
        call (``zen/does-not-exist`` 400s from opencode.ai; a completion-only
        ollama model 400s from the daemon). ``plan_route`` calls
        ``RoutingJudge.rank`` without a handler, so that failure reaches the
        caller as a ``RoutingError``. A dead judge endpoint must cost a
        degraded plan, not a failed ingest — therefore: retry exactly once
        with the deterministic rule table, then remember the verdict for the
        rest of the run so every later file skips the doomed call.

        ``NoEligibleBackendError`` is re-raised: no judge can rank candidates
        that do not exist, so it is a genuine routing dead-end and the caller
        must see it (this is why the handler cannot simply catch
        ``RoutingError``). A failure from the deterministic judge is re-raised
        for the same reason — the retry is not a general-purpose error
        absorber.
        """
        judge = self._get_judge()
        try:
            return execute(analysis, registry, constraints, source, judge=judge)
        except NoEligibleBackendError:
            raise
        except RoutingError as exc:
            if self._judge_degraded or isinstance(judge, DeterministicJudge):
                raise
            self._degrade_judge(exc)
            fallback = self._judge_fallback if self._judge_fallback is not None else DeterministicJudge()
            return execute(analysis, registry, constraints, source, judge=fallback)

    def _degrade_judge(self, exc: BaseException) -> None:
        """Drop a live judge that failed at call time; warn once per run."""
        self._judge = None
        self._judge_fallback = DeterministicJudge()
        self._judge_degraded = True
        logger.warning(
            "parsecraft judge %s/%s failed at call time (%s: %s); falling back to the deterministic rule table for the rest of this run.",
            self._judge_name,
            self._judge_model,
            type(exc).__name__,
            str(exc)[:_JUDGE_ERROR_EXCERPT_CHARS],
        )

    def _get_judge(self) -> RoutingJudge:
        """Return the routing judge for this parser, resolving it once per instance.

        Three exit paths, in order:

        1. ``parsecraft_judge == "none"`` (default): return a fresh
           :class:`DeterministicJudge` every call. Cheap (in-process rule
           table) and avoids sharing state between documents.
        2. ``parsecraft_judge`` was configured: resolve via the SDK's
           :func:`resolve_judge`, cache the verdict. A second parse reuses
           the same judge instance, so the SDK's intra-instance memo (per
           page class) survives across the run.
        3. Resolver raised (spec error, missing credential, unreachable
           endpoint, missing `decision` capability on the ollama model):
           warn once and fall back to :class:`DeterministicJudge`. A dead
           judge endpoint must never fail an ingest — the fallback is a
           degraded parse, not a hard error.

        Resolution is only half the story: ``resolve_judge`` validates the
        *spec*, never the *endpoint* (a cloud provider accepts any model token
        and 400s on the first call). The call-time failure is handled by
        :meth:`_execute_with_judge`.
        """
        if self._judge is not None:
            return self._judge
        if self._judge_fallback is not None:
            return self._judge_fallback
        if self._judge_name in ("", "none"):
            self._judge_fallback = DeterministicJudge()
            return self._judge_fallback
        if not self._judge_model:
            msg = (
                f"parsecraft_judge is {self._judge_name!r} but parsecraft_judge_model is empty; "
                "falling back to the deterministic rule table. Set [ingest] parsecraft_judge_model "
                "to the SDK's model token for the configured provider."
            )
            logger.warning(msg)
            self._judge_fallback = DeterministicJudge()
            return self._judge_fallback
        try:
            self._judge = resolve_judge(
                f"{self._judge_name}/{self._judge_model}",
                preference=_parse_preference(self._judge_preference),
            )
        except JudgeError as exc:
            # Every resolution failure is a JudgeError subclass: JudgeSpecError
            # (malformed spec), JudgeProviderUnavailableError (no loader / missing
            # credential), JudgeProviderLoadError (extra not installed).
            logger.warning(
                "parsecraft judge %s/%s could not be resolved (%s: %s); falling back to the deterministic rule table.",
                self._judge_name,
                self._judge_model,
                type(exc).__name__,
                str(exc)[:_JUDGE_ERROR_EXCERPT_CHARS],
            )
            self._judge_fallback = DeterministicJudge()
            return self._judge_fallback
        return self._judge


def _parse_preference(name: str) -> RoutingPreference:
    """Map a config string to parsecraft's ``RoutingPreference``.

    Accepts the same lowercase set the config validates against; an
    unrecognised value falls through to BALANCED, never errors — a typo
    in the config should not block a parse run.
    """
    try:
        return RoutingPreference(name)
    except ValueError:
        return RoutingPreference.BALANCED


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

    parsecraft carries the level in chunk metadata as a string (``"1"``,
    ``"2"``). 2026.10.2 renamed the key to ``n_level`` (the native-markdown
    adapter emits ``token.tag.removeprefix("h")``); ``heading_level`` is the
    2026.10.1 name and is still read because the parsecraft extra's floor is
    ``>=2026.10.1``. The docling path emits no level at all — a level is a
    native-adapter fact, not an IR-wide guarantee.
    """
    if chunk.kind != "heading":
        return None
    level = chunk.metadata.get("n_level") or chunk.metadata.get("heading_level")
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


def _factory(options: ParserBuildOptions) -> ParsecraftParser:
    """Build a parsecraft backend instance from the resolved build options.

    The judge tier is wired through here so the config-level ``[ingest]``
    routing inputs land on the parser without leaking into the router —
    the factory is the single point where ``ParserBuildOptions`` meets a
    backend-specific constructor.
    """
    return ParsecraftParser(
        parsecraft_judge=options.parsecraft_judge,
        parsecraft_judge_model=options.parsecraft_judge_model,
        parsecraft_preference=options.parsecraft_preference,
    )


register_backend("parsecraft", _factory, parser_cls=ParsecraftParser)  # ty: ignore[invalid-argument-type]
