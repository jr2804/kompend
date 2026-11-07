"""parsecraft adapter — routes kompend parsing through the external
`parsecraft <https://pypi.org/project/parsecraft/>`_ document-intelligence
pipeline.

kompend consumes parsecraft's **public API only** (``parsecraft.pipeline`` /
``parsecraft.backends`` / IR). The dependency is pinned and installed as a
wheel — never the parsecraft checkout. See
``docs/architecture/decisions.md`` ("Decision: parsecraft as the ingestion
backend") for the integration contract and the rationale for each mapping below.

Mapping summary:

- kompend file → parsecraft MIME via ``media_type_for``; the MIME feeds
  ``RoutingConstraints.formats`` and analyzer selection.
- ``analyze_source`` → ``pipeline.execute`` → IR (``DocumentResult`` →
  ``PageResult.blocks``) → kompend ``ParsedDocument`` / ``DocumentElement``.
- parsecraft's four error buckets → kompend parse failures with the cause
  preserved. Never emit placeholder pages as content: on total exhaustion
  parsecraft returns diagnostic-carrying pages, not text.
- Route degradation (OCR → native) arrives as ``QualitySignal`` entries in
  ``DocumentResult.quality`` and is forwarded as kompend metadata so it survives
  the parse cache.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from parsecraft.backends import BackendRegistry, default_registry
from parsecraft.backends.protocol import SourceDocument
from parsecraft.cli.config import config_scope
from parsecraft.environment import (
    EnvironmentInfo,
    probe_environment,
)
from parsecraft.pipeline import (
    MEDIA_TYPES,
    PageGroup,
    PipelineResult,
    UnsupportedSourceError,
    analyze_source,
    media_type_for,
)
from parsecraft.pipeline.analysis import AnalysisResult
from parsecraft.routing import RoutingConstraints
from parsecraft.routing.judge import DeterministicJudge, RoutingJudge

from kompend.core.config import ImageMode
from kompend.core.models import DocumentElement, ParsedDocument
from kompend.logging import get_logger
from kompend.preprocessing.images import ImageStore
from kompend.preprocessing.parsers.options import PARSER_PARSECRAFT, ParserOptions
from kompend.preprocessing.parsers.parsecraft import assets, conversion, identity, pages
from kompend.preprocessing.parsers.parsecraft_routing import ParsecraftRoutingConfig, constraints_for, load_parsecraft_config

#: Environment variable the docling engine reads for its ONNX thread count.
#: The engine builds its converter when the first conversion runs, so the
#: value has to be in place before that — the parser sets it when it is built.
DOCLING_THREADS_ENV: Final[str] = "DOCLING_RS_PDF_THREADS"
logger = get_logger(__name__)


class ParsecraftParser:
    """kompend Tier-1 parser backed by the parsecraft pipeline.

    Deliberately not a :class:`~kompend.preprocessing.parsers.base.BaseParser`
    subclass: it carries parsecraft-specific constructor state (the backend
    registry) that the kompend base has no business knowing about. The registry
    binds it through an explicit factory, and ``supported_extensions`` /
    ``can_parse`` / ``parse`` satisfy the structural surface ``DocumentParser`` needs.
    """

    #: Derived from parsecraft's own capability table (``MEDIA_TYPES``), not
    #: hardcoded — discovery and routing cannot disagree.
    supported_extensions: set[str] = set(MEDIA_TYPES)

    figure_enrichment = True

    def __init__(
        self,
        registry: BackendRegistry | None = None,
        *,
        parsecraft_config_files: tuple[Path, ...] = (),
        parsecraft_config: ParsecraftRoutingConfig | None = None,
        parsecraft_docling_threads: int | None = None,
        images_dir: Path | None = None,
        image_mode: ImageMode = ImageMode.FILE,
    ) -> None:
        self._registry = registry
        self._environment: EnvironmentInfo | None = None
        #: An already-loaded parsecraft config, for a caller that resolved it
        #: itself (tests, or a host that wants several parsers to share one
        #: read). ``None`` loads it lazily from ``_config_files`` on first use.
        self._parsecraft_config: ParsecraftRoutingConfig | None = parsecraft_config
        #: The config files to read, lowest priority first. Resolved once per
        #: instance alongside the environment probe, so a hundred vault files do
        #: not re-read the same TOML a hundred times.
        self._config_files = tuple(parsecraft_config_files)
        self._judge: RoutingJudge | None = None
        self._judge_fallback: RoutingJudge | None = None  # DeterministicJudge after a real judge failed
        self._judge_degraded = False  # live judge dropped this run (warn once, no retry)
        #: Figure images. The store holds the vault's image cache and the
        #: configured mode; the IR carries the crops, this decides where they
        #: land. ``ImageMode.NONE`` extracts no images at all.
        self._images = ImageStore(images_dir, image_mode)
        #: ONNX thread count for the docling engine. The engine reads it from
        #: the environment when it builds its converter, so it is applied
        #: here — before the first conversion of the process.
        self._docling_threads = parsecraft_docling_threads
        self._apply_docling_thread_limit()

    def can_parse(self, source_file: Path) -> bool:  # noqa: PLR6301 — parser-instance surface
        """True when parsecraft's classifier knows the file's suffix."""
        try:
            media_type_for(source_file)
        except UnsupportedSourceError:
            return False
        return True

    def parse(self, source_file: Path) -> ParsedDocument:
        """Parse *source_file* through parsecraft and map the IR to kompend.

        The parsecraft calls run inside
        :func:`~parsecraft.cli.config.config_scope`, so the judge spec, the
        judge's provider endpoint and the request timeout read the same
        configuration the routing constraints were derived from. Without it
        they would read the default layer set — the global file and the cwd
        project file — and a vault's ``.kompend/parsecraft.toml`` would route
        pages one way while the judge it names was never even resolved.
        """
        registry = self._registry if self._registry is not None else default_registry
        media_type = media_type_for(source_file)
        environment = self._ensure_environment()
        constraints = constraints_for(
            self.parsecraft_settings(),
            environment,
            formats={media_type},
            cached_assets=environment.cached_assets,
        )

        source = SourceDocument(uri=source_file.resolve().as_uri())
        with config_scope(self.parsecraft_settings().settings):
            analysis = analyze_source(
                source,
                registry,
                media_type=media_type,
                installed_extras=constraints.installed_extras,
            )
            result = self._execute_with_judge(analysis, registry, constraints, source)

        document = result.document
        elements = assets.map_document(document, source_file, images=self._images)
        empty_pages = pages.empty_pages(document)
        blank_pages, unreadable_pages = pages.classify_empty_pages(empty_pages, analysis, result.plan)
        pages.warn_on_empty_pages(source_file, unreadable_pages, blank_pages, result.plan, document)
        self._warn_on_ignored_backend_options(_winning_backends(result.groups), media_type, elements)

        return ParsedDocument(
            source_file=source_file,
            elements=elements,
            title=document.metadata.title or source_file.stem,
            author="",
            page_count=len(document.pages),
            metadata={
                "parser": PARSER_PARSECRAFT,
                "backends": _winning_backends(result.groups),
                "quality_signals": [signal.name for signal in document.quality],
                "degraded_pages": sorted(
                    signal.page_number for signal in document.quality if signal.name.startswith("degraded-") and signal.page_number is not None
                ),
                "empty_pages": empty_pages,
                "blank_pages": blank_pages,
                "unreadable_pages": unreadable_pages,
            },
        )

    def backend_identity(self) -> str:
        """Which backends can parse here, and which ones the config allows.

        The identity names backends, not machinery: the backends registered on
        this host, the extras that make their capabilities importable, and the
        restrictions the configuration puts on them (``only``/``preferred``
        backend, engine list, OCR permission). The parsecraft version, the
        judge, the preference and the effort are deliberately absent — a
        version bump or an unrelated bug fix must not re-parse a vault, while a
        changed backend set must.

        The restrictions are read off the *resolved* routing config, not off
        kompend's own attributes: they are parsecraft's settings now, so a
        change in ``parsecraft.toml`` has to show up here or the parse cache
        would serve documents the new routing never produced.

        Thin delegator: the identity composition lives in
        ``identity.backend_identity``; the state — the probed environment and
        the loaded settings — lives here.
        """
        return identity.backend_identity(self._ensure_environment(), self.parsecraft_settings())

    def cache_identity(self) -> str:
        """Routing inputs that change this backend's output without changing its class.

        Everything that decides which backend leads belongs here, because it changes
        what the document is parsed into. Two of those are not kompend's own config:

        * the installed extra set — a user adding or removing a parsecraft extra
          changes which backends are eligible;
        * the parsecraft version — upstream changes routing without kompend changing
          anything (the NATIVE tier flip, and mineru's ``estimated_vram_gb`` moving
          the lead from mineru to docling without touching eligibility).

        The version earns its churn: a bump is rare, coincides with an upgrade, and
        there is no way to know from here whether it changed what a backend
        produces. A backend *rename* alone would not need it, but that is the only
        case, and it is indistinguishable from this one at this boundary.

        ``judge_extras:1`` is a one-time marker, not a lever: ``DeterministicJudge``
        used to be built without the installed-extra set (knox-j5ig), so routing
        changed while every config value — and this identity — stayed put. The
        marker invalidates caches that outlived the over-demoting build; it
        carries no meaning beyond that transition.

        The no-judge, balanced-default case keeps an empty *routing* part: it is the
        rule table's neutral axis, and emitting it would re-invalidate every vault
        for no behavioural change.

        Thin delegator: the segment composition lives in
        ``identity.cache_identity``; the state — the probed environment, the
        loaded settings and the image store — lives here.
        """
        return identity.cache_identity(self._ensure_environment(), self.parsecraft_settings(), self._images)

    def parsecraft_settings(self) -> ParsecraftRoutingConfig:
        """Parsecraft's own configuration, read once per instance.

        Reading it here rather than per file matters for the same reason the
        probe does: a vault of a hundred documents must not read the same TOML
        a hundred times. It is also what makes the parse-cache identity and the
        planner see the same values — both read this one object.
        """
        if self._parsecraft_config is None:
            self._parsecraft_config = load_parsecraft_config(self._config_files)
        return self._parsecraft_config

    @classmethod
    def from_options(cls, options: ParserOptions) -> ParsecraftParser:
        """Build a parser from the resolved construction options.

        The single place where :class:`ParserOptions` meets this backend's
        constructor, so the config-level inputs land on the parser without
        leaking a second option vocabulary into the pipeline.
        """
        return cls(
            parsecraft_config_files=options.parsecraft_config_files,
            parsecraft_config=options.parsecraft_config,
            parsecraft_docling_threads=options.parsecraft_docling_threads,
            images_dir=options.images_dir,
            image_mode=options.image_mode,
        )

    def _apply_docling_thread_limit(self) -> None:
        """Bound the docling engine's ONNX thread count for this process.

        The docling engine loads its layout, table-structure and OCR models
        plus an ONNX arena; the arena scales with the thread count. Measured
        on Windows with docling-rs 2.65.1 (2026-11-04), one 8-page PDF: the
        engine default costs 7.1 s and 2790 MB of peak commit, four threads
        7.4 s and 1830 MB, a single thread 16.3 s and 1533 MB. Four is the
        knee, so it is the configured default. It is applied whether or not
        this run routes anything to docling: the bound only matters when the
        engine loads, and setting it late changes nothing.

        The value is process-wide because the engine reads it from the
        environment; it does not change the extracted text, so it is not part
        of the parse-cache fingerprint. An explicit ``DOCLING_RS_PDF_THREADS``
        in the environment wins — the caller asked for it by name.
        """
        if self._docling_threads is None:
            return
        if os.environ.get(DOCLING_THREADS_ENV):
            logger.debug(
                "keeping %s=%s from the environment; the configured %d thread(s) are not applied",
                DOCLING_THREADS_ENV,
                os.environ[DOCLING_THREADS_ENV],
                self._docling_threads,
            )
            return
        os.environ[DOCLING_THREADS_ENV] = str(self._docling_threads)
        logger.info(
            "docling engine limited to %d ONNX thread(s) via %s",
            self._docling_threads,
            DOCLING_THREADS_ENV,
        )

    def _backend_options(self) -> dict[str, dict[str, str | int | float | bool]]:
        """Per-backend constructor options for parsecraft, or ``{}`` when all default.

        Only non-default values are injected: parsecraft hashes this dict verbatim
        into the conversion-cache key, so passing ``effort="flash"`` would key
        differently from passing nothing and churn every existing vault's
        conversion cache on upgrade. The same reasoning covers
        ``extract_vector_graphics``, whose backend default is unset.

        ``liteparse.extract_images`` is DERIVED from the image mode, not a knob
        of its own: a run that stores figures needs the backend to emit them,
        and the default backend for a text PDF is liteparse, so leaving the
        option unset is what made the whole figure pipeline look broken for a
        document that routes natively (issue pc-65e fixed it upstream; this is
        the consumer side). Asking for figures while storing none would pay the
        engine's crop cost for nothing, hence the mode is the only input.

        The user's ``[routing] backend_options`` (parsecraft's config) are merged
        over the derived ones — an explicit user option wins over the derived
        default for the same key.
        """
        options: dict[str, dict[str, str | int | float | bool]] = {}
        if self._images.enabled:
            options.setdefault("liteparse", {})["extract_images"] = True
        for backend, backend_options in self.parsecraft_settings().routing.backend_options.items():
            merged = options.setdefault(backend, {})
            merged.update(backend_options)
        return options

    def _warn_on_ignored_backend_options(self, winners: str, media_type: str, elements: list[DocumentElement]) -> None:
        """Log loudly when a per-backend option never reached its backend.

        A backend option is only honoured by the backend it names, so when that
        backend is absent from the document's winners the option did nothing.
        That is not a failure to hide: parsecraft degrades to whatever backend is
        left, so an unusable option produces a *smaller* document rather than an
        error, and the only symptom is a shortfall nobody attributes to the
        config.

        The warning is scoped to backends that could have served this format: a
        ``.md`` source never reaching mineru is the configured behaviour, not a
        misconfiguration, and warning on it would bury the case that matters in
        per-file noise. A backend that *can* read the format but did not run is
        the signal.
        """
        requested = self._backend_options()
        if not requested:
            return
        executed = {name.strip() for name in winners.split(",") if name.strip()}
        ignored = sorted(name for name in requested if name not in executed and _serves_format(name, media_type))
        if not ignored:
            return
        logger.warning(
            "parsecraft ignored the configured backend option(s) %s: no page was served by %s "
            "(document came from %s%s). The named backend never ran, so the option had no effect "
            "and the document may be smaller than expected — upstream parsecraft degrades to the "
            "backends that are left instead of failing. Check that the backend is installed and "
            "eligible for this source.",
            {name: list(requested[name]) for name in ignored},
            ", ".join(ignored),
            winners or "no backend",
            f" ({len(elements)} elements)" if elements else "",
        )

    def _ensure_environment(self) -> EnvironmentInfo:
        """Probe once per instance. ``probe_environment`` shells out, so this must
        not run per file — hundreds of vault files must not mean hundreds of spawns.
        """
        if self._environment is None:
            self._environment = probe_environment()
        return self._environment

    def _deterministic_judge(self) -> DeterministicJudge:
        """The rule-table judge carrying the configured preference axis and the host's extra set.

        parsecraft's table takes the smallest declared ``estimated_vram_gb``
        inside a family for SPEED *and* BALANCED (balanced is the neutral
        default) and the largest for QUALITY; preference never reorders across
        families. Speed-vs-balanced is therefore a configured value only,
        never a routing difference — a three-way behavioural test cannot
        exist, only a wiring test plus the speed/quality flip.

        The host's installed extras ride along (parsecraft A10): a candidate
        whose ``convert_extra`` is present on this host ranks native instead
        of extra-backed. The set comes from the same probe that feeds
        ``RoutingConstraints``, so rule table and planner cannot disagree
        about the host. Omitting it would tell parsecraft "no host facts" —
        the honest shape for a caller that has none, and a systematic
        over-demotion of every extra-gated convert for one that has them
        (knox-j5ig).
        """
        return DeterministicJudge(self.parsecraft_settings().routing.preference, installed_extras=self._ensure_environment().installed_extras)

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

        Thin delegator: the policy lives in ``conversion.execute_with_judge``;
        the memo slots (``_judge``/``_judge_fallback``/``_judge_degraded``)
        live here, so a degrade is remembered for the rest of the run.
        """
        return conversion.execute_with_judge(
            analysis,
            registry,
            constraints,
            source,
            judge=self._get_judge(),
            degraded=self._judge_degraded,
            on_degrade=self._degrade_judge,
        )

    def _degrade_judge(self, exc: BaseException) -> RoutingJudge:
        """Drop a live judge that failed at call time; warn once per run.

        Thin delegator: the warning lives in ``conversion.degrade_judge``;
        the memo reset lives here.
        """
        self._judge = None
        self._judge_degraded = True
        self._judge_fallback = conversion.degrade_judge(self.parsecraft_settings(), self._ensure_environment(), exc)
        return self._judge_fallback

    def _get_judge(self) -> RoutingJudge:
        """Return the routing judge for this parser, resolving it once per instance.

        Thin delegator: the resolution paths (fresh rule table when no judge
        is configured, memo hit, spec resolution, degrade-on-``JudgeError``
        warning) live in ``conversion.get_judge``; the memo slots live here.
        Resolution happens inside the caller's ``config_scope`` (see
        :meth:`parse`), so the spec and the provider endpoint are read from
        the run's configuration. The call-time failure is handled by
        :meth:`_execute_with_judge`.
        """
        resolved, self._judge, self._judge_fallback = conversion.get_judge(
            self.parsecraft_settings(),
            self._ensure_environment(),
            judge=self._judge,
            fallback=self._judge_fallback,
        )
        return resolved


def _serves_format(backend_name: str, media_type: str) -> bool:
    """Whether a registered backend declares support for this media type.

    Used to keep the ignored-option warning actionable: a backend that cannot
    read the format was never going to run, so its silence is expected.
    """
    for descriptor in default_registry.list_backends():
        if descriptor.name == backend_name:
            return media_type in descriptor.capabilities.supported_formats
    return False


def _winning_backends(groups: list[PageGroup]) -> str:
    """Comma-separated backends that won page groups — parse provenance."""
    names = {group.winner for group in groups if group.winner is not None}
    return ",".join(sorted(names))
