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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final

from parsecraft.backends import BackendRegistry, default_registry
from parsecraft.backends.protocol import SourceDocument
from parsecraft.environment import (
    EnvironmentInfo,
    probe_environment,
)
from parsecraft.ir.models import AssetKind, ChunkKind, DocumentResult, ExtractedAsset, StructuredChunk
from parsecraft.pipeline import (
    ALL_PASSES_FAILED_CODE,
    DEGRADED_EMPTY_OUTPUT_CODE,
    MEDIA_TYPES,
    PageGroup,
    PipelineResult,
    UnsupportedSourceError,
    analyze_source,
    execute,
    media_type_for,
)
from parsecraft.pipeline.analysis import AnalysisResult
from parsecraft.routing import NoEligibleBackendError, RoutingConstraints, RoutingError, RoutingPlan
from parsecraft.routing.judge import DeterministicJudge, RoutingJudge
from parsecraft.routing.judge_providers import JudgeError, resolve_judge
from parsecraft.routing.models import RoutingPreference

from kompend.core.config import ImageMode, ParsecraftJudge
from kompend.core.models import DocumentElement, ElementType, MetadataField, ParsedDocument
from kompend.logging import get_logger
from kompend.preprocessing.images import ImageStore, asset_for_chunk, assets_by_id
from kompend.preprocessing.parsers.options import PARSER_PARSECRAFT, ParserOptions
from kompend.preprocessing.parsers.parsecraft_routing import ParsecraftRoutingConfig, constraints_for, load_parsecraft_config

#: parsecraft chunk kind → kompend element type. Kinds without a kompend equivalent
#: (header/footer/quote/unknown/caption) map to TEXT — they carry content and
#: dropping them would lose data. Keys are :class:`parsecraft.ir.models.ChunkKind`
#: members, not bare strings, so a chunk-kind rename upstream surfaces as a
#: type error here instead of a silent ``.get()`` miss.
_CHUNK_KIND_TO_ELEMENT: dict[ChunkKind, ElementType] = {
    ChunkKind.HEADING: ElementType.HEADING,
    ChunkKind.PARAGRAPH: ElementType.TEXT,
    ChunkKind.LIST: ElementType.LIST_ITEM,
    ChunkKind.TABLE: ElementType.TABLE,
    ChunkKind.FORMULA: ElementType.FORMULA,
    ChunkKind.CODE: ElementType.CODE,
    ChunkKind.FIGURE: ElementType.IMAGE,
    ChunkKind.CAPTION: ElementType.TEXT,
    ChunkKind.HEADER: ElementType.TEXT,
    ChunkKind.FOOTER: ElementType.TEXT,
    ChunkKind.QUOTE: ElementType.TEXT,
    ChunkKind.UNKNOWN: ElementType.TEXT,
}


#: Characters of a judge/SDK error message kept in a log line. The SDK wraps
#: provider errors with the full HTTP exchange; the first clause names the
#: cause and the rest is noise in a per-file warning.
_JUDGE_ERROR_EXCERPT_CHARS = 160

#: Environment variable the docling engine reads for its ONNX thread count.
#: The engine builds its converter when the first conversion runs, so the
#: value has to be in place before that — the parser sets it when it is built.
DOCLING_THREADS_ENV: Final[str] = "DOCLING_RS_PDF_THREADS"

#: parsecraft's page verdict for "this page holds nothing to lose" (no text and
#: no images). The planner emits it as ``PageRoute.degradation_code``; parsecraft
#: exports no symbol for it, so the literal lives here, once.
DEGRADED_BLANK_PAGE = "degraded-blank-page"

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
        parsecraft_judge: ParsecraftJudge = ParsecraftJudge.NONE,
        parsecraft_judge_model: str = "",
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
        #: Which provider credential to spend on the routing judge, and its model
        #: token. ``none`` keeps the deterministic rule table; any other judge
        #: value resolves to a real :class:`RoutingJudge` once ``parse()`` runs
        #: (and only then, so a misconfigured provider never costs anything at CLI
        #: ``--help`` time). Everything else about routing — preference, OCR
        #: permission, engines, backend pins and per-backend options — is read from
        #: parsecraft's own config files, never mirrored here.
        self._judge_name = parsecraft_judge
        self._judge_model = parsecraft_judge_model
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
        """Parse *source_file* through parsecraft and map the IR to kompend."""
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
        analysis = analyze_source(
            source,
            registry,
            media_type=media_type,
            installed_extras=constraints.installed_extras,
        )
        result = self._execute_with_judge(analysis, registry, constraints, source)

        document = result.document
        elements = _map_document(document, source_file, images=self._images)
        empty_pages = _empty_pages(document)
        blank_pages, unreadable_pages = _classify_empty_pages(empty_pages, analysis, result.plan)
        _warn_on_empty_pages(source_file, unreadable_pages, blank_pages, result.plan, document)
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
        """
        env = self._ensure_environment()
        routing = self.parsecraft_settings().routing
        parts = [
            f"backends:{','.join(sorted(env.backends))}",
            f"extras:{','.join(sorted(env.installed_extras))}",
        ]
        if routing.only_backend is not None:
            parts.append(f"only:{routing.only_backend}")
        if routing.preferred_backend is not None:
            parts.append(f"preferred:{routing.preferred_backend}")
        if routing.engines:
            parts.append(f"engines:{','.join(sorted(routing.engines))}")
        if routing.allow_ocr is not None:
            parts.append(f"ocr:{'on' if routing.allow_ocr else 'off'}")
        return "|".join(parts)

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
        """
        env = self._ensure_environment()
        settings = self.parsecraft_settings()
        routing = settings.routing
        host = f"extras:{','.join(sorted(env.installed_extras))}|parsecraft:{_parsecraft_version()}|judge_extras:1"
        # Routing segments: judge (when configured) or preference (when non-default),
        # plus independent `effort:` and `ocr:` segments when the user set them.
        # Joining with `|` keeps each lever its own slot so the cache key flips when
        # any of them changes, but stays stable for the all-default combination —
        # the leading empty join renders as `""`, not `"|"`.
        parts: list[str] = []
        if self._judge_name is not ParsecraftJudge.NONE:
            parts.append(f"judge:{self._judge_name}/{self._judge_model}:{routing.preference}")
        elif routing.preference is not RoutingPreference.BALANCED:
            parts.append(f"preference:{routing.preference}")
        if routing.allow_ocr is not None:
            parts.append("ocr:on" if routing.allow_ocr else "ocr:off")
        if routing.engines:
            parts.append(f"engines:{','.join(sorted(routing.engines))}")
        if routing.preferred_backend is not None:
            parts.append(f"preferred_backend:{routing.preferred_backend}")
        if routing.only_backend is not None:
            parts.append(f"only_backend:{routing.only_backend}")
        if routing.backend_options:
            rendered = ",".join(
                f"{backend}:{','.join(f'{key}={value}' for key, value in sorted(options.items()))}"
                for backend, options in sorted(routing.backend_options.items())
            )
            parts.append(f"backend_options:{rendered}")
        # The image mode changes what an element carries (a cached file path or
        # an inline payload), so it belongs to the identity like every other
        # output-shaping input. It is emitted only when the run actually
        # extracts images: a run that extracts none produces the same document
        # as before, and must not re-invalidate a vault for that.
        if self._images.enabled:
            parts.append(f"images:{self._images.mode.value}")
        routing = "|".join(parts)
        return f"{host}|{routing}"

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
            parsecraft_judge=options.parsecraft_judge,
            parsecraft_judge_model=options.parsecraft_judge_model,
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
            fallback = self._judge_fallback if self._judge_fallback is not None else self._deterministic_judge()
            return execute(analysis, registry, constraints, source, judge=fallback)

    def _degrade_judge(self, exc: BaseException) -> None:
        """Drop a live judge that failed at call time; warn once per run."""
        self._judge = None
        self._judge_fallback = self._deterministic_judge()
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
        if self._judge_name is ParsecraftJudge.NONE:
            self._judge_fallback = self._deterministic_judge()
            return self._judge_fallback
        if not self._judge_model:
            msg = (
                f"parsecraft_judge is {str(self._judge_name)!r} but parsecraft_judge_model is empty; "
                "falling back to the deterministic rule table. Set [ingest] parsecraft_judge_model "
                "to the SDK's model token for the configured provider."
            )
            logger.warning(msg)
            self._judge_fallback = self._deterministic_judge()
            return self._judge_fallback
        try:
            self._judge = resolve_judge(
                f"{self._judge_name}/{self._judge_model}",
                preference=self.parsecraft_settings().routing.preference,
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
            self._judge_fallback = self._deterministic_judge()
            return self._judge_fallback
        return self._judge


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


def _parsecraft_version() -> str:
    """The installed parsecraft version, or `"unknown"`.

    Part of the cache identity because upstream changes routing without kompend
    changing anything, and there is no way to tell from here whether that changed
    what a backend produces.
    """
    try:
        return version("parsecraft")
    except PackageNotFoundError:  # pragma: no cover — parsecraft is an extra
        return "unknown"


def _warn_on_empty_pages(
    source_file: Path,
    unreadable_pages: list[int],
    blank_pages: list[int],
    plan: RoutingPlan | None,
    document: DocumentResult,
) -> None:
    """Say out loud when pages that *had something to read* contributed nothing.

    Two different losses wear the same face, and only one of them is worth a
    warning:

    * a page with no text and **no images** is blank in the source — a separator
      or a deliberately empty page. Nothing was lost, so nothing is said; a
      warning here reads as a bug report against an OCR setup that cannot help,
      which is exactly how a vault full of ITU-T separators produced a screenful
      of "check the ocr extra" for pages that hold no ink (knox-dizo).
    * a page with no text but **images** is unread: something is there and no
      backend turned it into text. That is the case the warning exists for, and
      it quotes parsecraft's own routing reason ("OCR unavailable (the tesseract
      engine is not installed on this host)") instead of guessing at the remedy.

    Zero content instead of an exception stays possible for either case: a
    backend with no OCR path, or an upstream defect that returns an empty page
    for a scanned PDF (pc-04u). When the engine explains itself on the page —
    docling-rs reports ``docling-ocr-unavailable`` for a half-installed model
    cache — that explanation is quoted instead of being replaced by advice.
    """
    if blank_pages:
        logger.debug(
            "%s: %d blank page(s) (%s) — no text and no images, nothing to extract.",
            source_file.name,
            len(blank_pages),
            ", ".join(str(page) for page in blank_pages),
        )
    if not unreadable_pages:
        return
    logger.warning(
        "%s: %d page(s) hold images but produced no text (%s).%s%s%s",
        source_file.name,
        len(unreadable_pages),
        ", ".join(str(page) for page in unreadable_pages),
        _routing_reasons(plan, unreadable_pages),
        _page_diagnostics(document, unreadable_pages),
        _empty_output_detail(document, unreadable_pages),
    )


def _empty_output_detail(document: DocumentResult, page_numbers: list[int]) -> str:
    """What the executor said when a capable backend simply returned nothing.

    ``degraded-empty-output`` is a *quality signal*, not a page diagnostic: the
    executor adds it when an ``ocr_capable`` backend produced no content for a
    page that needs OCR, and it suppresses it on any page whose adapter already
    explained itself, so the two never describe the same page. It has no
    markdown projection, which makes ``document.quality`` the only place it can
    reach a user.
    """
    wanted = set(page_numbers)
    details = {
        f"{signal.detail} ({signal.name})"
        for signal in document.quality
        if signal.page_number in wanted and signal.name == DEGRADED_EMPTY_OUTPUT_CODE and signal.detail
    }
    if not details:
        return ""
    return f" Executor said: {'; '.join(sorted(details))}"


def _page_diagnostics(document: DocumentResult, page_numbers: list[int]) -> str:
    """What the engine itself said about these pages, if anything.

    A backend can return an empty page without failing: docling-rs reports a
    partially installed model cache as ``docling-ocr-unavailable`` on the page
    and then converts nothing. That diagnostic is the only place a user learns
    the difference between "this page is unreadable" and "your OCR weights are
    half-downloaded", so it is quoted rather than replaced by advice.
    """
    wanted = set(page_numbers)
    messages = {
        f"{diagnostic.message} ({diagnostic.code})"
        for page in document.pages
        if page.page_number in wanted
        for diagnostic in page.diagnostics
        if diagnostic.message
    }
    if not messages:
        return ""
    return f" Engine said: {'; '.join(sorted(messages))}"


def _routing_reasons(plan: RoutingPlan | None, page_numbers: list[int]) -> str:
    """The planner's own words for why these pages produced nothing, if it said any."""
    if plan is None:
        return ""
    reasons = {route.reason.strip() for route in plan.pages if route.page_number in set(page_numbers) and route.reason and route.reason.strip()}
    if not reasons:
        return ""
    return f" Routing said: {'; '.join(sorted(reasons))}"


def _empty_pages(document: DocumentResult) -> list[int]:
    """Page numbers whose blocks carry no content, in document order.

    Content, not block count, decides: a page can hold chunks (headers, an empty
    figure) and still say nothing.
    """
    return [page.page_number for page in document.pages if not any(_chunk_content(chunk) for chunk in page.blocks)]


def _classify_empty_pages(empty_pages: list[int], analysis: AnalysisResult, plan: RoutingPlan | None) -> tuple[list[int], list[int]]:
    """Split empty pages into blank-in-the-source and unread.

    The planner owns this decision: ``degradation_code == degraded-blank-page``
    means "nothing to lose" (parsecraft measured 527 s of OCR on one such page),
    and recombining analysis signals here would re-derive a rule it already
    applied. ``image_count`` is the fallback for a route that carries no
    degradation at all — a page with neither text nor images cannot be read by
    any backend. A page with neither a route nor a signal counts as unread:
    silence is not evidence of emptiness.
    """
    codes = {route.page_number: route.degradation_code for route in plan.pages} if plan is not None else {}
    images = {signal.page_number: signal.image_count for signal in analysis.signals}
    blank: list[int] = []
    unreadable: list[int] = []
    for page_number in empty_pages:
        code = codes.get(page_number)
        if code == DEGRADED_BLANK_PAGE or (code is None and images.get(page_number, 1) == 0):
            blank.append(page_number)
        else:
            unreadable.append(page_number)
    return blank, unreadable


def _map_document(document: DocumentResult, source_file: Path, *, images: ImageStore | None = None) -> list[DocumentElement]:
    """Map parsecraft IR pages/chunks to kompend elements, order + pages preserved.

    Figure images travel with their chunk: the IR names the asset a chunk owns
    in the chunk metadata, the store decides where the crop lands, and the
    element carries the reference. An asset no chunk claims still belongs to
    the document and becomes an element of its own.

    Raises :class:`ParsecraftParseError` when execution genuinely exhausted —
    discriminated by execution evidence, NOT by ``winner is None`` (a parsecraft
    cache hit also yields ``winner=None`` with plan-shaped ``attempts == []``
    groups and a perfectly valid document).
    """
    exhausted = {page.page_number for page in document.pages if any(diagnostic.code == ALL_PASSES_FAILED_CODE for diagnostic in page.diagnostics)}
    if exhausted and exhausted == {page.page_number for page in document.pages}:
        raise ParsecraftParseError(source_file, document)

    store = images if images is not None else ImageStore(None, ImageMode.NONE)
    assets = assets_by_id(document) if store.enabled else {}
    claimed: set[str] = set()

    elements: list[DocumentElement] = []
    for page in document.pages:
        for chunk in page.blocks:
            asset = asset_for_chunk(chunk, assets)
            image_metadata: dict[str, str] = {}
            if asset is not None:
                claimed.add(asset.id)
                image_metadata = store.store(asset, source_stem=source_file.stem)
            element = _chunk_to_element(chunk, source_file, len(elements), image_metadata=image_metadata)
            if element is not None:
                elements.append(element)
        for asset in page.assets:
            if asset.kind != AssetKind.IMAGE or asset.id in claimed:
                continue
            element = _image_element(asset, page.page_number, source_file, len(elements), store)
            if element is not None:
                elements.append(element)
    if not elements:
        # A valid parsecraft document always carries at least one block; zero
        # content is a failure regardless of whether diagnostics were attached.
        raise ParsecraftParseError(source_file, document)
    return elements


def _image_element(
    asset: ExtractedAsset,
    page_number: int,
    source_file: Path,
    reading_order: int,
    store: ImageStore,
) -> DocumentElement | None:
    """An image no chunk claimed, as an element of its own."""
    metadata = store.store(asset, source_stem=source_file.stem)
    reference = metadata.get(MetadataField.IMAGE_FILE) or metadata.get(MetadataField.IMAGE_BASE64)
    if not reference:
        return None
    return DocumentElement(
        type=ElementType.IMAGE,
        content=reference,
        page=page_number,
        reading_order=reading_order,
        metadata=metadata,
        source_file=source_file,
    )


def _chunk_to_element(
    chunk: StructuredChunk,
    source_file: Path,
    reading_order: int,
    *,
    image_metadata: dict[str, str] | None = None,
) -> DocumentElement | None:
    """One IR chunk → one kompend element, or ``None`` when it carries no content."""
    element_type = _CHUNK_KIND_TO_ELEMENT.get(chunk.kind, ElementType.TEXT)
    content = _chunk_content(chunk)
    metadata: dict[str, Any] = {
        "parsecraft_kind": chunk.kind.value,
        "confidence": chunk.confidence,
    }
    if image_metadata:
        metadata.update(image_metadata)
    if element_type is ElementType.IMAGE:
        # The image reference is the element's content — the chunker reads
        # it back as the figure reference. The figure's own text rides
        # along as its description, so nothing the backend said is lost
        # and the captioning stage can use it instead of a VLM call.
        reference = image_metadata.get(MetadataField.IMAGE_FILE) or image_metadata.get(MetadataField.IMAGE_BASE64) or "" if image_metadata else ""
        if reference:
            if isinstance(content, str) and content.strip():
                metadata[MetadataField.DESCRIPTION] = content
            content = reference
        else:
            # No image to point at (no cache directory, unreadable payload):
            # the figure's own text is the content, and it is text, not an
            # image — an image element without a reference would break the
            # figure extractor downstream.
            element_type = ElementType.TEXT
    if not content:
        return None
    return DocumentElement(
        type=element_type,
        content=content,
        page=chunk.page_number,
        reading_order=reading_order,
        heading_level=_heading_level(chunk),
        metadata=metadata,
        source_file=source_file,
    )


def _chunk_content(chunk: StructuredChunk) -> str | list[list[str]]:
    """Chunk content in kompend's element shape.

    Tables may arrive as Markdown text or as rows. Rows win when present — the
    IR hands them over as a tuple of tuples while a TABLE element takes
    ``list[list[str]]``, so that boundary conversion happens here.

    Which backends populate ``rows`` is their business: since parsecraft
    2026.10.14 the CSV/TSV tier always does (header row first, ragged rows
    preserved), and liteparse does when the opt-in ``extract_vector_graphics``
    backend option is on. Everything else emits Markdown, which kompend's own
    markdown converter restores to a grid downstream.
    """
    if chunk.rows is not None:
        return [list(row) for row in chunk.rows]
    return chunk.content if isinstance(chunk.content, str) else ""


def _heading_level(chunk: StructuredChunk) -> int | None:
    """Heading depth for heading chunks.

    parsecraft carries the level in chunk metadata as a string (``"1"``,
    ``"2"``); 2026.10.2 renamed the key to ``n_level``, and ``heading_level``
    is the 2026.10.1 name, still read for older wheels.

    The value is the *author* level: a Markdown ``#`` is 1, and 2026.10.14 made
    docling report the author's own outline level too (its renderer's ``+1``
    shift stays inside parsecraft). A missing level therefore means the backend
    had no outline to report, not that the path has no levels. kompend clamps
    the value to 1..6 where it renders.
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
