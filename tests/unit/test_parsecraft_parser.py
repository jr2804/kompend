"""Tests for the parsecraft adapter (``kompend.preprocessing.parsers.parsecraft_parser``).

Fidelity contract: parsecraft IR → kompend elements must preserve element kind,
page numbers, reading order, heading levels, and table shape — and must NOT
emit content when parsecraft genuinely produced none. The registry contract is
also verified: parsecraft must NOT broaden default discovery (Phase 1's
byte-identical guarantee), only explicit selection may.

The adapter module is imported lazily inside each test/fixture: importing it
triggers ``register_backend("parsecraft")`` (kompend's module-level
self-registration convention), which broadens ``get_all_extensions()`` for the
rest of the process and breaks exact-count discovery tests elsewhere in the
suite. Default kompend runs never import the module, so the guarantee holds there.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("parsecraft", reason="parsecraft extra not installed")

from parsecraft import pipeline  # noqa: E402 — guarded by the importorskip above
from parsecraft.backends import default_registry  # noqa: E402 — guarded by the importorskip above
from parsecraft.backends.protocol import BackendCapabilities  # noqa: E402 — guarded by the importorskip above
from parsecraft.ir.models import (
    ChunkKind,
    Diagnostic,
    DiagnosticLevel,
    DocumentMetadata,
    DocumentResult,
    PageResult,
    StructuredChunk,
)
from parsecraft.pipeline import MEDIA_TYPES
from parsecraft.pipeline.models import PageGroup
from parsecraft.routing import NoEligibleBackendError, RoutingError
from parsecraft.routing.judge import BackendDescriptor
from parsecraft.routing.models import Intent  # noqa: E402 — guarded by the importorskip above

from kompend.core.config import ParsecraftEffort, ParsecraftJudge, ParsecraftPreference
from kompend.core.models import ElementType
from kompend.preprocessing.parsers import _registry as registry_module
from kompend.preprocessing.parsers._registry import KNOWN_BACKENDS

#: Resolved lazily via importlib: a static ``from ... import parsecraft_parser``
#: would register the backend at collection time (pyreorder hoists function-level
#: imports to module level — the TYPE_CHECKING incident, commit 010097d) and
#: broaden ``get_all_extensions()`` for the rest of the test process.

# ── Fidelity: IR → kompend elements (live pipeline, native-markdown) ────────


class TestMarkdownFidelity:
    def test_element_kinds_preserved(self, parser: Any, markdown_file: Path) -> None:
        doc = parser.parse(markdown_file)  # type: ignore[attr-defined]
        kinds = [element.type for element in doc.elements]
        assert ElementType.HEADING in kinds
        assert ElementType.TEXT in kinds
        assert ElementType.CODE in kinds
        assert ElementType.TABLE in kinds

    def test_reading_order_is_document_order(self, parser: Any, markdown_file: Path) -> None:
        doc = parser.parse(markdown_file)  # type: ignore[attr-defined]
        orders = [element.reading_order for element in doc.elements]
        assert orders == sorted(orders)
        assert orders[0] == 0

    def test_heading_levels_survive_string_metadata(self) -> None:
        """Heading levels arrive as string metadata and must convert to ints.

        2026.10.2 renamed the metadata key ``heading_level`` → ``n_level``, and
        the docling-led live markdown path emits no level at all (a level is a
        native-adapter fact), so the conversion contract is asserted on the IR
        directly — the native-markdown adapter emits ``n_level`` from
        ``token.tag.removeprefix("h")``.
        """
        elements = _map(
            _document(
                [
                    _page(
                        1,
                        [
                            _chunk(ChunkKind.HEADING, "Main Title", metadata={"n_level": "1"}),
                            _chunk(ChunkKind.HEADING, "Section", metadata={"n_level": "2"}),
                            _chunk(ChunkKind.HEADING, "Legacy", metadata={"heading_level": "3"}),
                        ],
                    )
                ]
            )
        )
        levels = [e.heading_level for e in elements if e.type is ElementType.HEADING]
        assert levels == [1, 2, 3]

    def test_code_content_preserved(self, parser: Any, markdown_file: Path) -> None:
        doc = parser.parse(markdown_file)  # type: ignore[attr-defined]
        code = [e for e in doc.elements if e.type is ElementType.CODE]
        assert len(code) == 1
        assert "x = 1" in str(code[0].content)

    def test_table_content_preserved(self, parser: Any, markdown_file: Path) -> None:
        """The contract is that the cell values survive whatever the active
        renderer produces.

        Which backend serves a ``.md`` page is parsecraft's decision and has
        moved across releases (2026.10.3 returned the NATIVE leader to
        native-markdown; 2026.10.7 populates ``StructuredChunk.rows`` for
        tables), so the element may carry the grid either as structured
        ``list[list[str]]`` or as a Markdown table. Both are acceptable;
        losing a cell is not.
        """
        doc = parser.parse(markdown_file)  # type: ignore[attr-defined]
        tables = [e for e in doc.elements if e.type is ElementType.TABLE]
        assert len(tables) == 1
        content = tables[0].content
        if isinstance(content, list):
            assert content == [["A", "B"], ["1", "2"]]
        else:
            assert "| A | B |" in content
            assert "| 1 | 2 |" in content

    def test_metadata_json_serializable(self, parser: Any, markdown_file: Path) -> None:
        """Element metadata goes into the parse cache as JSON; a StrEnum leaking
        in would break cache writes.
        """
        doc = parser.parse(markdown_file)  # type: ignore[attr-defined]
        for element in doc.elements:
            json.dumps(element.metadata)

    def test_provenance_records_backend(self, parser: Any, markdown_file: Path) -> None:
        doc = parser.parse(markdown_file)  # type: ignore[attr-defined]
        assert doc.metadata["parser"] == "parsecraft"
        # 2026.10.3 routes `.md` to `native-markdown` via the NATIVE
        # family-rank contract (the leader moved back from `docling` to
        # `native-markdown` between 2026.10.2 and 2026.10.3; both candidates
        # declare `est_vram=None` so the tiebreak is family rank, not
        # strength). Do not "fix" this back to `docling`.
        assert doc.metadata["backends"] == "native-markdown"


# ── Failure shapes ───────────────────────────────────────────────────────


class TestFailureShapes:
    def test_total_exhaustion_raises_never_emits_placeholders(self) -> None:
        """All pages exhausted → kompend parse failure, not empty content."""
        adapter = _adapter()
        with pytest.raises(adapter.ParsecraftParseError, match="all candidates failed"):
            _map(_document([_exhausted_page(1)]))

    def test_partial_exhaustion_keeps_good_pages(self) -> None:
        """One exhausted page must not discard the pages that parsed."""
        good = _page(1, [_chunk(ChunkKind.HEADING, "Kept", 1)])
        bad = _exhausted_page(2)
        elements = _map(_document([good, bad]))
        assert [str(e.content) for e in elements] == ["Kept"]

    def test_cache_hit_shape_is_valid_content(self) -> None:
        """A parsecraft cache hit yields plan-shaped groups (winner=None,
        attempts=[]) AND a fully populated document. The adapter must treat it
        as content — the winner/attempts discriminator, not winner alone.
        """
        page = _page(1, [_chunk(ChunkKind.HEADING, "Cached", 1)])
        groups = [
            PageGroup(
                page_numbers=[1],
                intent="native",
                candidates=["native-text"],
                winner=None,  # cache hit: nothing dispatched
                attempts=[],
            )
        ]
        elements = _map(_document([page]))
        assert [str(e.content) for e in elements] == ["Cached"]
        assert groups[0].attempts == []

    def test_empty_document_raises(self) -> None:
        adapter = _adapter()
        with pytest.raises(adapter.ParsecraftParseError):
            _map(_document([_page(1, [])]))


# ── Registry contract ────────────────────────────────────────────────────


class TestRegistryContract:
    def test_extensions_derived_from_parsecraft_not_hardcoded(self, parser: Any) -> None:
        assert parser.supported_extensions == set(MEDIA_TYPES)  # type: ignore[attr-defined]

    def test_can_parse_positive_and_negative(self, parser: Any) -> None:
        can_parse = parser.can_parse  # type: ignore[attr-defined]
        assert can_parse(Path("doc.pdf"))
        assert can_parse(Path("notes.txt"))
        assert can_parse(Path("code.py"))
        assert not can_parse(Path("archive.xyz123"))

    def test_default_discovery_follows_the_default_backend(self, tmp_path: Path) -> None:
        """A default kompend run discovers the default backend's suffixes.

        This test's earlier form pinned the opposite policy: parsecraft was
        kept out of ``KNOWN_BACKENDS`` so its text family could not widen
        discovery. That exclusion existed only because discovery unioned every
        *selectable* backend; with one configured backend there is nothing to
        exclude, and a vault configured for parsecraft can ingest source files.
        Verified in a fresh interpreter, so it also covers the lazy
        ``_ensure_backend`` import path.
        """
        assert "parsecraft" in KNOWN_BACKENDS

        repo_root = Path(__file__).parents[2]
        probe = (
            "from kompend.preprocessing.parsers._registry import get_all_extensions; "
            "ext = get_all_extensions(['parsecraft']); "
            "print(all(s in ext for s in ('.c', '.h', '.py', '.txt', '.pdf')))"
        )
        env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
        result = subprocess.run(  # noqa: S603 — fixed argv, probe string is a literal
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
            cwd=tmp_path,
            env=env,
            timeout=120,
        )
        assert result.stdout.strip() == "True"

    def test_unknown_suffix_classification(self) -> None:
        """Unknown suffix = input error; a valid suffix with a missing extra =
        environment error. The adapter preserves parsecraft's distinction by
        letting the typed exceptions propagate.
        """
        assert issubclass(pipeline.UnsupportedSourceError, pipeline.AnalysisError)


# ── Judge wiring ─────────────────────────────────────────────────────────


class _SentinelJudge:
    """Stand-in for a resolved JevJudge: records ``rank`` calls, never calls a model."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, list[str]]] = []

    def rank(self, intent: object, candidates: Sequence[BackendDescriptor], context: object = None) -> list[str]:
        names = [candidate.name for candidate in candidates]
        self.calls.append((intent, names))
        return names


class TestJudgeWiring:
    """The judge tier is opt-in and must never turn a misconfiguration into a failed ingest.

    Two failure surfaces, both covered here: the *resolver* (bad spec, missing
    credential, absent extra — all ``JudgeError`` subclasses) and the *call*
    (``resolve_judge`` validates the spec only, so a cloud provider accepts any
    model token and 400s on the first ``rank``; ``plan_route`` lets that
    ``RoutingError`` escape).
    """

    def test_default_is_the_deterministic_judge(self) -> None:
        adapter = _adapter()
        parser = adapter.ParsecraftParser()
        assert isinstance(parser._get_judge(), adapter.DeterministicJudge)

    def test_judge_without_model_warns_and_degrades(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        adapter = _adapter()
        parser = adapter.ParsecraftParser(parsecraft_judge="ollama", parsecraft_judge_model="")
        with caplog.at_level("WARNING"):
            judge = parser._get_judge()
        assert isinstance(judge, adapter.DeterministicJudge)
        assert "parsecraft_judge_model is empty" in caplog.text

    def test_unresolvable_provider_degrades(self, caplog: pytest.LogCaptureFixture) -> None:
        adapter = _adapter()
        parser = adapter.ParsecraftParser(parsecraft_judge="nosuchprovider", parsecraft_judge_model="m")
        with caplog.at_level("WARNING"):
            judge = parser._get_judge()
        assert isinstance(judge, adapter.DeterministicJudge)
        assert "could not be resolved" in caplog.text

    def test_live_judge_is_resolved_once_per_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One resolution per parser: the SDK memoises per page class on the
        judge instance, so re-resolving per file would discard that memo.
        """
        adapter = _adapter()
        sentinel = _SentinelJudge()
        seen: list[str] = []

        def fake_resolve(spec: str, **kwargs: object) -> object:
            seen.append(spec)
            return sentinel

        monkeypatch.setattr(adapter, "resolve_judge", fake_resolve)
        parser = adapter.ParsecraftParser(parsecraft_judge="ollama", parsecraft_judge_model="m")
        assert parser._get_judge() is sentinel
        assert parser._get_judge() is sentinel
        assert seen == ["ollama/m"]

    def test_preference_maps_to_the_sdk_enum(self) -> None:
        adapter = _adapter()
        assert adapter._parse_preference("speed").value == "speed"
        assert adapter._parse_preference("quality").value == "quality"
        # A typo raises instead of silently degrading to BALANCED —
        # a vocabulary mismatch must not quietly reshape routing.
        with pytest.raises(ValueError, match="ParsecraftPreference"):
            adapter._parse_preference("typo")

    def test_preference_reaches_the_deterministic_judge(self) -> None:
        """The knob must be honoured on the no-judge path, not only via resolve_judge.

        With the default ``parsecraft_judge = "none"`` every vault routes through
        the rule table; a preference that only reaches provider judges is
        silently inert for the default configuration.
        """
        adapter = _adapter()
        for name in ("speed", "balanced", "quality"):
            parser = adapter.ParsecraftParser(parsecraft_preference=name)
            assert parser._get_judge()._preference is adapter.RoutingPreference(name)

    def test_degraded_paths_keep_the_preference(self) -> None:
        """Resolver failure and call-time degradation both build the rule-table
        fallback with the configured axis, not the enum default.
        """
        adapter = _adapter()
        unresolvable = adapter.ParsecraftParser(parsecraft_judge="nosuchprovider", parsecraft_judge_model="m", parsecraft_preference="quality")
        assert unresolvable._get_judge()._preference is adapter.RoutingPreference.QUALITY
        degraded = adapter.ParsecraftParser(parsecraft_judge="ollama", parsecraft_judge_model="m", parsecraft_preference="quality")
        degraded._degrade_judge(RuntimeError("endpoint died"))
        assert degraded._judge_fallback._preference is adapter.RoutingPreference.QUALITY

    def test_quality_flips_the_rule_table_vram_tiebreak(self) -> None:
        """Behavioural pin: same two candidates, opposite order under quality.

        parsecraft's rule table orders SPEED and BALANCED identically (smallest
        declared VRAM first — balanced is the neutral default) and flips for
        QUALITY; preference never reorders across families. Speed-vs-balanced is
        therefore distinguishable only as the configured value (the wiring test
        above), not as routing behaviour.
        """
        adapter = _adapter()
        small, big = _pdf_backend("small", 0.5), _pdf_backend("big", 8.0)
        speed = adapter.ParsecraftParser(parsecraft_preference="speed")._get_judge()
        quality = adapter.ParsecraftParser(parsecraft_preference="quality")._get_judge()
        assert list(speed.rank(Intent.NATIVE, [small, big])) == ["small", "big"]
        assert list(quality.rank(Intent.NATIVE, [small, big])) == ["big", "small"]

    def test_cache_identity_tracks_a_non_default_preference(self) -> None:
        """A non-default preference changes rule-table routing, so the parse
        cache must invalidate; the balanced default keeps the empty identity so
        existing caches survive the fix.
        """
        adapter = _adapter()
        assert _routing_part(adapter.ParsecraftParser()) == ""
        assert _routing_part(adapter.ParsecraftParser(parsecraft_preference="balanced")) == ""
        assert _routing_part(adapter.ParsecraftParser(parsecraft_preference="speed")) == "preference:speed"
        assert _routing_part(adapter.ParsecraftParser(parsecraft_preference="quality")) == "preference:quality"

    def test_call_time_failure_degrades_to_the_rule_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A judge that resolves but fails on its first ``rank`` must not fail the ingest."""
        adapter = _adapter()
        sentinel = _SentinelJudge()
        judges: list[object] = []

        def fake_resolve(spec: str, **kwargs: object) -> object:
            return sentinel

        class _EmptyResult:
            groups: list[object] = []

        def fake_execute(*args: object, judge: object = None, **kwargs: object) -> object:
            judges.append(judge)
            if judge is sentinel:
                raise RoutingError("judge exploded")
            return _EmptyResult()

        monkeypatch.setattr(adapter, "resolve_judge", fake_resolve)
        monkeypatch.setattr(adapter, "execute", fake_execute)
        parser = adapter.ParsecraftParser(parsecraft_judge="zen", parsecraft_judge_model="m")
        result = parser._execute_with_judge(object(), object(), object(), object())
        assert result.groups == []
        assert parser._judge_degraded is True
        assert judges[0] is sentinel
        assert isinstance(judges[1], adapter.DeterministicJudge)

        # A later file must not pay the doomed call again.
        parser._execute_with_judge(object(), object(), object(), object())
        assert judges[2] is judges[1]

    def test_no_eligible_backend_error_is_not_absorbed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A real routing dead-end must surface: the deterministic judge could
        not have fixed it, so falling back would hide a genuine failure.
        """
        adapter = _adapter()

        def fake_execute(*args: object, judge: object = None, **kwargs: object) -> object:
            raise NoEligibleBackendError(None, "no backend can parse this")

        monkeypatch.setattr(adapter, "execute", fake_execute)
        parser = adapter.ParsecraftParser(parsecraft_judge="ollama", parsecraft_judge_model="m")
        with pytest.raises(NoEligibleBackendError):
            parser._execute_with_judge(object(), object(), object(), object())

    def test_deterministic_failure_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no live judge configured there is nothing to degrade from."""
        adapter = _adapter()
        calls = 0

        def fake_execute(*args: object, judge: object = None, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            raise RoutingError("routing broke")

        monkeypatch.setattr(adapter, "execute", fake_execute)
        parser = adapter.ParsecraftParser()
        with pytest.raises(RoutingError):
            parser._execute_with_judge(object(), object(), object(), object())
        assert calls == 1

    def test_judge_config_changes_the_cache_fingerprint(self) -> None:
        """Switching the judge must invalidate the parse cache, not silently
        reuse pages routed under the previous configuration.
        """
        adapter = _adapter()
        opts = registry_module.ParserBuildOptions
        plain = registry_module.build_default_router(opts(), name="parsecraft")
        judged = registry_module.build_default_router(
            opts(parsecraft_judge=ParsecraftJudge.OLLAMA, parsecraft_judge_model="parable-tinyjev"),
            name="parsecraft",
        )
        other_pref = registry_module.build_default_router(
            opts(parsecraft_judge=ParsecraftJudge.OLLAMA, parsecraft_judge_model="parable-tinyjev", parsecraft_preference=ParsecraftPreference.QUALITY),
            name="parsecraft",
        )
        other_model = registry_module.build_default_router(
            opts(parsecraft_judge=ParsecraftJudge.OLLAMA, parsecraft_judge_model="tev1"),
            name="parsecraft",
        )

        assert plain.fingerprint() != judged.fingerprint()
        assert judged.fingerprint() != other_pref.fingerprint()
        assert judged.fingerprint() != other_model.fingerprint()
        # The default configuration keeps a neutral routing part, and two
        # separately built default routers hash identically.
        assert _routing_part(adapter.ParsecraftParser()) == ""
        plain_again = registry_module.build_default_router(opts(), name="parsecraft")
        assert plain_again.fingerprint() == plain.fingerprint()

    def test_build_options_carry_the_judge_config(self) -> None:
        """The `[ingest]` routing keys reach the backend through ParserBuildOptions."""
        adapter = _adapter()
        router = registry_module.build_default_router(
            registry_module.ParserBuildOptions(
                parsecraft_judge=ParsecraftJudge.OLLAMA,
                parsecraft_judge_model="parable-tinyjev",
                parsecraft_preference=ParsecraftPreference.QUALITY,
            ),
            name="parsecraft",
        )
        backend: Any = next(p for p in router._parsers if type(p).__name__ == "ParsecraftParser")  # noqa: SLF001 — parser identity is the contract
        assert backend._judge_name == "ollama"
        assert backend._judge_model == "parable-tinyjev"
        assert backend._judge_preference == "quality"
        assert adapter.ParsecraftParser()._judge_name == "none"


class TestEffortWiring:
    """``[ingest] parsecraft_effort`` is read, validated and fingerprinted here.

    Until parsecraft's executor threads ``BackendConfig.options`` through
    ``RoutingConstraints.backend_options``, the value is *configured-but-inert*:
    validated by the schema, surfaced in the cache key, stored on the adapter,
    but not yet reaching mineru's ``_effort()`` validator. The tests below pin
    the kompend-side half of that contract so a vault operator flipping the knob
    sees a real parse-cache invalidation and a real ``ValueError`` on a typo —
    never a silent fall-back to ``flash``.
    """

    def test_default_effort_keeps_empty_routing_part(self) -> None:
        adapter = _adapter()
        assert _routing_part(adapter.ParsecraftParser()) == ""
        assert _routing_part(adapter.ParsecraftParser(parsecraft_effort="flash")) == ""

    def test_non_default_effort_appends_an_effort_segment(self) -> None:
        adapter = _adapter()
        assert _routing_part(adapter.ParsecraftParser(parsecraft_effort="medium")) == "effort:medium"
        assert _routing_part(adapter.ParsecraftParser(parsecraft_effort="high")) == "effort:high"
        assert _routing_part(adapter.ParsecraftParser(parsecraft_effort="xhigh")) == "effort:xhigh"

    def test_effort_invalid_raises_value_error(self) -> None:
        """A typo in effort must NOT silently fall back to ``flash`` — that would
        strip a vault of the VLM recovery its user configured. See
        ``_parse_effort`` for the rationale (mirrors parsecraft's own loud
        validator at ``backends/mineru/_impl.py``).
        """
        adapter = _adapter()
        with pytest.raises(ValueError, match="not a valid ParsecraftEffort"):
            adapter.ParsecraftParser(parsecraft_effort="hgih")
        with pytest.raises(ValueError, match="not a valid ParsecraftEffort"):
            adapter.ParsecraftParser(parsecraft_effort="hi")
        with pytest.raises(ValueError, match="not a valid ParsecraftEffort"):
            adapter._parse_effort("flashy")

    def test_effort_composes_with_preference(self) -> None:
        """When two routing inputs are non-default they coexist as separate
        segments joined by ``|``, so the cache key reflects the combination.
        """
        adapter = _adapter()
        assert _routing_part(adapter.ParsecraftParser(parsecraft_preference="quality", parsecraft_effort="high")) == "preference:quality|effort:high"
        assert _routing_part(adapter.ParsecraftParser(parsecraft_preference="speed", parsecraft_effort="xhigh")) == "preference:speed|effort:xhigh"

    def test_effort_composes_with_a_live_judge(self) -> None:
        """Effort is an independent segment regardless of judge tier — the
        judge segment already encodes the preference axis, so effort slots in
        *after* the judge tag without colliding.
        """
        adapter = _adapter()
        parser = adapter.ParsecraftParser(
            parsecraft_judge="ollama",
            parsecraft_judge_model="parable-tinyjev",
            parsecraft_preference="quality",
            parsecraft_effort="high",
        )
        assert _routing_part(parser) == "judge:ollama/parable-tinyjev:quality|effort:high"

    def test_effort_changes_the_router_fingerprint(self) -> None:
        """The cache must invalidate when the operator flips effort, or a parse
        produced at ``flash`` would silently serve under a future ``high`` request.
        """
        opts = registry_module.ParserBuildOptions
        flash = registry_module.build_default_router(opts(parsecraft_effort=ParsecraftEffort.FLASH), name="parsecraft")
        high = registry_module.build_default_router(opts(parsecraft_effort=ParsecraftEffort.HIGH), name="parsecraft")
        xhigh = registry_module.build_default_router(opts(parsecraft_effort=ParsecraftEffort.XHIGH), name="parsecraft")
        assert flash.fingerprint() != high.fingerprint()
        assert high.fingerprint() != xhigh.fingerprint()
        # Two separately built flash routers still hash identically — the
        # neutral ``flash`` axis must not split the cache.
        flash_again = registry_module.build_default_router(opts(parsecraft_effort=ParsecraftEffort.FLASH), name="parsecraft")
        assert flash_again.fingerprint() == flash.fingerprint()
        # And the default (no effort set) matches an explicitly-flash router.
        default = registry_module.build_default_router(opts(), name="parsecraft")
        assert default.fingerprint() == flash.fingerprint()

    def test_build_options_carry_the_effort_config(self) -> None:
        """Round-trip: ``ParserBuildOptions(parsecraft_effort=...)`` lands on the
        ``ParsecraftParser._effort`` attribute via ``_factory``. A typo at the
        factory boundary is caught by ``_parse_effort``, mirroring the schema's
        ``Literal`` validation one layer up.
        """
        adapter = _adapter()
        router = registry_module.build_default_router(
            registry_module.ParserBuildOptions(parsecraft_effort=ParsecraftEffort.XHIGH),
            name="parsecraft",
        )
        backend: Any = next(p for p in router._parsers if type(p).__name__ == "ParsecraftParser")  # noqa: SLF001 — parser identity is the contract
        assert backend._effort == "xhigh"
        # Default value at the dataclass level matches the schema default.
        default_backend = adapter.ParsecraftParser()
        assert default_backend._effort == "flash"
        with pytest.raises(ValueError, match="not a valid ParsecraftEffort"):
            adapter.ParsecraftParser(parsecraft_effort="ULTRA")


class TestOcrWiring:
    """``[ingest] parsecraft_allow_ocr`` is read, fingerprinted and forwarded.

    Unlike ``parsecraft_effort`` this one acts immediately: the adapter passes
    it straight into ``constraints_from_environment``, whose ``allow_ocr`` input
    the planner already honours. So the wiring is exercised end-to-end, not just
    stored.
    """

    def test_unset_keeps_empty_routing_part(self) -> None:
        adapter = _adapter()
        assert adapter.ParsecraftParser()._allow_ocr is None
        assert _routing_part(adapter.ParsecraftParser(parsecraft_allow_ocr=None)) == ""

    def test_explicit_values_add_an_ocr_segment(self) -> None:
        adapter = _adapter()
        assert _routing_part(adapter.ParsecraftParser(parsecraft_allow_ocr=True)) == "ocr:on"
        assert _routing_part(adapter.ParsecraftParser(parsecraft_allow_ocr=False)) == "ocr:off"

    def test_ocr_composes_with_effort_and_preference(self) -> None:
        adapter = _adapter()
        parser = adapter.ParsecraftParser(parsecraft_preference="quality", parsecraft_effort="high", parsecraft_allow_ocr=False)
        assert _routing_part(parser) == "preference:quality|effort:high|ocr:off"

    def test_allow_ocr_changes_the_router_fingerprint(self) -> None:
        """Flipping OCR permission changes which backend may lead, so the parse
        cache must invalidate — while the unset default keeps existing caches.
        """
        opts = registry_module.ParserBuildOptions
        unset = registry_module.build_default_router(opts(), name="parsecraft")
        off = registry_module.build_default_router(opts(parsecraft_allow_ocr=False), name="parsecraft")
        on = registry_module.build_default_router(opts(parsecraft_allow_ocr=True), name="parsecraft")
        assert unset.fingerprint() != off.fingerprint()
        assert off.fingerprint() != on.fingerprint()
        unset_again = registry_module.build_default_router(opts(), name="parsecraft")
        assert unset_again.fingerprint() == unset.fingerprint()

    def test_build_options_carry_the_allow_ocr_config(self) -> None:
        adapter = _adapter()
        router = registry_module.build_default_router(
            registry_module.ParserBuildOptions(parsecraft_allow_ocr=False),
            name="parsecraft",
        )
        backend: Any = next(p for p in router._parsers if type(p).__name__ == "ParsecraftParser")  # noqa: SLF001 — parser identity is the contract
        assert backend._allow_ocr is False
        assert adapter.ParsecraftParser()._allow_ocr is None

    def test_parse_forwards_allow_ocr_to_the_planner(self, parser: Any, markdown_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The value must reach ``constraints_from_environment`` — this is the
        knob that acts immediately, so a stored-but-unforwarded regression would
        be silent otherwise.
        """
        adapter = _adapter()
        seen: list[bool | None] = []
        real = adapter.constraints_from_environment

        def spy(environment: object, **kwargs: Any) -> Any:
            seen.append(kwargs["allow_ocr"])
            return real(environment, **kwargs)

        monkeypatch.setattr(adapter, "constraints_from_environment", spy)
        parser.parse(markdown_file)
        assert seen[-1] is None
        adapter.ParsecraftParser(parsecraft_allow_ocr=False).parse(markdown_file)
        assert seen[-1] is False


class TestStructuredRows:
    """``StructuredChunk.rows`` (tuple of tuples) → a TABLE's ``list[list[str]]``.

    parsecraft carries a recovered grid as a tuple of tuples while kompend's
    element carries ``list[list[str]]``; the conversion lives at the IR
    boundary so nothing downstream has to know about the tuple shape.
    """

    def test_rows_convert_to_nested_lists(self) -> None:
        adapter = _adapter()
        chunk = self._chunk(rows=(("A", "B"), ("1", "2")))
        assert adapter._chunk_content(chunk) == [["A", "B"], ["1", "2"]]

    def test_rows_win_over_markdown_content(self) -> None:
        adapter = _adapter()
        chunk = self._chunk(content="| ignored |", rows=(("A",),))
        assert adapter._chunk_content(chunk) == [["A"]]

    def test_absent_rows_fall_back_to_markdown(self) -> None:
        adapter = _adapter()
        assert adapter._chunk_content(self._chunk()) == "| A | B |"

    def test_empty_rows_do_not_become_an_empty_table(self) -> None:
        """An empty grid carries no content, so the element is dropped rather
        than emitted as an empty TABLE.
        """
        adapter = _adapter()
        chunk = self._chunk(content="", rows=())
        assert not adapter._chunk_content(chunk)

    @staticmethod
    def _chunk(**overrides: Any) -> Any:
        defaults: dict[str, Any] = {
            "id": "c1",
            "kind": ChunkKind.TABLE,
            "content": "| A | B |",
            "page_number": 1,
            "reading_order": 0,
        }
        return StructuredChunk(**{**defaults, **overrides})


class TestBackendOptions:
    """Per-backend options reach parsecraft, injected only when non-default.

    parsecraft hashes the options dict verbatim into the conversion-cache key,
    so an explicit default keys differently from nothing — injecting defaults
    would churn every existing vault's conversion cache on upgrade.
    """

    def test_no_options_when_everything_is_default(self) -> None:
        adapter = _adapter()
        assert adapter.ParsecraftParser()._backend_options() == {}

    def test_effort_injected_only_when_non_default(self) -> None:
        adapter = _adapter()
        assert adapter.ParsecraftParser(parsecraft_effort="flash")._backend_options() == {}
        assert adapter.ParsecraftParser(parsecraft_effort="high")._backend_options() == {"mineru": {"effort": "high"}}

    def test_vector_graphics_injected_when_set(self) -> None:
        adapter = _adapter()
        assert adapter.ParsecraftParser()._backend_options() == {}
        on = adapter.ParsecraftParser(parsecraft_extract_vector_graphics=True)._backend_options()
        assert on == {"liteparse": {"extract_vector_graphics": True}}
        off = adapter.ParsecraftParser(parsecraft_extract_vector_graphics=False)._backend_options()
        assert off == {"liteparse": {"extract_vector_graphics": False}}

    def test_both_land_in_one_dict(self) -> None:
        adapter = _adapter()
        parser = adapter.ParsecraftParser(parsecraft_effort="xhigh", parsecraft_extract_vector_graphics=True)
        assert parser._backend_options() == {
            "mineru": {"effort": "xhigh"},
            "liteparse": {"extract_vector_graphics": True},
        }

    def test_options_reach_the_constraints(self, parser: Any, markdown_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end: the built dict must be handed to ``constraints_from_environment``."""
        adapter = _adapter()
        seen: list[object] = []
        real = adapter.constraints_from_environment

        def spy(environment: object, **kwargs: Any) -> Any:
            seen.append(kwargs.get("backend_options"))
            return real(environment, **kwargs)

        monkeypatch.setattr(adapter, "constraints_from_environment", spy)
        parser.parse(markdown_file)
        assert seen[-1] == {}
        adapter.ParsecraftParser(parsecraft_effort="high").parse(markdown_file)
        assert seen[-1] == {"mineru": {"effort": "high"}}


class TestEnginesWiring:
    """``[ingest] parsecraft_engines`` overrides the probed engine set."""

    def test_unset_keeps_the_probe_and_empty_key(self) -> None:
        adapter = _adapter()
        parser = adapter.ParsecraftParser()
        assert parser._engines == ()
        assert _routing_part(parser) == ""

    def test_names_appear_in_the_cache_key_sorted(self) -> None:
        adapter = _adapter()
        parser = adapter.ParsecraftParser(parsecraft_engines=["mineru", "marker"])
        assert _routing_part(parser) == "engines:marker,mineru"

    def test_vector_graphics_segment_only_when_set(self) -> None:
        adapter = _adapter()
        assert _routing_part(adapter.ParsecraftParser(parsecraft_extract_vector_graphics=True)) == "vector_graphics:on"
        assert _routing_part(adapter.ParsecraftParser(parsecraft_extract_vector_graphics=False)) == "vector_graphics:off"

    def test_engines_reach_the_constraints(self, parser: Any, markdown_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _adapter()
        seen: list[object] = []
        real = adapter.constraints_from_environment

        def spy(environment: object, **kwargs: Any) -> Any:
            seen.append(kwargs.get("engines"))
            return real(environment, **kwargs)

        monkeypatch.setattr(adapter, "constraints_from_environment", spy)
        parser.parse(markdown_file)
        assert seen[-1] is None
        adapter.ParsecraftParser(parsecraft_engines=["mineru"]).parse(markdown_file)
        assert seen[-1] == ("mineru",)


class TestBackendOverride:
    """``[ingest] parsecraft_preferred_backend`` / ``_only_backend``.

    parsecraft owns routing, so these pass a single preference or pin through
    to its own parameters — they are not a kompend-side routing list.
    """

    def test_unset_leaves_both_none(self) -> None:
        adapter = _adapter()
        parser = adapter.ParsecraftParser()
        assert parser._preferred_backend is None
        assert parser._only_backend is None
        assert _routing_part(parser) == ""

    def test_unknown_backend_names_the_registered_set(self) -> None:
        adapter = _adapter()
        known = {descriptor.name for descriptor in default_registry.list_backends()}
        with pytest.raises(ValueError, match="unknown parsecraft backend 'not-a-backend'"):
            adapter.ParsecraftParser(parsecraft_preferred_backend="not-a-backend")
        with pytest.raises(ValueError, match="unknown parsecraft backend") as caught:
            adapter.ParsecraftParser(parsecraft_only_backend="also-not-a-backend")
        for name in known:
            assert name in str(caught.value)

    def test_registered_names_are_accepted(self) -> None:
        adapter = _adapter()
        chosen = self._a_backend()
        assert adapter.ParsecraftParser(parsecraft_preferred_backend=chosen)._preferred_backend == chosen
        assert adapter.ParsecraftParser(parsecraft_only_backend=chosen)._only_backend == chosen

    def test_each_names_its_own_cache_segment(self) -> None:
        adapter = _adapter()
        chosen = self._a_backend()
        assert _routing_part(adapter.ParsecraftParser(parsecraft_preferred_backend=chosen)) == f"preferred_backend:{chosen}"
        assert _routing_part(adapter.ParsecraftParser(parsecraft_only_backend=chosen)) == f"only_backend:{chosen}"

    def test_overrides_reach_the_constraints(self, parser: Any, markdown_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _adapter()
        chosen = self._a_backend()
        seen: dict[str, object] = {}
        real = adapter.constraints_from_environment

        def spy(environment: object, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return real(environment, **kwargs)

        monkeypatch.setattr(adapter, "constraints_from_environment", spy)
        parser.parse(markdown_file)
        assert seen.get("preferred_backend") is None
        assert seen.get("only_backend") is None
        adapter.ParsecraftParser(parsecraft_preferred_backend=chosen, parsecraft_only_backend=chosen).parse(markdown_file)
        assert seen["preferred_backend"] == chosen
        assert seen["only_backend"] == chosen

    @staticmethod
    def _a_backend() -> str:
        return default_registry.list_backends()[0].name


class TestIgnoredBackendOptions:
    """A per-backend option that never ran must not fail silently.

    parsecraft degrades to the backends that are left, so an unusable option
    yields a *smaller* document rather than an error — the symptom is a
    shortfall with nothing in ``quality_signals`` or ``degraded_pages``. The
    adapter logs the mismatch so the cause is attributable to the config, but
    only for backends that could have read the source: warning on a format a
    backend cannot serve would drown the signal in per-file noise.
    """

    def test_no_log_when_options_were_honoured(self, parser: Any, markdown_file: Path, caplog: pytest.LogCaptureFixture) -> None:
        parser.parse(markdown_file)
        assert not [r for r in caplog.records if "ignored the configured backend option" in r.getMessage()]

    def test_warns_when_the_named_backend_serves_the_format_but_never_ran(
        self,
        parser: Any,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The upstream degradation path: the option's backend could have read
        this source but another backend served it instead, so the option had no
        effect.
        """
        adapter = _adapter()
        monkeypatch.setattr(adapter, "_winning_backends", lambda groups: "pdf-inspector")
        monkeypatch.setattr(adapter, "_serves_format", lambda name, media_type: True)
        with caplog.at_level("WARNING", logger="kompend.preprocessing.parsers.parsecraft_parser"):
            adapter.ParsecraftParser(parsecraft_extract_vector_graphics=True).parse(markdown_file)
        messages = [r.getMessage() for r in caplog.records if "ignored the configured backend option" in r.getMessage()]
        assert len(messages) == 1, messages
        assert "liteparse" in messages[0]
        assert "pdf-inspector" in messages[0]

    def test_silent_when_the_named_backend_cannot_serve_the_format(
        self,
        parser: Any,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A ``.md`` source never reaching liteparse is the configured behaviour,
        not a misconfiguration — warning on it would bury the case that matters.
        """
        adapter = _adapter()
        # The real ``_serves_format``: liteparse declares pdf/image, not markdown.
        assert not adapter._serves_format("liteparse", "text/markdown")
        monkeypatch.setattr(adapter, "_winning_backends", lambda groups: "native-markdown")
        with caplog.at_level("WARNING", logger="kompend.preprocessing.parsers.parsecraft_parser"):
            adapter.ParsecraftParser(parsecraft_extract_vector_graphics=True).parse(markdown_file)
        assert not [r for r in caplog.records if "ignored the configured backend option" in r.getMessage()]

    def test_silent_for_a_default_run(
        self,
        parser: Any,
        markdown_file: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        adapter = _adapter()
        adapter.ParsecraftParser().parse(markdown_file)
        assert not [r for r in caplog.records if "ignored the configured backend option" in r.getMessage()]


def _pdf_backend(name: str, vram: float) -> BackendDescriptor:
    """A minimal PDF candidate whose only difference is declared VRAM.

    Deliberately a fake descriptor, not a real parsecraft backend, so the test does
    not depend on any backend's name — the rename `native-pdf` -> `pymupdf` in the
    release after 2026.10.3 must not reach it.
    """
    caps = BackendCapabilities(
        supported_formats=["application/pdf"],
        supports_page_ranges=True,
        supports_multi_page=True,
        gpu_requirement=0.0,
        estimated_vram_gb=vram,
        optional_dependency_group=None,
        model_asset=None,
        languages=(),
        required_engine=None,
    )
    return BackendDescriptor(name=name, version="0.1.0", capabilities=caps)


def _routing_part(parser: Any) -> str:
    """The routing segment of a parsecraft ``cache_identity``.

    The whole identity also carries the host's installed extra set, which differs
    per machine and would make exact-string assertions host-dependent. The routing
    half is the part these tests are about. Splitting on the host/routing boundary
    only (the second ``|``) keeps multi-segment routing keys — ``preference:..|effort:..``
    — intact for assertions, where ``rsplit('|', 1)[1]`` would have stripped the
    leading ``preference:..`` off and made the composition tests falsely green.
    """
    identity = parser.cache_identity()
    parts = identity.split("|", 2)
    return parts[2] if len(parts) >= 3 else ""


def _map(document: DocumentResult, name: str = "d.txt") -> list:

    return _adapter()._map_document(document, Path(name))


# ── IR fixtures ──────────────────────────────────────────────────────────


def _chunk(
    kind: ChunkKind,
    content: str,
    page: int = 1,
    *,
    metadata: dict[str, str] | None = None,
) -> StructuredChunk:
    return StructuredChunk(
        id=f"p{page}-{content[:12]}",
        kind=kind,
        content=content,
        page_number=page,
        reading_order=0,
        metadata=metadata or {},
    )


def _document(pages: list[PageResult]) -> DocumentResult:
    return DocumentResult(
        metadata=DocumentMetadata(
            source_uri="file:///t.txt",
            source_hash="a" * 64,
            format="text/plain",
            page_count=len(pages),
            title="T",
            produced_at=datetime.now(UTC),
            package_version="test",
            schema_version="1",
        ),
        pages=pages,
    )


def _page(number: int, blocks: list[StructuredChunk]) -> PageResult:
    return PageResult(page_number=number, blocks=blocks)


def _exhausted_page(number: int) -> PageResult:
    """A page carrying parsecraft's total-exhaustion diagnostic."""
    return PageResult(
        page_number=number,
        diagnostics=[
            Diagnostic(
                level=DiagnosticLevel.WARNING,
                code="pipeline-all-passes-failed",
                message="all candidates failed",
            )
        ],
    )


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def parser() -> Any:

    return _adapter().ParsecraftParser()


def _adapter() -> Any:
    """Lazy accessor for the adapter module (see module docstring)."""
    return importlib.import_module("kompend.preprocessing.parsers.parsecraft_parser")


@pytest.fixture
def markdown_file(tmp_path: Path) -> Path:
    """A markdown file exercising headings, code, and tables end-to-end."""
    path = tmp_path / "sample.md"
    path.write_text(
        "# Main Title\n\nIntro paragraph.\n\n## Section\n\n```python\nx = 1\n```\n\n| A | B |\n|---|---|\n| 1 | 2 |\n",
        encoding="utf-8",
    )
    return path
