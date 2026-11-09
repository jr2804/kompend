"""Tests for the parsecraft adapter (``kompend.preprocessing.parsers.parsecraft.adapter``).

Fidelity contract: parsecraft IR → kompend elements must preserve element kind,
page numbers, reading order, heading levels, and table shape — and must NOT
emit content when parsecraft genuinely produced none. The single-parser surface
is also verified: parsecraft must NOT broaden what ``supported_extensions()``
discovers beyond its own descriptors.

The adapter module is imported lazily inside each test/fixture: importing it
pulls in parsecraft's native stack, which is slow and unnecessary for the tests
that never parse.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kompend.core.config._configs import IngestConfig
from kompend.core.types import ReasonCode

pytest.importorskip("parsecraft", reason="parsecraft extra not installed")

from parsecraft import pipeline  # noqa: E402 — guarded by the importorskip above
from parsecraft.backends import default_registry  # noqa: E402 — guarded by the importorskip above
from parsecraft.backends.protocol import BackendCapabilities  # noqa: E402 — guarded by the importorskip above
from parsecraft.cli.config import JudgeSettings, ParsecraftConfig, RoutingSettings
from parsecraft.config.errors import ConfigError
from parsecraft.ir.models import (
    AssetKind,
    ChunkKind,
    Diagnostic,
    DiagnosticLevel,
    DocumentMetadata,
    DocumentResult,
    ExtractedAsset,
    PageResult,
    PageSignal,
    QualitySignal,
    StructuredChunk,
)
from parsecraft.pipeline import MEDIA_TYPES
from parsecraft.pipeline.analysis import AnalysisResult
from parsecraft.pipeline.models import PageGroup
from parsecraft.providers._endpoints import select_judge_spec  # noqa: E402 — guarded by the importorskip above
from parsecraft.routing import NoEligibleBackendError, RoutingError
from parsecraft.routing.judge import BackendDescriptor
from parsecraft.routing.models import Intent, PageRoute, RoutingPlan, RoutingPreference  # noqa: E402 — guarded by the importorskip above
from parsecraft.routing.rules import DEGRADED_BLANK_PAGE_CODE  # noqa: E402 — guarded by the importorskip above

from kompend.core.config import ImageMode
from kompend.core.models import ElementType, MetadataField
from kompend.preprocessing.host_memory import MEGABYTE, InsufficientHostMemoryError
from kompend.preprocessing.images import ImageStore
from kompend.preprocessing.parsers import PARSER_PARSECRAFT, ParserOptions, build_parser
from kompend.preprocessing.parsers.parsecraft_routing import ParsecraftRoutingConfig

#: A one-pixel PNG — enough to prove the bytes survive the round trip.
_PNG_BYTES: bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

#: Resolved lazily via importlib: a static ``from ... import parsecraft``
#: would register the backend at collection time (pyreorder hoists function-level
#: imports to module level — the TYPE_CHECKING incident, commit 010097d) and
#: broaden ``supported_extensions()`` for the rest of the test process.

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
        the docling path used to emit no level at all (a level is a native-adapter
        fact), so the conversion contract is asserted on the IR directly — the
        native-markdown adapter emits ``n_level`` from
        ``token.tag.removeprefix("h")``. parsecraft 2026.10.14 made docling report
        author levels too, which the live-path test below covers.
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
        with pytest.raises(_assets().ParsecraftParseError, match="all candidates failed"):
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
        with pytest.raises(_assets().ParsecraftParseError):
            _map(_document([_page(1, [])]))


# ── Silent losses: pages that produced no text ────────────────────────────


class TestEmptyPages:
    """A page that produced no text must be visible — but only when it lost something.

    A backend can answer a scanned page with zero content rather than an error —
    for example parsecraft's docling backend on an image-only PDF, which exits
    clean with an empty page (filed upstream as pc-04u). The document then parses
    "successfully" and is simply shorter than the source.

    The opposite mistake is loud and useless: ITU-T PDFs are full of deliberate
    blank separator pages, and warning "install the ocr extra" for a page that
    holds no text *and no images* accuses a working setup of a bug it cannot fix
    (knox-dizo). The planner owns that split — `degradation_code ==
    degraded-blank-page` means "nothing to lose" — with `image_count == 0` only as
    the fallback for a route that carries no degradation. Unread pages are warned
    once per file, quoting parsecraft's own routing reason.
    """

    def test_degraded_blank_page_is_parsecrafts_symbol(self) -> None:
        """The verdict is imported, never re-declared (knox-udom).

        An upstream rename fails the import at load time; an upstream value
        change fails this pin before a vault is parsed with two different
        notions of "blank".
        """
        assert _pages().DEGRADED_BLANK_PAGE is DEGRADED_BLANK_PAGE_CODE
        assert DEGRADED_BLANK_PAGE_CODE == "degraded-blank-page"

    def test_page_with_images_is_recorded_and_warned(
        self,
        parser: Any,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pages = [_page(1, [_chunk(ChunkKind.PARAGRAPH, "Text")]), _page(2, [])]
        with caplog.at_level("WARNING", logger="kompend.preprocessing.parsers.parsecraft.pages"):
            doc = self._parsing(parser, monkeypatch, pages, images={1: 0, 2: 3}).parse(markdown_file)
        assert doc.metadata["empty_pages"] == [2]
        assert doc.metadata["unreadable_pages"] == [2]
        assert doc.metadata["blank_pages"] == []
        warnings = [r.getMessage() for r in caplog.records if "hold images but produced no text" in r.getMessage()]
        assert len(warnings) == 1, warnings
        assert "1 page(s)" in warnings[0]
        assert "(2)" in warnings[0]

    def test_blank_page_is_recorded_without_a_warning(
        self,
        parser: Any,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """No text and no images: a separator page. Nothing was lost, so say nothing loud."""
        pages = [_page(1, []), _page(2, [_chunk(ChunkKind.PARAGRAPH, "Text", 2)])]
        with caplog.at_level("DEBUG", logger="kompend.preprocessing.parsers.parsecraft.pages"):
            doc = self._parsing(parser, monkeypatch, pages, images={1: 0, 2: 0}).parse(markdown_file)
        assert doc.metadata["empty_pages"] == [1]
        assert doc.metadata["blank_pages"] == [1]
        assert doc.metadata["unreadable_pages"] == []
        assert not [r for r in caplog.records if r.levelname == "WARNING"]
        assert [r.getMessage() for r in caplog.records if "blank page(s)" in r.getMessage()]

    def test_planner_blank_marker_wins_over_analysis_signals(
        self,
        parser: Any,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A page the planner marked blank is blank even if it carries images.

        The signals say image_count=1 (which alone would read as "unread"); the
        planner already decided there is nothing to lose. Recombining the signals
        here would re-derive a rule it applied, and warn about a page better left
        alone.
        """
        pages = [_page(1, []), _page(2, [_chunk(ChunkKind.PARAGRAPH, "Text", 2)])]
        with caplog.at_level("WARNING", logger="kompend.preprocessing.parsers.parsecraft.pages"):
            doc = self._parsing(
                parser,
                monkeypatch,
                pages,
                images={1: 1, 2: 0},
                reasons={1: "page 1: blank page: no text or images to lose; first pass liteparse"},
                blank={1},
            ).parse(markdown_file)
        assert doc.metadata["blank_pages"] == [1]
        assert doc.metadata["unreadable_pages"] == []
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_page_without_a_signal_counts_as_unread(
        self,
        parser: Any,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Silence is not evidence of emptiness — an unanalysed page stays warned."""
        pages = [_page(1, []), _page(2, [_chunk(ChunkKind.PARAGRAPH, "Text", 2)])]
        doc = self._parsing(parser, monkeypatch, pages, images={2: 0}).parse(markdown_file)
        assert doc.metadata["unreadable_pages"] == [1]
        assert doc.metadata["blank_pages"] == []

    def test_warning_names_every_unreadable_page_once(
        self,
        parser: Any,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """One record per file, listing the pages — not one record per page."""
        pages = [
            _page(1, []),
            _page(2, [_chunk(ChunkKind.PARAGRAPH, "Text", 2)]),
            _page(3, []),
            _page(4, []),
        ]
        with caplog.at_level("WARNING", logger="kompend.preprocessing.parsers.parsecraft.pages"):
            doc = self._parsing(parser, monkeypatch, pages, images={1: 1, 2: 0, 3: 2, 4: 1}).parse(markdown_file)
        assert doc.metadata["unreadable_pages"] == [1, 3, 4]
        warnings = [r.getMessage() for r in caplog.records if "hold images but produced no text" in r.getMessage()]
        assert len(warnings) == 1, warnings
        assert "(1, 3, 4)" in warnings[0]

    def test_partial_exhaustion_is_recorded_and_warned(
        self,
        parser: Any,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An exhausted page is missing content, not an empty page — record it as such.

        One good page and one all-passes-failed page: the good elements survive,
        no exception is raised, ``exhausted_pages`` names the failed page next
        to ``degraded_pages``, and exactly one warning names it too (knox-grcb).
        """
        pages = [_page(3, [_chunk(ChunkKind.PARAGRAPH, "Kept", 3)]), _exhausted_page(4)]
        with caplog.at_level("WARNING", logger="kompend.preprocessing.parsers.parsecraft.pages"):
            doc = self._parsing(parser, monkeypatch, pages, images={3: 0, 4: 0}).parse(markdown_file)
        assert [str(e.content) for e in doc.elements] == ["Kept"]
        assert doc.metadata["exhausted_pages"] == [4]
        warnings = [r.getMessage() for r in caplog.records if "failed to parse on every pass" in r.getMessage()]
        assert len(warnings) == 1, warnings
        assert "(4)" in warnings[0]

    def test_warning_quotes_the_routing_reason(
        self,
        parser: Any,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The remedy comes from the planner, not from a guess in the adapter."""
        pages = [_page(1, []), _page(2, [_chunk(ChunkKind.PARAGRAPH, "Text", 2)])]
        reasons = {1: "OCR unavailable (the tesseract engine is not installed on this host); degraded to native"}
        with caplog.at_level("WARNING", logger="kompend.preprocessing.parsers.parsecraft.pages"):
            self._parsing(parser, monkeypatch, pages, images={1: 1, 2: 0}, reasons=reasons).parse(markdown_file)
        warnings = [r.getMessage() for r in caplog.records if "hold images but produced no text" in r.getMessage()]
        assert len(warnings) == 1, warnings
        assert "the tesseract engine is not installed on this host" in warnings[0]

    def test_silent_when_every_page_has_text(
        self,
        parser: Any,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pages = [_page(1, [_chunk(ChunkKind.PARAGRAPH, "One")]), _page(2, [_chunk(ChunkKind.PARAGRAPH, "Two", 2)])]
        with caplog.at_level("WARNING", logger="kompend.preprocessing.parsers.parsecraft.pages"):
            doc = self._parsing(parser, monkeypatch, pages).parse(markdown_file)
        assert doc.metadata["empty_pages"] == []
        assert not [r for r in caplog.records if "produced no text" in r.getMessage()]

    def test_warning_quotes_the_engine_diagnostic(
        self,
        parser: Any,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A half-installed OCR cache is explained by the engine, so quote it."""
        pages = [
            _page(1, [_chunk(ChunkKind.PARAGRAPH, "Body")]),
            PageResult(
                page_number=2,
                diagnostics=[
                    Diagnostic(
                        level=DiagnosticLevel.WARNING,
                        code="docling-ocr-unavailable",
                        message="OCR models are not installed",
                    )
                ],
            ),
        ]
        with caplog.at_level("WARNING", logger="kompend.preprocessing.parsers.parsecraft.pages"):
            self._parsing(parser, monkeypatch, pages, images={2: 1}).parse(markdown_file)

        warned = "\n".join(r.getMessage() for r in caplog.records if "produced no text" in r.getMessage())
        assert "OCR models are not installed (docling-ocr-unavailable)" in warned

    def test_executor_empty_output_is_quoted(
        self,
        parser: Any,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A silent empty result is a quality signal, not a diagnostic: still say it."""
        pages = [_page(1, [_chunk(ChunkKind.PARAGRAPH, "Body")]), _page(2, [])]
        quality = [
            QualitySignal(
                name="degraded-empty-output",
                score=0.0,
                page_number=2,
                detail="docling returned no content for a page that needs OCR",
            )
        ]
        with caplog.at_level("WARNING", logger="kompend.preprocessing.parsers.parsecraft.pages"):
            self._parsing(parser, monkeypatch, pages, images={2: 1}, quality=quality).parse(markdown_file)

        warned = "\n".join(r.getMessage() for r in caplog.records if "produced no text" in r.getMessage())
        assert "Executor said: docling returned no content for a page that needs OCR (degraded-empty-output)" in warned

    def test_empty_pages_stay_json_serializable(
        self,
        parser: Any,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The metadata lands in the parse cache as JSON: a set or StrEnum breaks it."""
        doc = self._parsing(
            parser,
            monkeypatch,
            [_page(1, []), _page(2, [_chunk(ChunkKind.PARAGRAPH, "T", 2)])],
            images={1: 0, 2: 0},
        ).parse(markdown_file)
        payload = json.loads(json.dumps(doc.metadata))
        assert payload["empty_pages"] == [1]
        assert payload["blank_pages"] == [1]
        assert payload["unreadable_pages"] == []

    @staticmethod
    def _parsing(
        parser: Any,
        monkeypatch: pytest.MonkeyPatch,
        pages: list[PageResult],
        *,
        images: dict[int, int] | None = None,
        reasons: dict[int, str] | None = None,
        blank: set[int] | None = None,
        quality: list[QualitySignal] | None = None,
    ) -> Any:
        document = _document(pages, quality)
        analysis = _analysis(pages, images or {})
        plan = _plan(reasons or {}, blank or set())
        monkeypatch.setattr(_adapter(), "analyze_source", lambda *args, **kwargs: analysis)
        monkeypatch.setattr(
            parser,
            "_execute_with_judge",
            lambda *args, **kwargs: SimpleNamespace(document=document, groups=[], plan=plan),
        )
        return parser


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
        assert PARSER_PARSECRAFT == "parsecraft"

        repo_root = Path(__file__).parents[2]
        probe = (
            "from kompend.preprocessing.parsers.extensions import supported_extensions; "
            "ext = supported_extensions(); "
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

    def test_judge_without_a_spec_uses_the_rule_table(self) -> None:
        """No ``[judge] model`` and no env override means the deterministic table."""
        adapter = _adapter()
        parser = _judged_parser(None)
        assert isinstance(parser._get_judge(), adapter.DeterministicJudge)

    def test_the_judge_spec_is_parsecrafts_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``[judge] model`` reaches ``resolve_judge`` verbatim and unparsed.

        kompend neither splits the spec nor validates the provider: that
        vocabulary belongs to parsecraft, and a provider kompend has never
        heard of must still resolve.
        """
        sentinel = _SentinelJudge()
        seen: list[str] = []

        def fake_resolve(spec: str, **kwargs: object) -> object:
            seen.append(spec)
            return sentinel

        monkeypatch.setattr(_conversion(), "resolve_judge", fake_resolve)
        assert _judged_parser("ollama/parable-tinyjev")._get_judge() is sentinel
        assert _judged_parser("typesafe-ai/jev-latest")._get_judge() is sentinel
        assert seen == ["ollama/parable-tinyjev", "typesafe-ai/jev-latest"]

    def test_the_judge_model_env_var_overrides_the_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``PARSECRAFT_JUDGE_MODEL`` wins over ``[judge] model``, as upstream."""
        _adapter()
        monkeypatch.setenv("PARSECRAFT_JUDGE_MODEL", "zen/some-model")
        assert _judged_parser("ollama/parable-tinyjev").parsecraft_settings().judge_spec == "zen/some-model"
        assert _judged_parser(None).parsecraft_settings().judge_spec == "zen/some-model"

    def test_an_empty_judge_section_is_no_judge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PARSECRAFT_JUDGE_MODEL", raising=False)
        assert _judged_parser("").parsecraft_settings().judge_spec is None
        assert _judged_parser(None).parsecraft_settings().judge_spec is None

    def test_the_judge_spec_ignores_a_project_file_kompend_did_not_load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The settings object decides, not whatever file the cwd happens to hold.

        parsecraft resolves the judge through ``effective_settings()``, which
        reads the global file and ``./parsecraft.toml`` unless the caller
        installed a scope. A vault's judge would be invisible that way, and a
        stray project file would silently win — so the property installs its
        own settings for the question.
        """
        monkeypatch.delenv("PARSECRAFT_JUDGE_MODEL", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "parsecraft.toml").write_text('[judge]\nmodel = "zen/some-model"\n', encoding="utf-8")
        settings = ParsecraftConfig(judge=JudgeSettings(model="ollama/parable-tinyjev"))
        assert ParsecraftRoutingConfig.from_settings(settings).judge_spec == "ollama/parable-tinyjev"

    def test_the_parse_installs_the_configuration_parsecraft_resolves_from(self, markdown_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Parsecraft's judge, endpoint and timeout read the *scoped* settings.

        Proven from inside the parse: the spec that ``select_judge_spec``
        answers there is the vault's, and it is gone again afterwards.
        """
        adapter = _adapter()
        monkeypatch.delenv("PARSECRAFT_JUDGE_MODEL", raising=False)
        monkeypatch.chdir(markdown_file.parent)
        seen: list[str | None] = []
        real_analyze = adapter.analyze_source

        def spy_analyze(*args: object, **kwargs: object) -> object:
            seen.append(select_judge_spec(None))
            return real_analyze(*args, **kwargs)

        monkeypatch.setattr(adapter, "analyze_source", spy_analyze)
        _judged_parser("ollama/parable-tinyjev").parse(markdown_file)
        assert seen == ["ollama/parable-tinyjev"]
        assert select_judge_spec(None) is None

    def test_unresolvable_provider_degrades(self, caplog: pytest.LogCaptureFixture) -> None:
        adapter = _adapter()
        parser = _judged_parser("nosuchprovider/m")
        with caplog.at_level("WARNING"):
            judge = parser._get_judge()
        assert isinstance(judge, adapter.DeterministicJudge)
        assert "could not be resolved" in caplog.text
        assert "nosuchprovider/m" in caplog.text

    def test_live_judge_is_resolved_once_per_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One resolution per parser: the SDK memoises per page class on the
        judge instance, so re-resolving per file would discard that memo.
        """
        sentinel = _SentinelJudge()
        seen: list[str] = []

        def fake_resolve(spec: str, **kwargs: object) -> object:
            seen.append(spec)
            return sentinel

        monkeypatch.setattr(_conversion(), "resolve_judge", fake_resolve)
        parser = _judged_parser("ollama/m")
        assert parser._get_judge() is sentinel
        assert parser._get_judge() is sentinel
        assert seen == ["ollama/m"]

    def test_an_unknown_preference_is_rejected_by_parsecraft(self) -> None:
        """The preference vocabulary is parsecraft's, validated where it is defined."""
        with pytest.raises(ValueError, match="preference"):
            _routing(preference="typo")

    def test_preference_reaches_the_deterministic_judge(self) -> None:
        """The knob must be honoured on the no-judge path, not only via resolve_judge.

        With no judge configured every vault routes through
        the rule table; a preference that only reaches provider judges is
        silently inert for the default configuration.
        """
        for name in ("speed", "balanced", "quality"):
            parser = _routing_parser(preference=name)
            assert parser._get_judge()._preference is RoutingPreference(name)

    def test_degraded_paths_keep_the_preference(self) -> None:
        """Resolver failure and call-time degradation both build the rule-table
        fallback with the configured axis, not the enum default.
        """
        unresolvable = _judged_parser("nosuchprovider/m", preference="quality")
        assert unresolvable._get_judge()._preference is RoutingPreference.QUALITY
        degraded = _judged_parser("ollama/m", preference="quality")
        degraded._degrade_judge(RuntimeError("endpoint died"))
        assert degraded._judge_fallback._preference is RoutingPreference.QUALITY

    def test_the_judge_degradation_warnings_carry_the_reason_code(self, caplog: pytest.LogCaptureFixture) -> None:
        """A judge that degrades to the rule table is a JUDGE_DEGRADED failure.

        knox-ty8f.1: the item path reports every failure with a reason code so
        the ledger can count a failure class instead of matching prose. Both
        judge-degradation sites carry it — the call-time failure and the
        resolution failure — and they are the same class of event.
        """
        with caplog.at_level(logging.WARNING, logger="kompend.preprocessing.parsers.parsecraft.conversion"):
            _judged_parser("nosuchprovider/m", preference="quality")._get_judge()
            _judged_parser("ollama/m", preference="quality")._degrade_judge(RuntimeError("endpoint died"))

        codes = {getattr(record, "reason_code", None) for record in caplog.records if "judge" in record.getMessage()}
        assert codes == {ReasonCode.JUDGE_DEGRADED}

    def test_quality_flips_the_rule_table_vram_tiebreak(self) -> None:
        """Behavioural pin: same two candidates, opposite order under quality.

        parsecraft's rule table orders SPEED and BALANCED identically (smallest
        declared VRAM first — balanced is the neutral default) and flips for
        QUALITY; preference never reorders across families. Speed-vs-balanced is
        therefore distinguishable only as the configured value (the wiring test
        above), not as routing behaviour.
        """
        _adapter()
        small, big = _pdf_backend("small", 0.5), _pdf_backend("big", 8.0)
        speed = _routing_parser(preference="speed")._get_judge()
        quality = _routing_parser(preference="quality")._get_judge()
        assert list(speed.rank(Intent.NATIVE, [small, big])) == ["small", "big"]
        assert list(quality.rank(Intent.NATIVE, [small, big])) == ["big", "small"]

    def test_installed_extras_reach_the_deterministic_judge(self) -> None:
        """Host facts must reach the rule table (knox-j5ig): the judge is built
        with the same installed-extra set the planner sees. parsecraft's
        constructor default is the empty set — the honest shape for a caller
        with *no* host facts, and a systematic over-demotion for kompend,
        which has them.
        """
        adapter = _adapter()
        parser = adapter.ParsecraftParser()
        parser._environment = SimpleNamespace(installed_extras=frozenset({"docling", "ocr"}))  # type: ignore[assignment] — the parser reads ``installed_extras`` only
        assert parser._get_judge()._installed_extras == frozenset({"docling", "ocr"})

    def test_installed_extra_promotes_its_backend_in_the_rule_table(self) -> None:
        """Behavioural pin: an extra-gated candidate ranks NATIVE once its extra
        is installed and stays extra-backed while it is missing — the A10
        demotion must apply only to extras this host actually lacks.
        """
        adapter = _adapter()
        free, gated = _pdf_backend("free", 8.0), _extra_pdf_backend("gated", "docling", 0.5)
        parser = adapter.ParsecraftParser()
        parser._environment = SimpleNamespace(installed_extras=frozenset({"docling"}))  # type: ignore[assignment] — the parser reads ``installed_extras`` only
        assert list(parser._get_judge().rank(Intent.NATIVE, [free, gated])) == ["gated", "free"]
        without_host_facts = adapter.DeterministicJudge(RoutingPreference.BALANCED)
        assert list(without_host_facts.rank(Intent.NATIVE, [free, gated])) == ["free", "gated"]

    def test_cache_identity_tracks_a_non_default_preference(self) -> None:
        """A non-default preference changes rule-table routing, so the parse
        cache must invalidate; the balanced default keeps the empty identity so
        existing caches survive the fix.
        """
        adapter = _adapter()
        assert _routing_part(adapter.ParsecraftParser()) == ""
        assert _routing_part(_routing_parser(preference="balanced")) == ""
        assert _routing_part(_routing_parser(preference="speed")) == "preference:speed"
        assert _routing_part(_routing_parser(preference="quality")) == "preference:quality"

    def test_cache_identity_marks_the_judge_extras_fix(self) -> None:
        """knox-j5ig changed default-path routing without any config change, so
        the identity carries a one-time marker: caches written by the
        over-demoting build must not survive it.
        """
        adapter = _adapter()
        assert "judge_extras:1" in adapter.ParsecraftParser().cache_identity()

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

        monkeypatch.setattr(_conversion(), "resolve_judge", fake_resolve)
        monkeypatch.setattr(_conversion(), "execute", fake_execute)
        parser = _judged_parser("zen/m")
        result = parser._execute_with_judge(object(), object(), object(), object())
        assert result.groups == []
        assert parser._judge_degraded is True
        assert judges[0] is sentinel
        assert isinstance(judges[1], adapter.DeterministicJudge)

        # A later file must not pay the doomed call again.
        parser._execute_with_judge(object(), object(), object(), object())
        assert judges[2] is judges[1]

    def test_planning_degrades_a_dead_judge_to_the_rule_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The budget check plans through the same degradation contract.

        A judge that dies while the plan is computed must be dropped there,
        not retried by the conversion that follows.
        """
        adapter = _adapter()
        sentinel = _SentinelJudge()
        judges: list[object] = []

        def fake_plan(analysis: object, backends: object, constraints: object, judge: object = None) -> object:
            judges.append(judge)
            if judge is sentinel:
                raise RoutingError("judge exploded")
            return _planned("liteparse")

        monkeypatch.setattr(_conversion(), "resolve_judge", lambda *args, **kwargs: sentinel)
        monkeypatch.setattr(_conversion(), "plan_route", fake_plan)
        parser = _judged_parser("zen/m")
        parser._min_free_commit_bytes = 1  # opt into the budget check
        registry = SimpleNamespace(list_backends=lambda: [])
        parser._ensure_headroom(object(), registry, object(), Path("report.pdf"))
        assert judges[0] is sentinel
        assert isinstance(judges[1], adapter.DeterministicJudge)
        assert parser._judge_degraded is True

    def test_no_eligible_backend_error_is_not_absorbed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A real routing dead-end must surface: the deterministic judge could
        not have fixed it, so falling back would hide a genuine failure.
        """

        def fake_execute(*args: object, judge: object = None, **kwargs: object) -> object:
            raise NoEligibleBackendError(None, "no backend can parse this")

        monkeypatch.setattr(_conversion(), "execute", fake_execute)
        parser = _judged_parser("ollama/m")
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

        monkeypatch.setattr(_conversion(), "execute", fake_execute)
        parser = adapter.ParsecraftParser()
        with pytest.raises(RoutingError):
            parser._execute_with_judge(object(), object(), object(), object())
        assert calls == 1

    def test_judge_config_changes_the_cache_fingerprint(self) -> None:
        """Switching the judge must invalidate the parse cache, not silently
        reuse pages routed under the previous configuration.
        """
        adapter = _adapter()
        plain = _judged_parser(None)
        judged = _judged_parser("ollama/parable-tinyjev")
        other_pref = _judged_parser("ollama/parable-tinyjev", preference="quality")
        other_model = _judged_parser("ollama/tev1")

        assert plain.cache_identity() != judged.cache_identity()
        assert judged.cache_identity() != other_pref.cache_identity()
        assert judged.cache_identity() != other_model.cache_identity()
        # The default configuration keeps a neutral routing part, and two
        # separately built default parsers hash identically.
        assert _routing_part(adapter.ParsecraftParser()) == ""
        assert _judged_parser(None).cache_identity() == plain.cache_identity()

    def test_the_judge_spec_is_part_of_the_cache_identity(self) -> None:
        """The identity names the spec, so a provider swap invalidates the cache."""
        assert "judge:ollama/tev1:" in _judged_parser("ollama/tev1").cache_identity()


class TestEffortIsABackendOption:
    """MinerU effort is a parsecraft backend option, not a kompend config key.

    ``[routing.backend_options]`` is the single surface for backend options:
    parsecraft owns the option names and their value domains, so kompend must not
    mirror `effort` (or liteparse's `extract_vector_graphics`) as typed fields. The
    cache identity follows the option dict, so flipping effort through the option
    map still invalidates the parse.
    """

    def test_effort_reaches_backend_options_verbatim(self) -> None:
        _adapter()
        parser = _routing_parser(backend_options={"mineru": {"effort": "high"}})
        assert parser._backend_options() == {"mineru": {"effort": "high"}}

    def test_effort_changes_the_cache_identity(self) -> None:
        """The option dict is part of the fingerprint, so a vault that raised
        effort never serves a parse produced at the old setting.
        """
        _adapter()
        flash = _routing_parser(backend_options={"mineru": {"effort": "flash"}})
        high = _routing_parser(backend_options={"mineru": {"effort": "high"}})
        assert _routing_part(flash) != _routing_part(high)
        assert "backend_options:" in _routing_part(high)

    def test_an_unknown_option_value_is_forwarded_and_validated_upstream(self) -> None:
        """Parsecraft owns the vocabulary; kompend forwards the value untouched.

        Pre-validating here would put a second copy of parsecraft's option
        domains in kompend — the duplication this design removes. The value
        reaches ``BackendConfig.options`` verbatim and parsecraft's own
        validator decides (loudly) whether it is real.
        """
        _adapter()
        parser = _routing_parser(backend_options={"mineru": {"effort": "ULTRA"}})
        assert parser._backend_options() == {"mineru": {"effort": "ULTRA"}}


class TestOcrWiring:
    """``[routing] allow_ocr`` is read from parsecraft's config and forwarded.

    The key acts immediately: the adapter passes it straight into
    ``RoutingSettings.routing_constraints``, whose ``allow_ocr`` input the planner
    already honours. So the wiring is exercised end-to-end, not just stored.
    """

    def test_unset_keeps_empty_routing_part(self) -> None:
        adapter = _adapter()
        assert adapter.ParsecraftParser().parsecraft_settings().routing.allow_ocr is None
        assert _routing_part(_routing_parser(allow_ocr=None)) == ""

    def test_explicit_values_add_an_ocr_segment(self) -> None:
        _adapter()
        assert _routing_part(_routing_parser(allow_ocr=True)) == "ocr:on"
        assert _routing_part(_routing_parser(allow_ocr=False)) == "ocr:off"

    def test_ocr_composes_with_preference_and_backend_options(self) -> None:
        _adapter()
        parser = _routing_parser(
            preference="quality",
            allow_ocr=False,
            backend_options={"mineru": {"effort": "high"}},
        )
        assert _routing_part(parser) == "preference:quality|ocr:off|backend_options:mineru:effort=high"

    def test_allow_ocr_changes_the_parser_identity(self) -> None:
        """Flipping OCR permission changes which backend may lead, so the parse
        cache must invalidate — while the unset default keeps existing caches.
        """
        unset = _parser_with()
        off = _parser_with(parsecraft_config=_routing(allow_ocr=False))
        on = _parser_with(parsecraft_config=_routing(allow_ocr=True))
        assert unset.backend_identity() != off.backend_identity()
        assert off.backend_identity() != on.backend_identity()
        unset_again = _parser_with()
        assert unset_again.backend_identity() == unset.backend_identity()

    def test_build_options_carry_the_allow_ocr_config(self) -> None:
        """``ParserOptions.parsecraft_config`` is the seam an embedding host uses.

        A caller that resolved parsecraft's config itself hands the same object
        to every parser it builds, so a vault reads its TOML once.
        """
        backend: Any = _parser_with(parsecraft_config=_routing(allow_ocr=False)).parser
        assert type(backend).__name__ == "ParsecraftParser"
        assert backend.parsecraft_settings().routing.allow_ocr is False
        assert _adapter().ParsecraftParser().parsecraft_settings().routing.allow_ocr is None

    def test_parse_forwards_allow_ocr_to_the_planner(self, parser: Any, markdown_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The value must reach ``constraints_for`` — this is the knob that
        acts immediately, so a stored-but-unforwarded regression would be
        silent otherwise.
        """
        adapter = _adapter()
        seen: list[bool | None] = []
        real = adapter.constraints_for

        def spy(config: Any, environment: object, **kwargs: Any) -> Any:
            seen.append(config.routing.allow_ocr)
            return real(config, environment, **kwargs)

        monkeypatch.setattr(adapter, "constraints_for", spy)
        parser.parse(markdown_file)
        assert seen[-1] is None
        _routing_parser(allow_ocr=False).parse(markdown_file)
        assert seen[-1] is False


class TestStructuredRows:
    """``StructuredChunk.rows`` (tuple of tuples) → a TABLE's ``list[list[str]]``.

    parsecraft carries a recovered grid as a tuple of tuples while kompend's
    element carries ``list[list[str]]``; the conversion lives at the IR
    boundary so nothing downstream has to know about the tuple shape.
    """

    def test_rows_convert_to_nested_lists(self) -> None:
        chunk = self._chunk(rows=(("A", "B"), ("1", "2")))
        assert _assets().chunk_content(chunk) == [["A", "B"], ["1", "2"]]

    def test_rows_win_over_markdown_content(self) -> None:
        chunk = self._chunk(content="| ignored |", rows=(("A",),))
        assert _assets().chunk_content(chunk) == [["A"]]

    def test_absent_rows_fall_back_to_markdown(self) -> None:
        assert _assets().chunk_content(self._chunk()) == "| A | B |"

    def test_empty_rows_do_not_become_an_empty_table(self) -> None:
        """An empty grid carries no content, so the element is dropped rather
        than emitted as an empty TABLE.
        """
        chunk = self._chunk(content="", rows=())
        assert not _assets().chunk_content(chunk)

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

    def test_a_user_option_passes_through_verbatim(self) -> None:
        """Parsecraft owns the option names, so kompend forwards them unchanged."""
        parser = _routing_parser(backend_options={"mineru": {"effort": "high"}})
        assert parser._backend_options() == {"mineru": {"effort": "high"}}

    def test_two_backends_land_in_one_dict(self) -> None:
        parser = _routing_parser(
            backend_options={
                "mineru": {"effort": "xhigh"},
                "liteparse": {"extract_vector_graphics": True},
            }
        )
        assert parser._backend_options() == {
            "mineru": {"effort": "xhigh"},
            "liteparse": {"extract_vector_graphics": True},
        }

    def test_extract_images_is_derived_from_the_image_mode(self, tmp_path: Path) -> None:
        """A run that stores figures asks the backend for them; nothing else does.

        ``liteparse`` leads the default route for a text PDF and its engine only
        crops images when asked, so an unset option is what made the figure
        pipeline silently empty. The mode is the only input: asking for figures
        while storing none would pay the crop cost for nothing.
        """
        adapter = _adapter()
        file_mode = adapter.ParsecraftParser(images_dir=tmp_path, image_mode=ImageMode.FILE)
        assert file_mode._backend_options() == {"liteparse": {"extract_images": True}}
        base64_mode = adapter.ParsecraftParser(images_dir=tmp_path, image_mode=ImageMode.BASE64)
        assert base64_mode._backend_options() == {"liteparse": {"extract_images": True}}
        assert adapter.ParsecraftParser(images_dir=tmp_path, image_mode=ImageMode.NONE)._backend_options() == {}
        # No cache directory means no place to put a crop, so none is requested.
        assert adapter.ParsecraftParser(images_dir=None, image_mode=ImageMode.FILE)._backend_options() == {}

    def test_a_user_option_wins_over_the_derived_image_flag(self, tmp_path: Path) -> None:
        """``[routing.backend_options]`` is the escape hatch, not a casualty."""
        adapter = _adapter()
        parser = adapter.ParsecraftParser(
            images_dir=tmp_path,
            image_mode=ImageMode.FILE,
            parsecraft_config=_routing(backend_options={"liteparse": {"extract_images": False}}),
        )
        assert parser._backend_options() == {"liteparse": {"extract_images": False}}

    def test_options_reach_the_constraints(self, parser: Any, markdown_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end: the built dict must be handed to ``constraints_for``."""
        adapter = _adapter()
        seen: list[object] = []
        real = adapter.constraints_for

        def spy(config: object, environment: object, **kwargs: Any) -> Any:
            constraints = real(config, environment, **kwargs)
            seen.append(constraints.backend_options)
            return constraints

        monkeypatch.setattr(adapter, "constraints_for", spy)
        parser.parse(markdown_file)
        assert seen[-1] == {}
        _routing_parser(backend_options={"mineru": {"effort": "high"}}).parse(markdown_file)
        assert seen[-1] == {"mineru": {"effort": "high"}}


class TestEnginesWiring:
    """``[routing] engines`` overrides the probed engine set."""

    def test_unset_keeps_the_probe_and_empty_key(self) -> None:
        adapter = _adapter()
        parser = adapter.ParsecraftParser()
        assert parser.parsecraft_settings().routing.engines == []
        assert _routing_part(parser) == ""

    def test_names_appear_in_the_cache_key_sorted(self) -> None:
        _adapter()
        parser = _routing_parser(engines=["mineru", "marker"])
        assert _routing_part(parser) == "engines:marker,mineru"

    def test_backend_options_segment_only_when_set(self) -> None:
        _adapter()
        on = _routing_parser(backend_options={"liteparse": {"extract_vector_graphics": True}})
        assert _routing_part(on) == "backend_options:liteparse:extract_vector_graphics=True"
        off = _routing_parser(backend_options={"liteparse": {"extract_vector_graphics": False}})
        assert _routing_part(off) == "backend_options:liteparse:extract_vector_graphics=False"

    def test_engines_reach_the_constraints(self, parser: Any, markdown_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _adapter()
        seen: list[object] = []
        real = adapter.constraints_for

        def spy(config: object, environment: object, **kwargs: Any) -> Any:
            constraints = real(config, environment, **kwargs)
            seen.append(constraints.engines)
            return constraints

        monkeypatch.setattr(adapter, "constraints_for", spy)
        parser.parse(markdown_file)
        # No `[routing] engines` in the config, so kompend's probed set fills in.
        assert seen[-1] == frozenset(parser._ensure_environment().engines)
        _routing_parser(engines=["tesseract"]).parse(markdown_file)
        assert seen[-1] == frozenset({"tesseract"})


class TestBackendOverride:
    """``[routing] preferred_backend`` / ``only_backend``.

    parsecraft owns routing, so these pass a single preference or pin through
    to its own parameters — they are not a kompend-side routing list.
    """

    def test_unset_leaves_both_none(self) -> None:
        adapter = _adapter()
        parser = adapter.ParsecraftParser()
        assert parser.parsecraft_settings().routing.preferred_backend is None
        assert parser.parsecraft_settings().routing.only_backend is None
        assert _routing_part(parser) == ""

    def test_unknown_backend_names_the_registered_set(self) -> None:
        known = {descriptor.name for descriptor in default_registry.list_backends()}
        # parsecraft validates the names when the constraints are built — at the
        # first parse, not at construction, so the TOML stays the single place a
        # typo has to be caught.
        with pytest.raises(ConfigError, match="preferred_backend") as caught:
            _routing_parser(preferred_backend="not-a-backend").parsecraft_settings().routing.routing_constraints(
                installed_extras=set(),
                formats={"text/markdown"},
            )
        for name in known:
            assert name in str(caught.value)
        with pytest.raises(ConfigError, match="only_backend"):
            _routing_parser(only_backend="also-not-a-backend").parsecraft_settings().routing.routing_constraints(
                installed_extras=set(),
                formats={"text/markdown"},
            )

    def test_registered_names_are_accepted(self) -> None:
        chosen = self._a_backend()
        assert _routing_parser(preferred_backend=chosen).parsecraft_settings().routing.preferred_backend == chosen
        assert _routing_parser(only_backend=chosen).parsecraft_settings().routing.only_backend == chosen

    def test_each_names_its_own_cache_segment(self) -> None:
        chosen = self._a_backend()
        assert _routing_part(_routing_parser(preferred_backend=chosen)) == f"preferred_backend:{chosen}"
        assert _routing_part(_routing_parser(only_backend=chosen)) == f"only_backend:{chosen}"

    def test_overrides_reach_the_constraints(self, parser: Any, markdown_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _adapter()
        seen: dict[str, object] = {}
        real = adapter.constraints_for

        def spy(config: object, environment: object, **kwargs: Any) -> Any:
            constraints = real(config, environment, **kwargs)
            seen["preferred_backend"] = constraints.preferred_backend
            seen["only_backend"] = constraints.only_backend
            return constraints

        monkeypatch.setattr(adapter, "constraints_for", spy)
        parser.parse(markdown_file)
        assert seen["preferred_backend"] is None
        assert seen["only_backend"] is None
        chosen = self._a_markdown_backend()
        _routing_parser(preferred_backend=chosen, only_backend=chosen).parse(markdown_file)
        assert seen["preferred_backend"] == chosen
        assert seen["only_backend"] == chosen

    def test_pinning_a_backend_that_cannot_read_the_source_fails_loudly(self, markdown_file: Path) -> None:
        """A pin the source rules out is an error, never a silent reroute."""
        pinned = self._a_non_markdown_backend()
        with pytest.raises(NoEligibleBackendError):
            _routing_parser(only_backend=pinned).parse(markdown_file)

    @staticmethod
    def _a_markdown_backend() -> str:
        for descriptor in default_registry.list_backends():
            if _claims_markdown(descriptor):
                return descriptor.name
        raise AssertionError("no registered backend claims text/markdown")

    @staticmethod
    def _a_non_markdown_backend() -> str:
        for descriptor in default_registry.list_backends():
            if not _claims_markdown(descriptor):
                return descriptor.name
        raise AssertionError("every registered backend claims text/markdown")

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
        with caplog.at_level("WARNING", logger="kompend.preprocessing.parsers.parsecraft.adapter"):
            _routing_parser(backend_options={"liteparse": {"extract_vector_graphics": True}}).parse(markdown_file)
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
        with caplog.at_level("WARNING", logger="kompend.preprocessing.parsers.parsecraft.adapter"):
            _routing_parser(backend_options={"liteparse": {"extract_vector_graphics": True}}).parse(markdown_file)
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


class TestDoclingThreadLimit:
    """The ONNX thread bound reaches the engine before it loads."""

    def test_a_thread_limit_is_applied_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The engine reads DOCLING_RS_PDF_THREADS when it builds its
        # converter (on the first conversion), so the value must be in the
        # environment before that — the parser does it on construction.
        monkeypatch.delenv("DOCLING_RS_PDF_THREADS", raising=False)
        adapter = _adapter()
        adapter.ParsecraftParser(parsecraft_docling_threads=1)
        assert os.environ["DOCLING_RS_PDF_THREADS"] == "1"

    def test_no_limit_leaves_the_environment_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DOCLING_RS_PDF_THREADS", raising=False)
        adapter = _adapter()
        adapter.ParsecraftParser()
        assert "DOCLING_RS_PDF_THREADS" not in os.environ

    def test_the_configured_default_bounds_the_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The default is a bound, not an engine-native value: it reaches the
        # process even when this run never routes a page to docling.
        monkeypatch.delenv("DOCLING_RS_PDF_THREADS", raising=False)

        configured = IngestConfig().parsecraft_docling_threads
        adapter = _adapter()
        adapter.ParsecraftParser(parsecraft_docling_threads=configured)
        assert os.environ["DOCLING_RS_PDF_THREADS"] == str(configured)

    def test_an_explicit_engine_value_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Someone who set the engine's own variable by name asked for that
        # number; overwriting it here would silently undo their choice.
        monkeypatch.setenv("DOCLING_RS_PDF_THREADS", "2")
        adapter = _adapter()
        adapter.ParsecraftParser(parsecraft_docling_threads=4)
        assert os.environ["DOCLING_RS_PDF_THREADS"] == "2"


class TestHeadroom:
    """The memory budget follows the routing plan, not the file suffix.

    A PDF is not inherently heavy: liteparse reads it in a few hundred
    megabytes, docling needs gigabytes. Classifying by suffix refused every
    PDF in a liteparse vault with docling's budget, which is the bug this
    class pins down.
    """

    def test_a_liteparse_plan_is_not_held_to_the_engine_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _adapter()
        monkeypatch.setattr("kompend.preprocessing.host_memory.available_commit_bytes", lambda: 1)
        monkeypatch.setattr(_conversion(), "plan_with_judge", lambda *args, **kwargs: _planned("liteparse"))
        parser = adapter.ParsecraftParser(parse_min_free_commit_bytes=4096 * MEGABYTE)
        parser._ensure_headroom(object(), object(), object(), Path("report.pdf"))

    def test_a_docling_plan_below_the_budget_stops_the_parse_before_any_conversion(
        self,
        markdown_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _adapter()
        monkeypatch.setattr("kompend.preprocessing.host_memory.available_commit_bytes", lambda: 512 * MEGABYTE)
        monkeypatch.setattr(adapter, "analyze_source", lambda *args, **kwargs: object())
        monkeypatch.setattr(_conversion(), "plan_with_judge", lambda *args, **kwargs: _planned("liteparse", "docling"))
        monkeypatch.setattr(_conversion(), "execute", _never_called)
        parser = adapter.ParsecraftParser(parse_min_free_commit_bytes=4096 * MEGABYTE)
        with pytest.raises(InsufficientHostMemoryError, match="docling"):
            parser.parse(markdown_file)

    def test_a_zero_budget_plans_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _adapter()
        monkeypatch.setattr(_conversion(), "plan_with_judge", _never_called)
        adapter.ParsecraftParser()._ensure_headroom(object(), object(), object(), Path("report.pdf"))

    def test_from_options_converts_the_configured_megabytes_to_bytes(self) -> None:
        adapter = _adapter()
        parser = adapter.ParsecraftParser.from_options(ParserOptions(parse_min_free_commit_mb=4096))
        assert parser._min_free_commit_bytes == 4096 * MEGABYTE


class TestImageExtraction:
    """Figure crops from the IR reach the vault's image cache.

    The IR names the asset a chunk owns in the chunk metadata; the store
    decides where the bytes land, and the element carries the reference.
    """

    def test_a_figure_chunk_carries_the_stored_image(self, tmp_path: Path) -> None:
        document = _document([_page_with_asset(_figure_chunk(), "asset-1")])
        elements = _assets().map_document(document, Path("doc.pdf"), images=_store(tmp_path, ImageMode.FILE))
        image = next(element for element in elements if element.type is ElementType.IMAGE)
        stored = Path(image.metadata["image_file"])
        assert stored.parent == tmp_path
        assert stored.read_bytes() == _PNG_BYTES
        assert image.content == str(stored)

    def test_the_figures_own_text_rides_along_as_its_description(self, tmp_path: Path) -> None:
        document = _document([_page_with_asset(_figure_chunk(content="Figure 1 — the block diagram"), "asset-1")])
        elements = _assets().map_document(document, Path("doc.pdf"), images=_store(tmp_path, ImageMode.FILE))
        image = next(element for element in elements if element.type is ElementType.IMAGE)
        assert image.metadata["description"] == "Figure 1 — the block diagram"

    def test_a_figure_without_a_caption_still_carries_its_image(self, tmp_path: Path) -> None:
        # The docling backend emits a figure chunk whose content is the
        # caption, and the caption may be empty. The image must survive
        # that: it is the content of the element, the caption is only ever
        # a description.
        document = _document([_page_with_asset(_figure_chunk(content=""), "asset-1")])
        elements = _assets().map_document(document, Path("doc.pdf"), images=_store(tmp_path, ImageMode.FILE))
        image = next(element for element in elements if element.type is ElementType.IMAGE)
        assert Path(image.metadata["image_file"]).exists()
        assert image.content == image.metadata["image_file"]
        assert "description" not in image.metadata

    def test_base64_mode_keeps_the_payload_inline(self, tmp_path: Path) -> None:
        document = _document([_page_with_asset(_figure_chunk(), "asset-1")])
        elements = _assets().map_document(document, Path("doc.pdf"), images=_store(tmp_path, ImageMode.BASE64))
        image = next(element for element in elements if element.type is ElementType.IMAGE)
        assert image.metadata["image_base64"].startswith("data:image/png;base64,")
        assert "image_file" not in image.metadata
        assert not list(tmp_path.iterdir())

    def test_none_mode_extracts_no_images(self, tmp_path: Path) -> None:
        document = _document([_page_with_asset(_figure_chunk(content="Figure 1"), "asset-1")])
        elements = _assets().map_document(document, Path("doc.pdf"), images=_store(tmp_path, ImageMode.NONE))
        # No image to point at: the figure's own text stays, as text.
        assert [element.type for element in elements] == [ElementType.TEXT]
        assert elements[0].content == "Figure 1"
        assert "image_file" not in elements[0].metadata


class TestChunkMetadataProjection:
    """The IR's kind-specific hints reach the element, on named keys only.

    ``chunk_to_element`` projects a fixed allowlist of ``StructuredChunk.metadata``
    keys onto the element so chunking, export and the VLM enricher read one
    vocabulary. A key that is not on the allowlist stays in the IR: it must
    never surface in a cached element, so an unpinned projection is a bug.
    """

    def test_figure_alt_text_survives_the_projection(self) -> None:
        chunk = _assets().chunk_to_element(
            StructuredChunk(
                id="fig-1",
                kind=ChunkKind.FIGURE,
                content="![diagram](diagram.png)",
                page_number=1,
                reading_order=0,
                metadata={"image_alt": "System architecture"},
            ),
            Path("doc.md"),
            0,
            image_metadata={MetadataField.IMAGE_FILE: "assets/fig.png"},
        )
        assert chunk is not None
        assert chunk.metadata[MetadataField.IMAGE_ALT] == "System architecture"

    def test_code_language_survives_the_projection(self) -> None:
        chunk = _assets().chunk_to_element(
            StructuredChunk(
                id="c1",
                kind=ChunkKind.CODE,
                content="print(1)",
                page_number=1,
                reading_order=0,
                metadata={"language": "python"},
            ),
            Path("doc.md"),
            0,
        )
        assert chunk is not None
        assert chunk.metadata[MetadataField.LANGUAGE] == "python"

    def test_unknown_ir_keys_stay_in_the_ir(self) -> None:
        chunk = _assets().chunk_to_element(
            StructuredChunk(
                id="p1",
                kind=ChunkKind.PARAGRAPH,
                content="Body",
                page_number=1,
                reading_order=0,
                metadata={"caption": "never projected", "context": "never projected", "list_type": "never projected"},
            ),
            Path("doc.md"),
            0,
        )
        assert chunk is not None
        assert "caption" not in chunk.metadata
        assert "context" not in chunk.metadata
        assert "list_type" not in chunk.metadata

    def test_the_frontmatter_chunk_is_metadata_not_prose(self) -> None:
        # parsecraft emits a document's YAML fence as an UNKNOWN chunk flagged
        # ``frontmatter`` (adapters/markdown.py, installed 2026.11.7). The YAML
        # is metadata: it rides on the parsed document but never becomes chunk
        # text, so retrieval and generation never see it as the first paragraph.
        chunk = _assets().chunk_to_element(
            StructuredChunk(
                id="md-0000-unknown",
                kind=ChunkKind.UNKNOWN,
                content="---\ntitle: Doc\n---",
                page_number=1,
                reading_order=0,
                metadata={"frontmatter": "true"},
            ),
            Path("doc.md"),
            0,
        )
        assert chunk is not None
        assert chunk.type is ElementType.METADATA
        assert chunk.content == "---\ntitle: Doc\n---"
        # The flag stays in the IR; the element projection stays closed.
        assert chunk.metadata[MetadataField.PARSECRAFT_KIND] == "unknown"
        assert "frontmatter" not in chunk.metadata

    def test_an_unflagged_unknown_chunk_is_still_text(self) -> None:
        chunk = _assets().chunk_to_element(
            StructuredChunk(
                id="md-0001-unknown",
                kind=ChunkKind.UNKNOWN,
                content="stray block",
                page_number=1,
                reading_order=1,
                metadata={},
            ),
            Path("doc.md"),
            1,
        )
        assert chunk is not None
        assert chunk.type is ElementType.TEXT

    def test_an_asset_no_chunk_claims_becomes_its_own_element(self, tmp_path: Path) -> None:
        document = _document([_page_with_asset(_text_chunk(), "asset-1")])
        elements = _assets().map_document(document, Path("doc.pdf"), images=_store(tmp_path, ImageMode.FILE))
        assert [element.type for element in elements] == [ElementType.TEXT, ElementType.IMAGE]
        assert Path(elements[1].metadata["image_file"]).exists()

    def test_an_unreadable_asset_does_not_lose_the_text(self, tmp_path: Path) -> None:
        page = PageResult(
            page_number=1,
            blocks=[_figure_chunk(content="Figure 1")],
            assets=[ExtractedAsset(id="asset-1", kind=AssetKind.IMAGE, media_type="image/png", uri="data:image/png;base64,%%%not-base64%%%")],
        )
        document = _document([page])
        elements = _assets().map_document(document, Path("doc.pdf"), images=_store(tmp_path, ImageMode.FILE))
        assert [element.type for element in elements] == [ElementType.TEXT]
        assert elements[0].content == "Figure 1"
        assert "image_file" not in elements[0].metadata

    def test_the_file_name_pairs_the_source_stem_with_the_digest(self, tmp_path: Path) -> None:
        # Content-addressed: the digest of the bytes, with the source stem
        # kept in front for human readability.
        document = _document([_page_with_asset(_figure_chunk(), "asset-1")])
        elements = _assets().map_document(document, Path("report.pdf"), images=_store(tmp_path, ImageMode.FILE))
        image = next(element for element in elements if element.type is ElementType.IMAGE)
        digest = hashlib.sha256(_PNG_BYTES).hexdigest()[:16]
        assert Path(image.metadata["image_file"]).name == f"report__{digest}.png"

    def test_identical_bytes_keep_one_file_across_ids_and_re_parses(self, tmp_path: Path) -> None:
        # The name keys on the bytes, not the asset id: a re-parse (fresh
        # ids from the backend) reuses the file instead of accumulating copies.
        images_dir = tmp_path / "images"
        first = _document([_page_with_asset(_figure_chunk(), "asset-1")])
        _assets().map_document(first, Path("doc.pdf"), images=_store(images_dir, ImageMode.FILE))
        reparsed = _document([_page_with_asset(_figure_chunk(), "asset-99")])
        elements = _assets().map_document(reparsed, Path("doc.pdf"), images=_store(images_dir, ImageMode.FILE))
        image = next(element for element in elements if element.type is ElementType.IMAGE)
        assert [path.name for path in images_dir.iterdir()] == [Path(image.metadata["image_file"]).name]
        assert (images_dir / Path(image.metadata["image_file"]).name).read_bytes() == _PNG_BYTES

    def test_changed_bytes_get_a_new_file(self, tmp_path: Path) -> None:
        # A changed figure must not overwrite the old crop — the new bytes
        # land under their own digest.
        changed = _PNG_BYTES + b"\x00"
        rerender = PageResult(
            page_number=1,
            blocks=[_figure_chunk()],
            assets=[
                ExtractedAsset(
                    id="asset-1",
                    kind=AssetKind.IMAGE,
                    media_type="image/png",
                    uri=f"data:image/png;base64,{base64.b64encode(changed).decode('ascii')}",
                )
            ],
        )
        _assets().map_document(_document([_page_with_asset(_figure_chunk(), "asset-1")]), Path("doc.pdf"), images=_store(tmp_path, ImageMode.FILE))
        elements = _assets().map_document(_document([rerender]), Path("doc.pdf"), images=_store(tmp_path, ImageMode.FILE))
        image = next(element for element in elements if element.type is ElementType.IMAGE)
        assert len(list(tmp_path.iterdir())) == 2
        assert Path(image.metadata["image_file"]).read_bytes() == changed
        assert Path(image.metadata["image_file"]).name != hashlib.sha256(_PNG_BYTES).hexdigest()[:16] + ".png"


class TestIdentityGoldenStrings:
    """Byte-exact golden strings for both identity halves (knox-tsr4, step 0).

    The identity strings are the parse-cache key's raw material, and the
    planned split of this module moves their code twice. A refactor that
    silently reorders segments, drops one, or changes a separator would pass
    every existing test (they only assert that the identity *changes* on a
    change) while invalidating every vault's parse cache. These pins freeze
    the exact bytes for a fully pinned configuration.

    The one legitimate variable is the parsecraft version segment — it comes
    from the installed wheel and a bump flipping ``cache_identity`` is the
    documented contract, so the test sources that one segment from the same
    authority the code uses instead of freezing a stale version in.
    """

    def test_backend_identity_is_byte_stable(self, tmp_path: Path) -> None:
        expected = "backends:docling,liteparse,native-text|extras:docling,pandoc|only:mineru|preferred:docling|engines:pandoc,tesseract|ocr:on"
        assert self._pinned_parser("ollama/parable-tinyjev", tmp_path).backend_identity() == expected

    def test_cache_identity_is_byte_stable(self, tmp_path: Path) -> None:
        installed = package_version("parsecraft")
        expected = (
            "extras:docling,pandoc"
            f"|parsecraft:{installed}"
            "|judge_extras:1"
            "|judge:ollama/parable-tinyjev:quality"
            "|ocr:on"
            "|engines:pandoc,tesseract"
            "|preferred_backend:docling"
            "|only_backend:mineru"
            "|backend_options:mineru:effort=flash"
            "|images:file"
        )
        assert self._pinned_parser("ollama/parable-tinyjev", tmp_path).cache_identity() == expected

    def test_a_judge_only_change_flips_the_cache_identity_and_not_the_backend_identity(self, tmp_path: Path) -> None:
        """The trap the package AGENTS.md documents: the judge is a routing
        input (``[judge] model``), not a backend capability, so swapping the
        judge must invalidate the parse cache while leaving the backend
        identity — and therefore the backend selection key — untouched.
        """
        judged = self._pinned_parser("ollama/parable-tinyjev", tmp_path)
        unjudged = self._pinned_parser(None, tmp_path)
        assert judged.cache_identity() != unjudged.cache_identity()
        assert judged.backend_identity() == unjudged.backend_identity()

    def _pinned_parser(self, judge_model: str | None, images_dir: Path) -> Any:
        """A parser whose every routing lever is set, on a fixed fake host.

        The environment is a stand-in, not a probe: backends and extras are
        explicit, so the golden string cannot drift when the test host's
        parsecraft extras change. Images carry ``ImageMode.FILE`` so the
        ``images:file`` segment is deterministic; the directory itself is the
        test's ``tmp_path`` — the identity reads the mode and the enabled
        flag only, it never touches the filesystem.
        """
        routing: dict[str, Any] = {
            "preference": "quality",
            "only_backend": "mineru",
            "preferred_backend": "docling",
            "engines": ["pandoc", "tesseract"],
            "allow_ocr": True,
            "backend_options": {"mineru": {"effort": "flash"}},
        }
        settings = ParsecraftConfig(routing=RoutingSettings(**routing), judge=JudgeSettings(model=judge_model))
        parser = _adapter().ParsecraftParser(
            parsecraft_config=ParsecraftRoutingConfig(settings=settings, written=frozenset(routing)),
            images_dir=images_dir,
            image_mode=ImageMode.FILE,
        )
        parser._environment = SimpleNamespace(  # type: ignore[assignment] — the identities read these two names only
            backends={"docling", "liteparse", "native-text"},
            installed_extras=frozenset({"docling", "pandoc"}),
        )
        return parser


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


def _extra_pdf_backend(name: str, extra: str, vram: float) -> BackendDescriptor:
    """A minimal PDF candidate gated on *extra* (parsecraft A10): native on hosts
    that have the extra, extra-backed on hosts that lack it.
    """
    caps = BackendCapabilities(
        supported_formats=["application/pdf"],
        supports_page_ranges=True,
        supports_multi_page=True,
        gpu_requirement=0.0,
        estimated_vram_gb=vram,
        optional_dependency_group=None,
        convert_extra=extra,
        model_asset=None,
        languages=(),
        required_engine=None,
    )
    return BackendDescriptor(name=name, version="0.1.0", capabilities=caps)


def _routing_part(parser: Any) -> str:
    """The routing segment of a parsecraft ``cache_identity``.

    The whole identity also carries the host slice — installed extra set,
    parsecraft version, and the one-time ``judge_extras`` marker — whose exact
    text differs per machine and would make exact-string assertions
    host-dependent. The routing half is the part these tests are about.
    Splitting on the host/routing boundary only (the third ``|``) keeps
    multi-segment routing keys — ``preference:..|effort:..`` — intact for
    assertions, where ``rsplit('|', 1)[1]`` would have stripped the
    leading ``preference:..`` off and made the composition tests falsely green.
    """
    identity = parser.cache_identity()
    parts = identity.split("|", 3)
    return parts[3] if len(parts) >= 4 else ""


def _map(document: DocumentResult, name: str = "d.txt") -> list:

    return _assets().map_document(document, Path(name))


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


def _document(pages: list[PageResult], quality: list[QualitySignal] | None = None) -> DocumentResult:
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
        quality=quality or [],
    )


def _page(number: int, blocks: list[StructuredChunk]) -> PageResult:
    return PageResult(page_number=number, blocks=blocks)


def _figure_chunk(content: str = "") -> StructuredChunk:
    """A figure chunk that owns an image asset."""
    return StructuredChunk(
        id="fig-1",
        kind=ChunkKind.FIGURE,
        content=content,
        page_number=1,
        reading_order=0,
        metadata={"asset_id": "asset-1"},
    )


def _text_chunk() -> StructuredChunk:
    """A paragraph chunk — owns no asset."""
    return StructuredChunk(id="t1", kind=ChunkKind.PARAGRAPH, content="Body text.", page_number=1, reading_order=0)


def _page_with_asset(chunk: StructuredChunk, asset_id: str) -> PageResult:
    """A page whose blocks own one image asset."""
    return PageResult(
        page_number=chunk.page_number,
        blocks=[chunk],
        assets=[
            ExtractedAsset(
                id=asset_id,
                kind=AssetKind.IMAGE,
                media_type="image/png",
                uri=f"data:image/png;base64,{base64.b64encode(_PNG_BYTES).decode('ascii')}",
            )
        ],
    )


def _store(images_dir: Path, mode: ImageMode) -> ImageStore:
    """An image store pointed at *images_dir*."""
    return ImageStore(images_dir, mode)


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


def _claims_markdown(descriptor: Any) -> bool:
    """Whether a registered backend declares ``text/markdown``.

    Read from the backend's own capability table rather than a hardcoded name:
    which backend leads markdown is parsecraft's decision (2026.10.14 withdrew
    docling's text routes), and this test is about override plumbing.
    """
    formats = {str(fmt) for fmt in (getattr(descriptor.capabilities, "supported_formats", ()) or ())}
    return "text/markdown" in formats


def _analysis(pages: list[PageResult], images: dict[int, int]) -> AnalysisResult:
    """Analysis signals for `pages`, with `images` mapping page number to image count.

    A page missing from `images` deliberately gets no signal — that is how a host
    that never analysed the page looks to the classifier.
    """
    blocks = {page.page_number: page.blocks for page in pages}
    return AnalysisResult(
        source_hash="a" * 64,
        page_count=len(pages),
        signals=[
            PageSignal(
                page_number=number,
                has_native_text=bool(blocks.get(number)),
                text_chars=sum(len(chunk.content) for chunk in blocks.get(number, [])),
                image_count=count,
                blank=not blocks.get(number) and count == 0,
            )
            for number, count in sorted(images.items())
        ],
        diagnostics=[],
    )


def _plan(reasons: dict[int, str], blank: set[int]) -> RoutingPlan | None:
    """A routing plan whose per-page `reason` strings are exactly `reasons`.

    Pages listed in `blank` carry the planner's `degraded-blank-page` verdict; the
    rest carry no degradation code, which is what a native page normally looks
    like.
    """
    if not reasons:
        return None
    return RoutingPlan(
        primary="liteparse",
        pages=[
            PageRoute(
                page_number=number,
                intent=Intent.NATIVE,
                candidates=["liteparse"],
                chosen="liteparse",
                reason=reason,
                degradation_code=_pages().DEGRADED_BLANK_PAGE if number in blank else None,
                degradation_score=0.0 if number in blank else None,
            )
            for number, reason in sorted(reasons.items())
        ],
    )


def _planned(*backends: str) -> RoutingPlan:
    """A plan whose pages chose *backends*, one page each, in order."""
    return RoutingPlan(
        primary=backends[0],
        pages=[
            PageRoute(
                page_number=number,
                intent=Intent.NATIVE,
                candidates=[backend],
                chosen=backend,
                reason="planned for a budget test",
            )
            for number, backend in enumerate(backends, start=1)
        ],
    )


def _never_called(*args: object, **kwargs: object) -> object:
    """A seam that must not be reached; the budget check owns the decision."""
    raise AssertionError("this step must not run")


def _parser_with(**options: Any) -> Any:
    """A :class:`DocumentParser` built from *options*, for identity assertions."""
    return build_parser(ParserOptions(**options))


def _routing_parser(**routing: Any) -> Any:
    """The adapter wired to a config carrying *routing*."""
    return _adapter().ParsecraftParser(parsecraft_config=_routing(**routing))


def _judged_parser(spec: str | None, **routing: Any) -> Any:
    """The adapter wired to a config whose ``[judge] model`` names *spec*.

    The judge spec is parsecraft's setting, so the test writes it where a user
    writes it — in ``[judge]`` — rather than passing a kompend kwarg that no
    longer exists. ``None`` leaves the section empty, i.e. the rule table.
    """
    settings = ParsecraftConfig(routing=RoutingSettings(**routing), judge=JudgeSettings(model=spec))
    return _adapter().ParsecraftParser(parsecraft_config=ParsecraftRoutingConfig(settings=settings, written=frozenset(routing)))


def _adapter() -> Any:
    """Lazy accessor for the adapter *module* (see module docstring).

    The submodule, not the package: the patch seams below attach to names
    that live on the module (``analyze_source``, ``constraints_for``, …),
    and the lazy package ``__init__`` re-exports only the public surface.
    As the split lands, each seam is repointed at the module that now owns
    its name.
    """
    return importlib.import_module("kompend.preprocessing.parsers.parsecraft.adapter")


def _conversion() -> Any:
    """The judge-machinery module (``execute``, ``resolve_judge`` seams)."""
    return importlib.import_module("kompend.preprocessing.parsers.parsecraft.conversion")


def _assets() -> Any:
    """The IR-to-element module (``map_document``, ``chunk_content``, the error)."""
    return importlib.import_module("kompend.preprocessing.parsers.parsecraft.assets")


def _pages() -> Any:
    """The empty-page module (``DEGRADED_BLANK_PAGE`` and the warning block)."""
    return importlib.import_module("kompend.preprocessing.parsers.parsecraft.pages")


def _routing(**routing: Any) -> ParsecraftRoutingConfig:
    """A parsecraft config whose ``[routing]`` table carries *routing*.

    Routing is parsecraft's config now, so a test that exercises a routing
    lever builds the settings object rather than passing a kompend kwarg. Only
    the keys passed count as *written*, exactly as a TOML file listing them
    would: an omitted key stays unwritten, which is what lets the host facts
    fill it in.
    """
    return ParsecraftRoutingConfig(settings=ParsecraftConfig(routing=RoutingSettings(**routing)), written=frozenset(routing))


@pytest.fixture
def markdown_file(tmp_path: Path) -> Path:
    """A markdown file exercising headings, code, and tables end-to-end."""
    path = tmp_path / "sample.md"
    path.write_text(
        "# Main Title\n\nIntro paragraph.\n\n## Section\n\n```python\nx = 1\n```\n\n| A | B |\n|---|---|\n| 1 | 2 |\n",
        encoding="utf-8",
    )
    return path
