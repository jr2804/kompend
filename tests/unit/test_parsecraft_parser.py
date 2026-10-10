"""Tests for the parsecraft adapter (``knox.preprocessing.parsers.parsecraft_parser``).

Fidelity contract: parsecraft IR → KNOX elements must preserve element kind,
page numbers, reading order, heading levels, and table shape — and must NOT
emit content when parsecraft genuinely produced none. The registry contract is
also verified: parsecraft must NOT broaden default discovery (Phase 1's
byte-identical guarantee), only explicit selection may.

The adapter module is imported lazily inside each test/fixture: importing it
triggers ``register_backend("parsecraft")`` (KNOX's module-level
self-registration convention), which broadens ``get_all_extensions()`` for the
rest of the process and breaks exact-count discovery tests elsewhere in the
suite. Default KNOX runs never import the module, so the guarantee holds there.
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
from parsecraft import pipeline
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

from knox.core.models import ElementType
from knox.preprocessing.parsers import _registry as registry_module
from knox.preprocessing.parsers._registry import KNOWN_BACKENDS

pytest.importorskip("parsecraft", reason="parsecraft extra not installed")

#: Resolved lazily via importlib: a static ``from ... import parsecraft_parser``
#: would register the backend at collection time (pyreorder hoists function-level
#: imports to module level — the TYPE_CHECKING incident, commit 010097d) and
#: broaden ``get_all_extensions()`` for the rest of the test process.

# ── Fidelity: IR → KNOX elements (live pipeline, native-markdown) ────────


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
        doc = parser.parse(markdown_file)  # type: ignore[attr-defined]
        tables = [e for e in doc.elements if e.type is ElementType.TABLE]
        assert len(tables) == 1
        content = str(tables[0].content)
        # 2026.10.2 renders tables pipe-aligned. Alignment is a rendering
        # change; the contract is that the cell values survive it.
        assert "|   A |   B |" in content
        assert "|   1 |   2 |" in content

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
        # 2026.10.2's EXTRA_IMPORTS admits docling on hosts with the parsecraft
        # extra, and the deterministic table ranks the native-capable docling
        # ahead of native-markdown for markdown — do not "fix" this back.
        assert doc.metadata["backends"] == "docling"


# ── Failure shapes ───────────────────────────────────────────────────────


class TestFailureShapes:
    def test_total_exhaustion_raises_never_emits_placeholders(self) -> None:
        """All pages exhausted → KNOX parse failure, not empty content."""
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
        """A default KNOX run discovers the default backend's suffixes.

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
            "from knox.preprocessing.parsers._registry import get_all_extensions; "
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
        assert adapter._parse_preference("typo").value == "balanced"

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
            opts(parsecraft_judge="ollama", parsecraft_judge_model="parable-tinyjev"),
            name="parsecraft",
        )
        other_pref = registry_module.build_default_router(
            opts(parsecraft_judge="ollama", parsecraft_judge_model="parable-tinyjev", parsecraft_preference="quality"),
            name="parsecraft",
        )
        other_model = registry_module.build_default_router(
            opts(parsecraft_judge="ollama", parsecraft_judge_model="tev1"),
            name="parsecraft",
        )

        assert plain.fingerprint() != judged.fingerprint()
        assert judged.fingerprint() != other_pref.fingerprint()
        assert judged.fingerprint() != other_model.fingerprint()
        # The default configuration's key must not shift: no judge, no identity,
        # so two separately built default routers hash identically.
        assert adapter.ParsecraftParser().cache_identity() == ""
        plain_again = registry_module.build_default_router(opts(), name="parsecraft")
        assert plain_again.fingerprint() == plain.fingerprint()

    def test_build_options_carry_the_judge_config(self) -> None:
        """The `[ingest]` routing keys reach the backend through ParserBuildOptions."""
        adapter = _adapter()
        router = registry_module.build_default_router(
            registry_module.ParserBuildOptions(
                parsecraft_judge="ollama",
                parsecraft_judge_model="parable-tinyjev",
                parsecraft_preference="quality",
            ),
            name="parsecraft",
        )
        backend: Any = next(p for p in router._parsers if type(p).__name__ == "ParsecraftParser")  # noqa: SLF001 — parser identity is the contract
        assert backend._judge_name == "ollama"
        assert backend._judge_model == "parable-tinyjev"
        assert backend._judge_preference == "quality"
        assert adapter.ParsecraftParser()._judge_name == "none"


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
    return importlib.import_module("knox.preprocessing.parsers.parsecraft_parser")


@pytest.fixture
def markdown_file(tmp_path: Path) -> Path:
    """A markdown file exercising headings, code, and tables end-to-end."""
    path = tmp_path / "sample.md"
    path.write_text(
        "# Main Title\n\nIntro paragraph.\n\n## Section\n\n```python\nx = 1\n```\n\n| A | B |\n|---|---|\n| 1 | 2 |\n",
        encoding="utf-8",
    )
    return path
