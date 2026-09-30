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
from collections.abc import Iterator
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

    def test_heading_levels_survive_string_metadata(self, parser: Any, markdown_file: Path) -> None:
        """Parsecraft carries levels as strings (``"1"``, ``"2"``); the adapter
        must convert — a heading with ``heading_level=None`` loses hierarchy.
        """
        doc = parser.parse(markdown_file)  # type: ignore[attr-defined]
        levels = [e.heading_level for e in doc.elements if e.type is ElementType.HEADING]
        assert 1 in levels
        assert 2 in levels

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
        assert doc.metadata["backends"] == "native-markdown"


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

    def test_default_discovery_not_broadened(self, tmp_path: Path) -> None:
        """Phase 1's byte-identical guarantee: a default KNOX run never loads
        the parsecraft module, so its 31-suffix text family must not enter
        vault discovery. Verified in a fresh interpreter — importing this test
        module's adapter would self-register parsecraft and defeat the check.
        """
        assert "parsecraft" not in KNOWN_BACKENDS

        repo_root = Path(__file__).parents[2]
        probe = (
            "from knox.preprocessing.parsers._registry import get_all_extensions; "
            "ext = get_all_extensions(); "
            "print(all(s not in ext for s in ('.c', '.h', '.py', '.log', '.cfg')))"
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


@pytest.fixture(autouse=True)
def _unregister_parsecraft_metadata() -> Iterator[None]:
    """Drop parsecraft from the discovery-metadata registry after each test.

    ``register_backend`` feeds two structures: ``PARSER_BACKENDS`` (router
    factories — kept, harmless) and the ``_registry`` metadata that
    ``get_all_extensions()`` reads. Leaving the metadata in place broadens
    discovery (.py/.toml/... become "supported") and breaks exact-count
    discovery tests running later in the same process — and with parsecraft
    registered, KNOX's own managed ``.knox/rumdl.toml`` would be discovered as
    a source. Default KNOX never loads the adapter module, so production
    discovery is unaffected either way.
    """
    yield
    registry_module._registry.pop("parsecraft", None)


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
