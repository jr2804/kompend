"""A file that blows its parse deadline must not take the ingest run with it.

The deadline exists because a backend can wedge and never return (a docling
markdown parse once held an ingest for nine hours at 98.5 % of one core). The
contract a user depends on is that such a file costs exactly one file: it is
counted as failed and the run keeps going, so the remaining sources still land.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import cast

import pytest

from kompend.core.models import ParsedDocument
from kompend.core.pipeline.bounded_parse import ParseDeadlineExceeded
from kompend.core.pipeline.ingest import ingest_documents
from kompend.preprocessing.parsers import ParserRouter

_ingest = importlib.import_module("kompend.core.pipeline.ingest")


class _FakeParser:
    """Picklable stand-in: the deadline path sends the parser to a child process."""

    def __init__(self, seen: list[str]) -> None:
        self.seen = seen

    def parse(self, target: Path) -> ParsedDocument:
        self.seen.append(Path(target).name)
        return _document(Path(target))


class _Router:
    def __init__(self) -> None:
        self.parsed: list[str] = []

    def fingerprint(self) -> str:
        return "deadline-chain"

    def route(self, source_file: Path) -> _FakeParser:
        return _FakeParser(self.parsed)


def _document(source: Path) -> ParsedDocument:
    return ParsedDocument(source_file=source, elements=[], title=source.stem)


def test_a_deadline_failure_is_counted_and_the_run_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _seed_vault(tmp_path)
    stuck = base / "sources" / "stuck.md"
    fine = base / "sources" / "fine.md"
    stuck.write_text("# stuck\n", encoding="utf-8")
    fine.write_text("# fine\n", encoding="utf-8")
    router = _Router()

    def _parse(self: object, parser: _FakeParser, target: Path) -> ParsedDocument:
        if Path(target).name == "stuck.md":
            raise ParseDeadlineExceeded(Path(target), 5.0)
        # The good file is parsed in-process: this test is about the ingest
        # bookkeeping, and a real child would spend its budget importing this
        # test module rather than the parser stack.
        return parser.parse(target)

    monkeypatch.setattr(_ingest.DeadlineWorker, "parse", _parse)

    result = ingest_documents(
        files=[stuck, fine],
        base_dir=base,
        router=cast("ParserRouter", router),
        no_cache=True,
        config=None,
        parse_timeout_s=5.0,
    )

    assert result.failed == 1
    assert result.count == 1
    assert "fine.md" in router.parsed


def _seed_vault(tmp_path: Path) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".kompend").mkdir(parents=True)
    (base / ".kompend" / "config.toml").write_text("", encoding="utf-8")
    return base
