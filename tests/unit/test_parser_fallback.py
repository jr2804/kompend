"""Parser-chain fallback tests (``[ingest] parser_fallback``)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from knox.core.models import ParsedDocument
from knox.preprocessing.parsers.base import BaseParser, FileExtension, ParserRouter

_LOGGER = "knox.preprocessing.parsers.base"


def _stub(
    name: str,
    extensions: set[FileExtension],
    *,
    error: Exception | None = None,
) -> type[BaseParser]:
    """Build a named stub backend so log/fingerprint assertions see real names."""

    def parse(self: BaseParser, source_file: Path) -> ParsedDocument:
        if error is not None:
            raise error
        return ParsedDocument(source_file=source_file, elements=[], title=name)

    return type(name, (BaseParser,), {"parse": parse, "supported_extensions": extensions})


DoclingStub = _stub("DoclingStub", {FileExtension.DOC}, error=RuntimeError("libreoffice missing"))
LiteparseStub = _stub("LiteparseStub", {FileExtension.DOC})
PandocStub = _stub("PandocStub", set())


def _router(*parsers: type[BaseParser], fallback: bool = True) -> ParserRouter:
    router = ParserRouter(fallback=fallback)
    for parser in parsers:
        router.register(parser())
    return router


class TestFallbackChain:
    def test_falls_through_to_next_backend(self, tmp_path: Path) -> None:
        doc = tmp_path / "legacy.doc"
        doc.write_bytes(b"\xd0\xcf\x11\xe0")  # OLE magic; never actually parsed
        router = _router(DoclingStub, LiteparseStub)

        assert router.parse(doc).title == "LiteparseStub"

    def test_doc_claimed_by_two_backends_falls_through(self, tmp_path: Path) -> None:
        doc = tmp_path / "legacy.doc"
        doc.write_text("x")
        router = _router(DoclingStub, LiteparseStub)

        result = router.parse(doc)
        assert result.source_file == doc

    def test_fallback_logs_warning_naming_file_failure_and_success(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        doc = tmp_path / "legacy.doc"
        doc.write_text("x")
        router = _router(DoclingStub, LiteparseStub)

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            router.parse(doc)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "legacy.doc" in message
        assert "DoclingStub" in message
        assert "libreoffice missing" in message
        assert "LiteparseStub" in message

    def test_no_warning_when_first_backend_succeeds(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        doc = tmp_path / "modern.doc"
        doc.write_text("x")
        router = _router(LiteparseStub)

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            router.parse(doc)

        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_all_backends_fail_raises(self, tmp_path: Path) -> None:
        doc = tmp_path / "legacy.doc"
        doc.write_text("x")
        router = _router(DoclingStub, _stub("SecondStub", {FileExtension.DOC}, error=ValueError("boom")))

        with pytest.raises(ValueError, match="Every parser failed"):
            router.parse(doc)

    def test_unclaimed_text_uses_text_fallback_only(self, tmp_path: Path) -> None:
        txt = tmp_path / "notes.txt"
        txt.write_text("plain")
        router = _router(DoclingStub)
        router.set_text_fallback(PandocStub())

        assert router.parse(txt).title == "PandocStub"

    def test_unclaimed_binary_without_backend_raises(self, tmp_path: Path) -> None:
        blob = tmp_path / "image.png"
        blob.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xff")
        router = _router(DoclingStub)

        with pytest.raises(ValueError, match="No parser found"):
            router.parse(blob)


class TestFallbackDisabled:
    def test_disabled_propagates_first_error_and_stops(self, tmp_path: Path) -> None:
        doc = tmp_path / "legacy.doc"
        doc.write_text("x")
        calls: list[str] = []

        def parse_first(self: BaseParser, source_file: Path) -> ParsedDocument:
            calls.append("first")
            raise RuntimeError("docling exploded")

        def parse_second(self: BaseParser, source_file: Path) -> ParsedDocument:
            calls.append("second")
            return ParsedDocument(source_file=source_file, elements=[])

        first = type("FirstStub", (BaseParser,), {"parse": parse_first, "supported_extensions": {FileExtension.DOC}})
        second = type("SecondStub", (BaseParser,), {"parse": parse_second, "supported_extensions": {FileExtension.DOC}})
        router = _router(first, second, fallback=False)

        with pytest.raises(RuntimeError, match="docling exploded"):
            router.parse(doc)
        assert calls == ["first"]

    def test_route_is_unaffected_by_fallback(self, tmp_path: Path) -> None:
        doc = tmp_path / "legacy.doc"
        doc.write_text("x")

        assert type(_router(DoclingStub, LiteparseStub).route(doc)).__name__ == "DoclingStub"
        assert type(_router(DoclingStub, LiteparseStub, fallback=False).route(doc)).__name__ == "DoclingStub"


class TestFingerprint:
    def test_fingerprint_changes_when_flag_flips(self) -> None:
        enabled = _router(DoclingStub, LiteparseStub).fingerprint()
        disabled = _router(DoclingStub, LiteparseStub, fallback=False).fingerprint()

        assert enabled != disabled

    def test_fingerprint_is_stable_for_same_config(self) -> None:
        assert _router(DoclingStub, LiteparseStub).fingerprint() == _router(DoclingStub, LiteparseStub).fingerprint()

    def test_fingerprint_changes_when_chain_changes(self) -> None:
        assert _router(DoclingStub).fingerprint() != _router(DoclingStub, LiteparseStub).fingerprint()
