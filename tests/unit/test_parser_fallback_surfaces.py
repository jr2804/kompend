"""``[ingest] parser_fallback`` must reach every CLI/MCP router-construction site.

Regression: `knox parse`, `knox ingest`, and MCP `wiki_ingest` built a router
without passing the config flag, so `parser_fallback = false` silently didn't
apply. `knox parse` additionally bypassed fallback by calling ``route().parse()``
(single backend) instead of ``router.parse()`` (fallback-aware chain).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from knox.cli.app import app
from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.mcp.tools.pipeline import wiki_ingest
from knox.preprocessing.parsers.base import BaseParser, FileExtension, ParserRouter

_cli_runner = CliRunner()


class _Captured(Exception):
    """Sentinel raised by the stubbed router factory to stop the command early."""


def _capturing_factory(captured: dict[str, Any]) -> Any:
    def _factory(*args: Any, **kwargs: Any) -> ParserRouter:
        captured.update(kwargs)
        captured["args"] = args
        raise _Captured

    return _factory


def _stub(name: str, *, error: Exception | None = None) -> type[BaseParser]:
    def parse(self: BaseParser, source_file: Path) -> ParsedDocument:
        if error is not None:
            raise error
        return ParsedDocument(
            source_file=source_file,
            elements=[DocumentElement(type=ElementType.TEXT, content="x")],
            title=name,
        )

    return type(name, (BaseParser,), {"parse": parse, "supported_extensions": {FileExtension.TXT}})


FirstFail = _stub("FirstFail", error=RuntimeError("libreoffice missing"))
SecondOk = _stub("SecondOk")


# ── (a) the flag reaches each construction site ──────────────────────────


def test_parse_cli_passes_parser_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, parser_fallback=False)
    doc = base / "sources" / "note.txt"
    captured: dict[str, Any] = {}
    monkeypatch.setattr("knox.cli.cmds.pipeline.pipeline.build_default_router", _capturing_factory(captured))

    with pytest.raises(_Captured):
        _cli_runner.invoke(app, ["--wiki-dir", str(base), "parse", str(doc)], catch_exceptions=False)

    assert captured.get("fallback") is False


def test_ingest_cli_passes_parser_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, parser_fallback=False)
    captured: dict[str, Any] = {}
    monkeypatch.setattr("knox.cli.cmds.pipeline.pipeline.build_default_router", _capturing_factory(captured))

    with pytest.raises(_Captured):
        _cli_runner.invoke(app, ["--wiki-dir", str(base), "ingest"], catch_exceptions=False)

    assert captured.get("fallback") is False


def test_mcp_ingest_passes_parser_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, parser_fallback=False)
    captured: dict[str, Any] = {}
    monkeypatch.setattr("knox.mcp.tools.pipeline.build_default_router", _capturing_factory(captured))

    with pytest.raises(_Captured):
        wiki_ingest(wiki_dir=str(base))

    assert captured.get("fallback") is False


# ── (b) the parse path runs the fallback chain ───────────────────────────


def test_parse_cli_falls_through_to_next_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`knox parse` must use ``router.parse`` (chain), not ``route().parse()``.

    Fails on the pre-fix code: ``route()`` returns the first backend only, so a
    failing first backend aborts instead of falling through to the second.
    """
    base = _vault(tmp_path, parser_fallback=True)
    doc = base / "sources" / "legacy.txt"
    doc.write_text("x", encoding="utf-8")

    def _factory(*args: Any, **kwargs: Any) -> ParserRouter:  # noqa: ARG001
        return _chain(FirstFail, SecondOk, fallback=True)

    monkeypatch.setattr("knox.cli.cmds.pipeline.pipeline.build_default_router", _factory)

    result = _cli_runner.invoke(app, ["--wiki-dir", str(base), "parse", str(doc)])
    assert result.exit_code == 0, result.output
    assert "1 element(s)" in result.output


def _vault(tmp_path: Path, *, parser_fallback: bool) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".knox").mkdir(parents=True)
    (base / ".knox" / "config.toml").write_text(f"[ingest]\nparser_fallback = {str(parser_fallback).lower()}\n", encoding="utf-8")
    (base / "sources" / "note.txt").write_text("hello", encoding="utf-8")
    return base


def _chain(*parsers: type[BaseParser], fallback: bool) -> ParserRouter:
    router = ParserRouter(fallback=fallback)
    for parser in parsers:
        router.register(parser())
    return router
