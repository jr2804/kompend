"""``[ingest] parser_fallback`` must reach every CLI/MCP router-construction site.

Regression: `knox parse`, `knox ingest`, and MCP `wiki_ingest` built a router
without passing the config flag, so `parser_fallback = false` silently didn't
apply. `knox parse` additionally bypassed fallback by calling ``route().parse()``
(single backend) instead of ``router.parse()`` (fallback-aware chain).

Same failure mode, second flag: the ``[ingest] parsecraft_*`` judge keys must
reach the same sites. Missing them left `knox ingest`/`knox parse` routing by
the deterministic rule table while `knox compile` used the configured judge —
and because the parsecraft backend folds the judge into its
``cache_identity()``, the two paths hashed to different parser fingerprints and
invalidated each other's parse cache on every alternate run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from knox.cli.app import app
from knox.core.config import resolve_config
from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.core.pipeline.config import PipelineConfig
from knox.mcp.tools.pipeline import wiki_ingest
from knox.preprocessing.parsers import ParserBuildOptions
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


def test_ingest_cli_passes_judge_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, parser_fallback=True, judge="ollama", judge_model="parable-tinyjev", preference="quality")
    captured: dict[str, Any] = {}
    monkeypatch.setattr("knox.cli.cmds.pipeline.pipeline.build_default_router", _capturing_factory(captured))

    with pytest.raises(_Captured):
        _cli_runner.invoke(app, ["--wiki-dir", str(base), "ingest"], catch_exceptions=False)

    options = captured["args"][0]
    assert options.parsecraft_judge == "ollama"
    assert options.parsecraft_judge_model == "parable-tinyjev"
    assert options.parsecraft_preference == "quality"
    _assert_full_options(options)


def test_parse_cli_passes_judge_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, parser_fallback=True, judge="ollama", judge_model="parable-tinyjev", preference="speed")
    doc = base / "sources" / "note.txt"
    captured: dict[str, Any] = {}
    monkeypatch.setattr("knox.cli.cmds.pipeline.pipeline.build_default_router", _capturing_factory(captured))

    with pytest.raises(_Captured):
        _cli_runner.invoke(app, ["--wiki-dir", str(base), "parse", str(doc)], catch_exceptions=False)

    options = captured["args"][0]
    assert options.parsecraft_judge == "ollama"
    assert options.parsecraft_judge_model == "parable-tinyjev"
    assert options.parsecraft_preference == "speed"
    _assert_full_options(options)


def test_mcp_ingest_passes_judge_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, parser_fallback=True, judge="zen", judge_model="some-model", preference="balanced")
    captured: dict[str, Any] = {}
    monkeypatch.setattr("knox.mcp.tools.pipeline.build_default_router", _capturing_factory(captured))

    with pytest.raises(_Captured):
        wiki_ingest(wiki_dir=str(base))

    options = captured["args"][0]
    assert options.parsecraft_judge == "zen"
    assert options.parsecraft_judge_model == "some-model"
    assert options.parsecraft_preference == "balanced"
    _assert_full_options(options)


def test_pipeline_config_carries_config_derived_parser_options(tmp_path: Path) -> None:
    """``build_pipeline`` reads ``config.parser_options``; it must be the
    config-derived set, not the dataclass default.
    """
    base = _vault(tmp_path, parser_fallback=True, judge="ollama", judge_model="parable-tinyjev", preference="quality")
    cfg = resolve_config(base_dir=base)

    pc = PipelineConfig.from_resolved(cfg, source_dir=base / "sources", output_dir=base / "wiki", input_files=None)

    assert pc.parser_options.parsecraft_judge == "ollama"
    assert pc.parser_options.parsecraft_judge_model == "parable-tinyjev"
    assert pc.parser_options.parsecraft_preference == "quality"
    assert pc.parser_options.pdf_config == cfg.knox.compile.pdf
    assert pc.parser_options.image_mode == cfg.knox.compile.image_mode


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


def _assert_full_options(options: ParserBuildOptions) -> None:
    """Every site must hand the router the *whole* option set.

    ``pdf_config`` is the field that actually drifted: ``knox ingest`` passed
    ``images_dir`` alone, so it parsed with parser defaults while
    ``knox compile`` parsed with ``[compile.pdf]`` — under one parse cache.
    """
    assert options.pdf_config is not None, "pdf_config must come from [compile.pdf]"
    assert options.image_mode is not None
    assert options.images_dir is not None


def _vault(
    tmp_path: Path,
    *,
    parser_fallback: bool,
    judge: str | None = None,
    judge_model: str | None = None,
    preference: str | None = None,
) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".knox").mkdir(parents=True)
    lines = ["[ingest]", f"parser_fallback = {str(parser_fallback).lower()}"]
    if judge is not None:
        lines.append(f'parsecraft_judge = "{judge}"')
    if judge_model is not None:
        lines.append(f'parsecraft_judge_model = "{judge_model}"')
    if preference is not None:
        lines.append(f'parsecraft_preference = "{preference}"')
    # ``PipelineConfig.from_resolved`` refuses an empty default_model; set one
    # so the pipeline-config test does not depend on a global config file.
    lines += ["", "[compile.llm]", 'default_model = "ollama/test-model"']
    (base / ".knox" / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (base / "sources" / "note.txt").write_text("hello", encoding="utf-8")
    return base


def _chain(*parsers: type[BaseParser], fallback: bool) -> ParserRouter:
    router = ParserRouter(fallback=fallback)
    for parser in parsers:
        router.register(parser())
    return router
