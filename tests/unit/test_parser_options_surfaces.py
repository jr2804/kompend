"""Every router-construction site must hand the router the same options.

Three rounds of the same failure mode:

1. ``[ingest] parser_fallback`` reached only some construction sites, so the
   flag silently did not apply.
2. The ``[ingest] parsecraft_*`` judge keys did the same — and because the
   parsecraft backend folds the judge into its ``cache_identity()``, the two
   paths hashed to *different* parser fingerprints and invalidated each other's
   parse cache on every alternate run.
3. ``knox ingest`` passed ``images_dir`` alone while ``knox compile`` also
   passed ``pdf_config`` and ``image_mode``, so the two could serve each other
   parses produced under different ``[compile.pdf]`` settings — the parse-cache
   key covers none of those fields.

``ParserBuildOptions.from_config()`` is now the single construction point. These
tests capture the options at each site and assert the whole set arrives, so a
site that builds a narrower set fails here rather than in a cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from knox.cli.app import app
from knox.core.config import resolve_config
from knox.core.pipeline.config import PipelineConfig
from knox.mcp.tools.pipeline import wiki_ingest
from knox.preprocessing.parsers import ParserBuildOptions

_cli_runner = CliRunner()


class _Captured(Exception):
    """Sentinel raised by the stubbed router factory to stop the command early."""


def _capturing_factory(captured: dict[str, Any]) -> Any:
    def _factory(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        captured["args"] = args
        raise _Captured

    return _factory


def test_ingest_cli_passes_the_configured_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, judge="ollama", judge_model="parable-tinyjev", preference="quality")
    captured: dict[str, Any] = {}
    monkeypatch.setattr("knox.cli.cmds.pipeline.pipeline.build_default_router", _capturing_factory(captured))

    with pytest.raises(_Captured):
        _cli_runner.invoke(app, ["--wiki-dir", str(base), "ingest"], catch_exceptions=False)

    options = captured["args"][0]
    assert options.parsecraft_judge == "ollama"
    assert options.parsecraft_judge_model == "parable-tinyjev"
    assert options.parsecraft_preference == "quality"
    assert captured["name"] == "parsecraft"
    _assert_full_options(options)


def test_parse_cli_passes_the_configured_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, judge="ollama", judge_model="parable-tinyjev", preference="speed")
    doc = base / "sources" / "note.txt"
    captured: dict[str, Any] = {}
    monkeypatch.setattr("knox.cli.cmds.pipeline.pipeline.build_default_router", _capturing_factory(captured))

    with pytest.raises(_Captured):
        _cli_runner.invoke(app, ["--wiki-dir", str(base), "parse", str(doc)], catch_exceptions=False)

    options = captured["args"][0]
    assert options.parsecraft_judge == "ollama"
    assert options.parsecraft_judge_model == "parable-tinyjev"
    assert options.parsecraft_preference == "speed"
    # Without --parser the configured backend is used, not a hardcoded default.
    assert captured["name"] == "parsecraft"
    _assert_full_options(options)


def test_mcp_ingest_passes_the_configured_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, judge="zen", judge_model="some-model", preference="balanced")
    captured: dict[str, Any] = {}
    monkeypatch.setattr("knox.mcp.tools.pipeline.build_default_router", _capturing_factory(captured))

    with pytest.raises(_Captured):
        wiki_ingest(wiki_dir=str(base))

    options = captured["args"][0]
    assert options.parsecraft_judge == "zen"
    assert options.parsecraft_judge_model == "some-model"
    assert options.parsecraft_preference == "balanced"
    assert captured["name"] == "parsecraft"
    _assert_full_options(options)


def test_pipeline_config_builds_the_same_options_as_the_cli(tmp_path: Path) -> None:
    """``build_pipeline`` reads ``config.parser_options``; it must be the
    config-derived set — the same one the CLI and MCP sites build directly.
    """
    base = _vault(tmp_path, judge="ollama", judge_model="parable-tinyjev", preference="quality")
    cfg = resolve_config(base_dir=base)

    pc = PipelineConfig.from_resolved(cfg, source_dir=base / "sources", output_dir=base / "wiki", input_files=None)

    assert pc.parser_options == ParserBuildOptions.from_config(cfg, images_dir=pc.paths.images_dir)
    _assert_full_options(pc.parser_options)


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
    judge: str | None = None,
    judge_model: str | None = None,
    preference: str | None = None,
) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".knox").mkdir(parents=True)
    lines = ["[ingest]"]
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
