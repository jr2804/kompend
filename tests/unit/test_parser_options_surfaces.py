"""Every router-construction site must hand the router the same options.

Three rounds of the same failure mode:

1. ``[ingest] parser_fallback`` reached only some construction sites, so the
   flag silently did not apply.
2. The ``[ingest] parsecraft_*`` judge keys did the same — and because the
   parsecraft backend folds the judge into its ``cache_identity()``, the two
   paths hashed to *different* parser fingerprints and invalidated each other's
   parse cache on every alternate run.
3. ``kompend ingest`` passed ``images_dir`` alone while ``kompend compile`` also
   passed the config-derived ``image_mode``, so the two could serve each other
   parses produced under different settings — the parse-cache key covers none
   of those fields.

``ParserBuildOptions.from_config()`` is now the single construction point. These
tests capture the options at each site and assert the whole set arrives, so a
site that builds a narrower set fails here rather than in a cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from kompend.cli.app import app
from kompend.core.config import resolve_config
from kompend.core.pipeline.config import PipelineConfig
from kompend.mcp.tools.pipeline import wiki_ingest
from kompend.preprocessing.parsers import ParserBuildOptions

_cli_runner = CliRunner()


class _Captured(Exception):
    """Sentinel raised by the stubbed router factory to stop the command early."""


def test_ingest_cli_passes_the_configured_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(
        tmp_path,
        judge="ollama",
        judge_model="parable-tinyjev",
        preference="quality",
        effort="high",
        allow_ocr=False,
        engines='"mineru", "marker"',
        vector_graphics=True,
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr("kompend.cli.cmds.pipeline.pipeline.build_default_router", _capturing_factory(captured))

    with pytest.raises(_Captured):
        _cli_runner.invoke(app, ["--wiki-dir", str(base), "ingest"], catch_exceptions=False)

    options = captured["args"][0]
    assert options.parsecraft_judge == "ollama"
    assert options.parsecraft_judge_model == "parable-tinyjev"
    assert options.parsecraft_preference == "quality"
    assert options.parsecraft_effort == "high"
    assert options.parsecraft_allow_ocr is False
    assert options.parsecraft_engines == ["mineru", "marker"]
    assert options.parsecraft_extract_vector_graphics is True
    # No backend is named any more: parsecraft is the only parser path, so the
    # router takes the registry default and every site passes options only.
    assert "name" not in captured
    _assert_full_options(options)


def test_parse_cli_passes_the_configured_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, judge="ollama", judge_model="parable-tinyjev", preference="speed", effort="medium")
    doc = base / "sources" / "note.txt"
    captured: dict[str, Any] = {}
    monkeypatch.setattr("kompend.cli.cmds.pipeline.pipeline.build_default_router", _capturing_factory(captured))

    with pytest.raises(_Captured):
        _cli_runner.invoke(app, ["--wiki-dir", str(base), "parse", str(doc)], catch_exceptions=False)

    options = captured["args"][0]
    assert options.parsecraft_judge == "ollama"
    assert options.parsecraft_judge_model == "parable-tinyjev"
    assert options.parsecraft_preference == "speed"
    assert options.parsecraft_effort == "medium"
    assert "name" not in captured
    _assert_full_options(options)


def test_mcp_ingest_passes_the_configured_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, judge="zen", judge_model="some-model", preference="balanced", effort="xhigh")
    captured: dict[str, Any] = {}
    monkeypatch.setattr("kompend.mcp.tools.pipeline.build_default_router", _capturing_factory(captured))

    with pytest.raises(_Captured):
        wiki_ingest(wiki_dir=str(base))

    options = captured["args"][0]
    assert options.parsecraft_judge == "zen"
    assert options.parsecraft_judge_model == "some-model"
    assert options.parsecraft_preference == "balanced"
    assert options.parsecraft_effort == "xhigh"
    assert "name" not in captured
    _assert_full_options(options)


def _capturing_factory(captured: dict[str, Any]) -> Any:
    def _factory(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        captured["args"] = args
        raise _Captured

    return _factory


def test_pipeline_config_builds_the_same_options_as_the_cli(tmp_path: Path) -> None:
    """``build_pipeline`` reads ``config.parser_options``; it must be the
    config-derived set — the same one the CLI and MCP sites build directly.
    """
    base = _vault(
        tmp_path,
        judge="ollama",
        judge_model="parable-tinyjev",
        preference="quality",
        effort="high",
        allow_ocr=False,
        engines='"mineru", "marker"',
        vector_graphics=True,
        preferred_backend="liteparse",
    )
    cfg = resolve_config(base_dir=base)

    pc = PipelineConfig.from_resolved(cfg, source_dir=base / "sources", output_dir=base / "wiki", input_files=None)

    assert pc.parser_options == ParserBuildOptions.from_config(cfg, images_dir=pc.paths.images_dir)
    _assert_full_options(pc.parser_options)
    assert pc.parser_options.parsecraft_effort == "high"
    assert pc.parser_options.parsecraft_allow_ocr is False
    assert pc.parser_options.parsecraft_engines == ["mineru", "marker"]
    assert pc.parser_options.parsecraft_extract_vector_graphics is True
    assert pc.parser_options.parsecraft_preferred_backend == "liteparse"
    assert pc.parser_options.parsecraft_only_backend is None


def _assert_full_options(options: ParserBuildOptions) -> None:
    """Every site must hand the router the *whole* option set.

    ``image_mode`` is the field that drifts if a site assembles the options by
    hand: it is the config-derived value every parsing site must agree on, and
    the parse-cache key does not cover it.
    """
    assert options.image_mode is not None
    assert options.images_dir is not None


def _vault(
    tmp_path: Path,
    *,
    judge: str | None = None,
    judge_model: str | None = None,
    preference: str | None = None,
    effort: str | None = None,
    allow_ocr: bool | None = None,
    engines: str | None = None,
    vector_graphics: bool | None = None,
    preferred_backend: str | None = None,
    only_backend: str | None = None,
) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".kompend").mkdir(parents=True)
    lines = ["[ingest]"]
    if judge is not None:
        lines.append(f'parsecraft_judge = "{judge}"')
    if judge_model is not None:
        lines.append(f'parsecraft_judge_model = "{judge_model}"')
    if preference is not None:
        lines.append(f'parsecraft_preference = "{preference}"')
    if effort is not None:
        lines.append(f'parsecraft_effort = "{effort}"')
    if allow_ocr is not None:
        lines.append(f"parsecraft_allow_ocr = {str(allow_ocr).lower()}")
    if engines is not None:
        lines.append(f"parsecraft_engines = [{engines}]")
    if vector_graphics is not None:
        lines.append(f"parsecraft_extract_vector_graphics = {str(vector_graphics).lower()}")
    if preferred_backend is not None:
        lines.append(f'parsecraft_preferred_backend = "{preferred_backend}"')
    if only_backend is not None:
        lines.append(f'parsecraft_only_backend = "{only_backend}"')
    # ``PipelineConfig.from_resolved`` refuses an empty default_model; set one
    # so the pipeline-config test does not depend on a global config file.
    lines += ["", "[compile.llm]", 'default_model = "ollama/test-model"']
    (base / ".kompend" / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (base / "sources" / "note.txt").write_text("hello", encoding="utf-8")
    return base
