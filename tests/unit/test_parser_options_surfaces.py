"""Every parser-construction site must hand the parser the same options.

Three rounds of the same failure mode:

1. ``[ingest] parser_fallback`` reached only some construction sites, so the
   flag silently did not apply.
2. The ``[ingest] parsecraft_*`` routing keys did the same — and because the
   parsecraft backend folds them into its ``cache_identity()``, the two
   paths hashed to *different* parser identities and invalidated each other's
   parse cache on every alternate run.
3. ``kompend ingest`` passed ``images_dir`` alone while ``kompend compile`` also
   passed the config-derived ``image_mode``, so the two could serve each other
   parses produced under different settings — the parse-cache key covers none
   of those fields.

``ParserOptions.from_config()`` is now the single construction point. These
tests capture the options at each site and assert the whole set arrives, so a
site that builds a narrower set fails here rather than in a cache.

Routing — including which judge model to ask — is the part of the option set
kompend no longer copies: the parser receives parsecraft's config *file paths*
and reads the values itself, so a site that omits them would silently drop a
vault's ``.kompend/parsecraft.toml``. The first assertion of every site test is
therefore about the path list, and one end-to-end test proves the values behind
those paths actually land.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kompend.cli.app import app
from kompend.core.config import resolve_config
from kompend.core.paths import VaultPaths
from kompend.core.pipeline.config import PipelineConfig
from kompend.mcp.tools.pipeline import wiki_ingest
from kompend.preprocessing.parsers import ParserOptions, build_parser
from kompend.preprocessing.parsers.parsecraft_parser import ParsecraftParser
from kompend.preprocessing.parsers.parsecraft_routing import load_parsecraft_config
from tests.helpers.cli import CliRunner

_cli_runner = CliRunner()


class _Captured(Exception):
    """Sentinel raised by the stubbed parser factory to stop the command early."""


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
    monkeypatch.setattr("kompend.cli.cmds.pipeline.pipeline.build_parser", _capturing_factory(captured))

    with pytest.raises(_Captured):
        _cli_runner.invoke(app, ["--wiki-dir", str(base), "ingest"], catch_exceptions=False)

    options = captured["args"][0]
    assert _judge_spec(options) == "ollama/parable-tinyjev"
    assert options.parsecraft_config_files[0] == VaultPaths(base).parsecraft_config_file
    # No backend is named any more: parsecraft is the only parser path, so every
    # site passes options and nothing else.
    assert "name" not in captured
    _assert_full_options(options)


def test_parse_cli_passes_the_configured_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, judge="ollama", judge_model="parable-tinyjev", preference="speed", effort="medium")
    doc = base / "sources" / "note.txt"
    captured: dict[str, Any] = {}
    monkeypatch.setattr("kompend.cli.cmds.pipeline.pipeline.build_parser", _capturing_factory(captured))

    with pytest.raises(_Captured):
        _cli_runner.invoke(app, ["--wiki-dir", str(base), "parse", str(doc)], catch_exceptions=False)

    options = captured["args"][0]
    assert _judge_spec(options) == "ollama/parable-tinyjev"
    assert options.parsecraft_config_files[0] == VaultPaths(base).parsecraft_config_file
    assert "name" not in captured
    _assert_full_options(options)


def test_mcp_ingest_passes_the_configured_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path, judge="zen", judge_model="some-model", preference="balanced", effort="xhigh")
    captured: dict[str, Any] = {}
    monkeypatch.setattr("kompend.mcp.tools.pipeline.build_parser", _capturing_factory(captured))

    with pytest.raises(_Captured):
        wiki_ingest(wiki_dir=str(base))

    options = captured["args"][0]
    assert _judge_spec(options) == "zen/some-model"
    assert options.parsecraft_config_files[0] == VaultPaths(base).parsecraft_config_file
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

    assert pc.parser_options == ParserOptions.from_config(cfg, base_dir=pc.paths.base_dir)
    _assert_full_options(pc.parser_options)
    assert pc.parser_options.parsecraft_config_files[0] == VaultPaths(base).parsecraft_config_file


def test_the_vault_routing_values_reach_the_parser(tmp_path: Path) -> None:
    """End-to-end: the paths in ``ParserOptions`` really do carry the values.

    The site tests only prove the path list arrives. This one builds the parser
    from that list and reads the routing table back, so a vault-local file that
    parsecraft's engine never picked up would fail here instead of silently
    routing under another vault's settings.
    """
    base = _vault(
        tmp_path,
        judge="ollama",
        judge_model="parable-tinyjev",
        preference="quality",
        effort="high",
        allow_ocr=False,
        engines='"pandoc"',
        vector_graphics=True,
        preferred_backend="liteparse",
    )
    cfg = resolve_config(base_dir=base)
    options = ParserOptions.from_config(cfg, base_dir=base)

    parser = build_parser(options).parser
    assert isinstance(parser, ParsecraftParser)
    assert _judge_spec(options) == "ollama/parable-tinyjev"
    routing = parser.parsecraft_settings().routing

    assert routing.preference.value == "quality"
    assert routing.allow_ocr is False
    assert routing.engines == ["pandoc"]
    assert routing.preferred_backend == "liteparse"
    assert routing.only_backend is None
    assert routing.backend_options == {
        "mineru": {"effort": "high"},
        "liteparse": {"extract_vector_graphics": True},
    }


def _judge_spec(options: ParserOptions) -> str | None:
    """The judge the config files behind *options* actually name.

    Read through parsecraft's engine rather than off a kompend attribute: the
    point of these tests is that the *paths* a site hands over carry the
    settings, so anything else would test kompend's copy of a value it no
    longer keeps.
    """
    return load_parsecraft_config(options.parsecraft_config_files).judge_spec


def _assert_full_options(options: ParserOptions) -> None:
    """Every site must hand the parser the *whole* option set.

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
    lines = ["[compile.llm]", 'default_model = "ollama/test-model"']
    # ``PipelineConfig.from_resolved`` refuses an empty default_model; set one
    # so the pipeline-config test does not depend on a global config file.
    (base / ".kompend" / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Routing lives in parsecraft's own file, which is a PROJECT-layer file for
    # its engine: plain keys first, sub-tables last (a sub-table header swallows
    # every plain key written after it).
    routing = []
    if judge is not None and judge_model is not None:
        # The judge is parsecraft's setting too: ``[judge] model``, in the same
        # file as ``[routing]``, ahead of it only for readability.
        routing += ["[judge]", f'model = "{judge}/{judge_model}"', ""]
    routing.append("[routing]")
    if preference is not None:
        routing.append(f'preference = "{preference}"')
    if allow_ocr is not None:
        routing.append(f"allow_ocr = {str(allow_ocr).lower()}")
    if engines is not None:
        routing.append(f"engines = [{engines}]")
    if preferred_backend is not None:
        routing.append(f'preferred_backend = "{preferred_backend}"')
    if only_backend is not None:
        routing.append(f'only_backend = "{only_backend}"')
    if effort is not None:
        routing += ["", "[routing.backend_options.mineru]", f'effort = "{effort}"']
    if vector_graphics is not None:
        routing += ["", "[routing.backend_options.liteparse]", f"extract_vector_graphics = {str(vector_graphics).lower()}"]
    (base / ".kompend" / "parsecraft.toml").write_text("\n".join(routing) + "\n", encoding="utf-8")
    (base / "sources" / "note.txt").write_text("hello", encoding="utf-8")
    return base
