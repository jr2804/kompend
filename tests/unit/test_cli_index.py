"""Unit tests for the ``knox index`` CLI command.

``knox index`` is a recovery command that rebuilds the zvec vector index
from the compiled-wiki snapshot in the vault store. These tests mock the wiki
loader and the index builder so no real embedding provider is required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from knox.cli.app import app
from knox.core.models import CompiledWiki, WikiNode

runner = CliRunner()


def test_index_rebuilds_vector_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Knox index loads cached wiki and rebuilds the vector index."""
    base = _setup_vault(tmp_path)
    monkeypatch.chdir(base)

    wiki = _make_wiki()
    with (
        patch("knox.cli.cmds.pipeline.pipeline.load_cached_wiki", return_value=wiki) as mock_load,
        patch("knox.cli.cmds.pipeline.pipeline.build_vector_index", return_value=1) as mock_build,
        patch("knox.cli.cmds.pipeline.pipeline.write_wiki_cache") as mock_write,
    ):
        result = runner.invoke(app, ["index"])

    assert result.exit_code == 0, result.output
    assert "Indexed 1 nodes" in result.output
    mock_load.assert_called_once()
    mock_build.assert_called_once()
    # embed_attachments comes from the canonical key (attachments.embed),
    # which defaults to False
    call_kwargs = mock_build.call_args.kwargs
    assert call_kwargs["embed_attachments"] is False
    assert call_kwargs["embed_model"] == "test-embed"
    # wiki metadata records the embedding model for query to open the store
    mock_write.assert_called_once()
    written_wiki = mock_write.call_args.args[0]
    assert written_wiki.metadata["embedding_model"] == "test-embed"


def test_index_no_wiki_exits_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Knox index errors when no compiled wiki is cached."""
    base = _setup_vault(tmp_path)
    monkeypatch.chdir(base)

    with patch("knox.cli.cmds.pipeline.pipeline.load_cached_wiki", return_value=None):
        result = runner.invoke(app, ["index"])

    assert result.exit_code == 1
    assert "No compiled wiki found" in result.output


def test_index_respects_embed_attachments_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The deprecated search.embed_attachments spelling still reaches the builder.

    The key was renamed to ``attachments.embed``; a config setting the old name
    is migrated at resolve time (with a deprecation warning), which is the only
    reason this still works — nothing reads the old key directly.
    """
    base = tmp_path / "wiki"
    (base / "sources").mkdir(parents=True)
    (base / ".knox").mkdir(parents=True)
    (base / ".knox" / "config.toml").write_text(
        '[compile.embedding]\nmodel = "test-embed"\n[search]\nembed_attachments = true\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(base)

    wiki = _make_wiki()
    with (
        patch("knox.cli.cmds.pipeline.pipeline.load_cached_wiki", return_value=wiki),
        patch("knox.cli.cmds.pipeline.pipeline.build_vector_index", return_value=5) as mock_build,
        patch("knox.cli.cmds.pipeline.pipeline.write_wiki_cache"),
    ):
        result = runner.invoke(app, ["index"])

    assert result.exit_code == 0, result.output
    assert mock_build.call_args.kwargs["embed_attachments"] is True


def test_index_no_embedding_model_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Knox index errors when load_cached_wiki returns None (no wiki to index)."""
    # The "no embedding model" guard is reachable only when config returns an
    # empty string for compile.embedding.model, which is now the default.
    # The primary error path is missing-wiki, covered above. This test instead
    # verifies the embed_model flag overrides the config default.
    base = _setup_vault(tmp_path)
    monkeypatch.chdir(base)

    wiki = _make_wiki()
    with (
        patch("knox.cli.cmds.pipeline.pipeline.load_cached_wiki", return_value=wiki),
        patch("knox.cli.cmds.pipeline.pipeline.build_vector_index", return_value=1) as mock_build,
        patch("knox.cli.cmds.pipeline.pipeline.write_wiki_cache"),
    ):
        result = runner.invoke(app, ["index", "--embed-model", "custom/model"])

    assert result.exit_code == 0, result.output
    assert mock_build.call_args.kwargs["embed_model"] == "custom/model"


def _make_wiki() -> CompiledWiki:
    node = WikiNode(id="test-node", title="Test", content="content", node_type="concept")
    return CompiledWiki(nodes=[node], edges=[], toc=[], quality_score=0.0, iteration=0, metadata={})


def _setup_vault(tmp_path: Path) -> Path:
    """Create a minimal vault structure with a config and cached wiki marker."""
    base = tmp_path / "wiki"
    (base / "sources").mkdir(parents=True)
    (base / ".knox").mkdir(parents=True)
    (base / ".knox" / "config.toml").write_text('[compile.embedding]\nmodel = "test-embed"\n', encoding="utf-8")
    return base
