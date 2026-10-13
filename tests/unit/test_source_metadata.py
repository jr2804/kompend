"""Cross-boundary tests for source metadata (parse-cache store).

The tag/attribute readers and writers for sources were migrated from a
``.kompend/cache/<stem>-*.json`` glob (which nothing wrote) to the ``documents``
table of the single store. These tests pin the shared store path and, crucially,
that a tag set through the **MCP** tool is visible to the **CLI** reader — the
cross-boundary assertion that would have caught the silent no-op.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from kompend.cli.app import app as cli_app
from kompend.core import source_metadata
from kompend.core.pipeline.parse_cache import write_document
from kompend.core.store import vault_db_path
from kompend.mcp.tools.sources import sources_attr, sources_list, sources_tag, sources_untag

_SOURCE = "sources/doc.pdf"

_cli_runner = CliRunner()


# ── store-backed helpers ─────────────────────────────────────────────────


class TestSourceMetadataStore:
    def test_load_entries_keyed_by_source_file(self, tmp_path: Path) -> None:
        base = _seed_vault(tmp_path, ("doc-aaaaaaaa", _record()))
        entries = source_metadata.load_source_entries(base / ".kompend" / "cache")
        assert set(entries) == {_SOURCE}

    def test_apply_tag_round_trips_through_store(self, tmp_path: Path) -> None:
        base = _seed_vault(tmp_path, ("doc-aaaaaaaa", _record()))
        cache_dir = base / ".kompend" / "cache"
        change = source_metadata.apply_tag(cache_dir, _SOURCE, "deep-learning", add=True)
        assert change.changed is True
        assert source_metadata.source_tags(source_metadata.find_entry(cache_dir, _SOURCE) or {}) == ["deep-learning"]

    def test_apply_tag_missing_raises(self, tmp_path: Path) -> None:
        base = _seed_vault(tmp_path)
        with pytest.raises(LookupError):
            source_metadata.apply_tag(base / ".kompend" / "cache", _SOURCE, "x", add=True)

    def test_cached_text_and_none_for_uncached(self, tmp_path: Path) -> None:
        base = _seed_vault(tmp_path, ("doc-aaaaaaaa", _record()))
        cache_dir = base / ".kompend" / "cache"
        assert source_metadata.cached_text(cache_dir, "doc.pdf") == "cached body\n\n"
        assert source_metadata.cached_text(cache_dir, "absent.pdf") is None


# ── cross-boundary: MCP writer → CLI reader ──────────────────────────────


class TestMcpToCliVisibility:
    def test_tag_set_via_mcp_is_visible_to_cli_reader(self, tmp_path: Path) -> None:
        base = _seed_vault(tmp_path, ("doc-aaaaaaaa", _record()))

        result = sources_tag(wiki_dir=str(base), file=_SOURCE, tag="physics")
        assert result["status"] == "ok"
        assert result["changed"] is True

        # CLI reader path: the same helper `kompend sources list` uses.
        entries = source_metadata.load_source_entries(base / ".kompend" / "cache")
        assert source_metadata.source_tags(entries[_SOURCE]) == ["physics"]

    def test_mcp_list_reflects_the_tag(self, tmp_path: Path) -> None:
        base = _seed_vault(tmp_path, ("doc-aaaaaaaa", _record()))
        sources_tag(wiki_dir=str(base), file=_SOURCE, tag="physics")

        listing = sources_list(wiki_dir=str(base))
        assert listing["count"] == 1
        assert listing["sources"][0]["tags"] == ["physics"]

    def test_tag_command_end_to_end(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        base = _seed_vault(tmp_path, ("doc-aaaaaaaa", _record()))
        monkeypatch.chdir(base)

        assert cli_app is not None  # ensure CLI import side effects ran
        tagged = _cli_runner.invoke(cli_app, ["sources", "tag", _SOURCE, "physics"])
        assert tagged.exit_code == 0, tagged.output

        listed = _cli_runner.invoke(cli_app, ["sources", "list"])
        assert listed.exit_code == 0, listed.output
        assert "physics" in listed.output

    def test_attribute_set_via_mcp_is_visible_to_reader(self, tmp_path: Path) -> None:
        base = _seed_vault(tmp_path, ("doc-aaaaaaaa", _record()))
        result = sources_attr(wiki_dir=str(base), file=_SOURCE, key_value="audience=expert")
        assert result["status"] == "ok"

        entries = source_metadata.load_source_entries(base / ".kompend" / "cache")
        assert source_metadata.source_attributes(entries[_SOURCE]) == {"audience": "expert"}

    def test_untag_removes_the_tag(self, tmp_path: Path) -> None:
        base = _seed_vault(tmp_path, ("doc-aaaaaaaa", _record()))
        sources_tag(wiki_dir=str(base), file=_SOURCE, tag="physics")
        sources_untag(wiki_dir=str(base), file=_SOURCE, tag="physics")

        entries = source_metadata.load_source_entries(base / ".kompend" / "cache")
        assert source_metadata.source_tags(entries[_SOURCE]) == []


# ── clean cut ────────────────────────────────────────────────────────────


class TestLegacyFileIgnored:
    def test_legacy_cache_dir_json_is_not_read(self, tmp_path: Path) -> None:
        base = _seed_vault(tmp_path)
        legacy = base / ".kompend" / "cache" / "doc-aaaaaaaa.json"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text('{"source_file": "sources/ghost.pdf", "metadata": {}}', encoding="utf-8")

        entries = source_metadata.load_source_entries(base / ".kompend" / "cache")
        assert entries == {}


def _seed_vault(tmp_path: Path, *records: tuple[str, dict]) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".kompend").mkdir(parents=True)
    (base / ".kompend" / "config.toml").write_text("", encoding="utf-8")
    (base / "sources" / "doc.pdf").write_bytes(b"%PDF-1.4")
    db_path = vault_db_path(base / ".kompend" / "cache")
    for key, data in records:
        write_document(db_path, key, data)
    return base


def _record(**meta: object) -> dict:
    return {
        "source_file": _SOURCE,
        "title": "Doc",
        "author": "",
        "language": "en",
        "page_count": 1,
        "metadata": dict(meta),
        "elements": [{"type": "text", "content": "cached body"}],
    }
