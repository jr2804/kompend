"""Tests for knox.core.pipeline.export (``knox sources export``)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

# ── --all bulk export ──────────────────────────────────────────────────
from typer.testing import CliRunner  # noqa: E402

from knox.cli.app import app as cli_app  # noqa: E402
from knox.core.models.attachments import Attachment, AttachmentManifest
from knox.core.models.enums import AssetKind
from knox.core.pipeline.assets import AssetStore
from knox.core.pipeline.export import (
    _asset_id,
    _element_from_dict,
    _structured_dict,
    document_summary,
    export_source,
    load_cache_entry,
    render_document_markdown,
)
from knox.core.pipeline.parse_cache import write_document
from knox.core.store import vault_db_path
from knox.mcp.tools.sources import sources_export as mcp_sources_export  # noqa: E402

_cli_runner = CliRunner()

# ── element rendering ────────────────────────────────────────────────────


class TestElementRendering:
    def test_all_element_types(self, tmp_path: Path) -> None:
        data = _cache_data(
            _element("heading", "Intro", heading_level=2),
            _element("text", "Some paragraph."),
            _element("list_item", "first"),
            _element("list_item", "second"),
            _element("code", "print(1)", metadata={"language": "python"}),
            _element("formula", r"E = mc^2"),
            _element("table", [["a", "b"], ["1", "2"]]),
            _element("metadata", '{"key": "value"}'),
        )
        md = _render(data, tmp_path)
        assert "## Intro" in md
        assert "Some paragraph." in md
        assert "- first" in md
        assert "- second" in md
        assert "```python\nprint(1)\n```" in md
        assert "$$\nE = mc^2\n$$" in md
        assert "| a | b |" in md
        assert "| --- | --- |" in md
        assert "| 1 | 2 |" in md
        assert '{"key": "value"}' not in md  # metadata elements are skipped

    def test_title_and_summary_placement(self, tmp_path: Path) -> None:
        data = _cache_data(_element("heading", "Intro", heading_level=2))
        md = _render(data, tmp_path, summary="inline")
        assert md.index("# Doc Title") < md.index("## Summary") < md.index("## Intro")

    def test_summary_from_headings(self, tmp_path: Path) -> None:
        data = _cache_data(_element("heading", "A", heading_level=1), _element("heading", "B", heading_level=1))
        assert document_summary(data) == "A; B"

    def test_summary_falls_back_to_stem(self, tmp_path: Path) -> None:
        assert document_summary(_cache_data(title="")) == "doc"

    def test_table_escapes_pipes(self, tmp_path: Path) -> None:
        data = _cache_data(_element("table", [["a|b"], ["1"]]))
        md = _render(data, tmp_path)
        assert "| a\\|b |" in md

    def test_image_placeholder_rendered_as_text(self, tmp_path: Path) -> None:
        data = _cache_data(_element("image", "[Image: figure]"))
        md = _render(data, tmp_path)
        assert "*[Image: figure]*" in md


# ── images ───────────────────────────────────────────────────────────────


class TestImages:
    def test_no_inline_copies_image_relative_link(self, tmp_path: Path) -> None:
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        data = _cache_data(_element("image", str(img)))
        _write_cache(tmp_path, data)
        out = tmp_path / "out.md"
        export_source(
            base_dir=tmp_path,
            cache_dir=tmp_path / ".knox" / "cache",
            source_id="doc",
            out_path=out,
            inline=False,
            annotations="off",
            summary="off",
        )
        md = out.read_text(encoding="utf-8")
        assert "out_assets/pic.png" in md
        assert (tmp_path / "out_assets" / "pic.png").exists()
        assert str(img) not in md  # no absolute path in output

    def test_relative_path_resolves_against_base_dir(self, tmp_path: Path) -> None:
        img_dir = tmp_path / ".knox" / "cache" / "images"
        img_dir.mkdir(parents=True)
        img = img_dir / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        data = _cache_data(_element("image", ".knox/cache/images/pic.png"))
        md = _render(data, tmp_path)
        assert "out_assets/pic.png" in md

    def test_inline_embeds_data_uri(self, tmp_path: Path) -> None:
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        data = _cache_data(_element("image", str(img)))
        md = render_document_markdown(
            data,
            base_dir=tmp_path,
            assets_dir=tmp_path / "out_assets",
            inline=True,
            annotations="off",
            summary="off",
        )
        assert "data:image/png;base64," in md
        assert not (tmp_path / "out_assets").exists()

    def test_data_uri_decoded_in_no_inline_mode(self, tmp_path: Path) -> None:
        raw = b"\x89PNG\r\n\x1a\n"
        uri = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
        data = _cache_data(_element("image", uri))
        md = _render(data, tmp_path)
        assert "out_assets/image1.png" in md
        assert (tmp_path / "out_assets" / "image1.png").read_bytes() == raw


# ── cache entry resolution ───────────────────────────────────────────────


class TestLoadCacheEntry:
    def test_unknown_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_cache_entry("nope", tmp_path / ".knox" / "cache")

    def test_same_stem_exact_source_wins(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".knox" / "cache"
        cache_dir.mkdir(parents=True)
        db_path = vault_db_path(cache_dir)
        write_document(db_path, "doc-11111111", _cache_data(title="one", source_file="sources/other.pdf"))
        write_document(db_path, "doc-22222222", _cache_data(title="two", source_file="sources/doc.pdf"))
        data = load_cache_entry("sources/doc.pdf", cache_dir)
        assert data["title"] == "two"
        data = load_cache_entry("doc", cache_dir)
        assert data["title"] == "one"  # bare stem: first sorted match


# ── attachments ──────────────────────────────────────────────────────────


class TestAttachments:
    def test_attachments_table(self, tmp_path: Path) -> None:
        knox_dir = tmp_path / ".knox"
        att_dir = knox_dir / "attachments" / "doc"
        att_dir.mkdir(parents=True)
        att_dir.joinpath("notes.txt").write_text("hi", encoding="utf-8")
        manifest = AttachmentManifest(knox_dir / "attachments.jsonl")
        manifest.append(
            Attachment(
                path="doc/notes.txt",
                source="doc",
                ingestible=True,
                registered_at="2026-08-16T00:00:00+00:00",
            )
        )
        data = _cache_data(_element("text", "body"))
        md = _render(data, tmp_path)
        assert "## Attachments" in md
        assert "notes.txt" in md
        assert "[notes.txt](doc/notes.txt)" in md

    def test_no_attachments_no_section(self, tmp_path: Path) -> None:
        md = _render(_cache_data(), tmp_path)
        assert "## Attachments" not in md


class TestAnnotations:
    def test_inline_section_with_matching_record(self, tmp_path: Path) -> None:
        # Asset ids are content-addressed: seed the store with the id export
        # computes for the element, then assert the section appears.
        el = _element("table", [["h"], ["v"]])
        el_dict = _element_from_dict(el)
        record_dict = _structured_dict(el_dict)
        assert record_dict is not None
        aid = _asset_id(AssetKind.TABLE, record_dict)

        _seed_asset_store(
            tmp_path,
            {
                "id": aid,
                "kind": "table",
                "source": "sources/doc.pdf",
                "data": record_dict,
                "summary": "1 cols x 1 rows",
                "ref_count": 1,
            },
        )
        data = _cache_data(el)
        md = _render(data, tmp_path, annotations="inline")
        assert "## Annotated Items" in md
        assert aid in md
        assert "1 cols x 1 rows" in md

    def test_no_match_no_section(self, tmp_path: Path) -> None:
        data = _cache_data(_element("table", [["h"], ["v"]]))
        md = _render(data, tmp_path, annotations="inline")
        assert "## Annotated Items" not in md

    def test_equations_have_no_annotations(self, tmp_path: Path) -> None:
        data = _cache_data(_element("formula", r"E = mc^2"))
        md = _render(data, tmp_path, annotations="inline")
        assert "## Annotated Items" not in md

    def test_asset_mode_writes_extracted_json(self, tmp_path: Path) -> None:
        data = _cache_data(
            _element("heading", "Intro", heading_level=1),
            _element("table", [["h"], ["v"]]),
        )
        _write_cache(tmp_path, data)
        out = tmp_path / "out.md"
        export_source(
            base_dir=tmp_path,
            cache_dir=tmp_path / ".knox" / "cache",
            source_id="doc",
            out_path=out,
            inline=False,
            annotations="asset",
            summary="asset",
        )
        payload = json.loads((tmp_path / "out_assets" / "extracted.json").read_text(encoding="utf-8"))
        assert payload["source"] == "sources/doc.pdf"
        assert payload["summary"] == "Intro"
        assert payload["annotations"] == []
        md = out.read_text(encoding="utf-8")
        assert "## Summary" not in md
        assert "## Annotated Items" not in md

    def test_off_mode_no_sections_no_json(self, tmp_path: Path) -> None:
        data = _cache_data(_element("text", "body"))
        _write_cache(tmp_path, data)
        out = tmp_path / "out.md"
        export_source(
            base_dir=tmp_path,
            cache_dir=tmp_path / ".knox" / "cache",
            source_id="doc",
            out_path=out,
            inline=False,
            annotations="off",
            summary="off",
        )
        md = out.read_text(encoding="utf-8")
        assert "## Summary" not in md
        assert "## Annotated Items" not in md
        assert not (tmp_path / "out_assets").exists()


class TestCliExportAll:
    def test_exports_every_cache_entry_and_splices_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        base = _seed_vault_with_cache(
            tmp_path,
            _cache_data(title="One", source_file="sources/one.pdf"),
            _cache_data(title="Two", source_file="sources/two.pdf"),
        )
        (base / "index.md").write_text("# Wiki Index\n", encoding="utf-8")
        monkeypatch.chdir(base)

        result = _cli_runner.invoke(cli_app, ["sources", "export", "--all"])

        assert result.exit_code == 0, result.output
        assert (base / "exported" / "one.md").is_file()
        assert (base / "exported" / "two.md").is_file()
        index = (base / "index.md").read_text(encoding="utf-8")
        assert "## Exported sources" in index
        assert "- [one.md](exported/one.md)" in index
        assert "- [two.md](exported/two.md)" in index

    def test_no_cache_entries_exits_successfully(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        base = _seed_vault_with_cache(tmp_path)
        monkeypatch.chdir(base)

        result = _cli_runner.invoke(cli_app, ["sources", "export", "--all"])

        assert result.exit_code == 0, result.output
        assert "No sources found" in result.output

    def test_all_rejects_output_and_source_id(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        base = _seed_vault_with_cache(tmp_path, _cache_data(source_file="sources/one.pdf"))
        monkeypatch.chdir(base)

        with_output = _cli_runner.invoke(cli_app, ["sources", "export", "--all", "--output", "x.md"])
        assert with_output.exit_code == 1
        assert "OUTPUT must be empty" in with_output.output

        with_source = _cli_runner.invoke(cli_app, ["sources", "export", "--all", "sources/one.pdf"])
        assert with_source.exit_code == 1
        assert "SOURCE_ID cannot be combined" in with_source.output


class TestMcpExportAll:
    def test_export_all_parity(self, tmp_path: Path) -> None:
        base = _seed_vault_with_cache(
            tmp_path,
            _cache_data(title="One", source_file="sources/one.pdf"),
            _cache_data(title="Two", source_file="sources/two.pdf"),
        )
        (base / "index.md").write_text("# Wiki Index\n", encoding="utf-8")

        result = mcp_sources_export(wiki_dir=str(base), export_all=True)

        assert result["status"] == "ok"
        assert result["count"] == 2
        assert (base / "exported" / "one.md").is_file()
        assert (base / "exported" / "two.md").is_file()
        index = (base / "index.md").read_text(encoding="utf-8")
        assert "## Exported sources" in index

    def test_export_all_no_cache(self, tmp_path: Path) -> None:
        base = _seed_vault_with_cache(tmp_path)

        result = mcp_sources_export(wiki_dir=str(base), export_all=True)

        assert result["status"] == "ok"
        assert result["count"] == 0


def _element(
    etype: str,
    content: object,
    *,
    heading_level: int | None = None,
    page: int | None = None,
    metadata: dict | None = None,
) -> dict:
    return {
        "type": etype,
        "content": content,
        "page": page,
        "reading_order": None,
        "heading_level": heading_level,
        "metadata": metadata or {},
        "source_file": None,
    }


def _cache_data(*elements: dict, title: str = "Doc Title", source_file: str = "sources/doc.pdf") -> dict:
    return {
        "source_file": source_file,
        "title": title,
        "author": "",
        "language": "en",
        "page_count": 1,
        "metadata": {},
        "elements": list(elements),
    }


def _write_cache(tmp_path: Path, data: dict, stem: str = "doc") -> None:
    cache_dir = tmp_path / ".knox" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_document(vault_db_path(cache_dir), f"{stem}-a1b2c3d4", data)


def _render(data: dict, tmp_path: Path, **kwargs: object) -> str:
    opts: dict[str, object] = {"inline": False, "annotations": "off", "summary": "off"}
    opts.update(kwargs)
    return render_document_markdown(
        data,
        base_dir=tmp_path,
        assets_dir=tmp_path / "out_assets",
        **opts,
    )


# ── annotations + summary modes ──────────────────────────────────────────


def _seed_asset_store(tmp_path: Path, record: dict) -> None:
    store = AssetStore(tmp_path / ".knox" / "assets.json")
    store._data[record["id"]] = record  # noqa: SLF001
    store.save()


def _seed_vault_with_cache(tmp_path: Path, *cache: dict) -> Path:
    """Minimal vault: .knox/config.toml, sources/, cache entries per source."""
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".knox").mkdir(parents=True)
    (base / ".knox" / "config.toml").write_text("", encoding="utf-8")
    for i, data in enumerate(cache):
        stem = Path(data["source_file"]).stem
        cache_dir = base / ".knox" / "cache"
        cache_dir.mkdir(exist_ok=True)
        write_document(vault_db_path(cache_dir), f"{stem}-{i:08x}", data)
    return base
