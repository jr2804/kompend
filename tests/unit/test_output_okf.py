"""Tests for the OKF (Open Knowledge Format) v0.2 output writer."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from knox.core.models import CompiledWiki, MetadataField, WikiNode
from knox.core.models.assets import AssetRecord
from knox.core.models.enums import AssetKind
from knox.core.models.structured import StructuredFigure, StructuredTable
from knox.core.pipeline.assets import AssetStore
from knox.output.base import BaseOutputWriter
from knox.output.okf_writer import OkfWriter, convert_wikilinks

# ── Fixtures ────────────────────────────────────────────────────────────


_ISO_DATE_HEADING_RE = re.compile(r"## \d{4}-\d{2}-\d{2}")


# ── Asset footnotes ──────────────────────────────────────────────────────


class TestAssetFootnotes:
    """Asset references (``¶ <id>``) in body text become ``[^<id>]`` footnotes."""

    @pytest.fixture
    def asset_store(self, tmp_path: Path) -> Path:
        p = tmp_path / "assets.json"
        store = AssetStore(p)
        store.put(
            AssetRecord(
                id="tab-a1b2c3",
                kind=AssetKind.TABLE,
                source="src.pdf",
                data=StructuredTable(columns=["h1", "h2"], rows=[{"h1": "v1", "h2": "v2"}]),
                summary="A sample table",
            )
        )
        store.put(
            AssetRecord(
                id="fig-9c2e",
                kind=AssetKind.FIGURE,
                source="src.pdf",
                data=StructuredFigure(caption="Figure 1"),
                summary="Architecture diagram",
            )
        )
        store.save()
        return p

    def test_footnote_appears_for_asset_reference(self, tmp_path: Path, asset_store: Path) -> None:
        """A ``¶ tab-a1b2c3`` in body text produces ``[^tab-a1b2c3]: ...``."""
        node = _make_node(
            node_id="asset-page",
            title="Asset Test",
            content="This references ¶ tab-a1b2c3 in the text.",
        )
        wiki = _make_wiki([node])
        writer = OkfWriter(assets_file=asset_store)
        writer.write(wiki, tmp_path)

        text = (tmp_path / "concepts" / "asset-page.md").read_text(encoding="utf-8")
        assert "[^tab-a1b2c3]: A sample table" in text

    def test_multiple_asset_footnotes(self, tmp_path: Path, asset_store: Path) -> None:
        """Multiple ``¶ <id>`` markers each get their own footnote."""
        node = _make_node(
            node_id="multi-asset",
            title="Multi Asset",
            content="See ¶ tab-a1b2c3 and ¶ fig-9c2e.",
        )
        wiki = _make_wiki([node])
        writer = OkfWriter(assets_file=asset_store)
        writer.write(wiki, tmp_path)

        text = (tmp_path / "concepts" / "multi-asset.md").read_text(encoding="utf-8")
        assert "[^tab-a1b2c3]: A sample table" in text
        assert "[^fig-9c2e]: Architecture diagram" in text

    def test_no_footnotes_without_asset_refs(self, tmp_path: Path) -> None:
        """Body without ``¶ <id>`` markers does not produce asset footnotes."""
        node = _make_node(node_id="plain", title="Plain", content="No asset references here.")
        wiki = _make_wiki([node])
        writer = OkfWriter()
        writer.write(wiki, tmp_path)

        text = (tmp_path / "concepts" / "plain.md").read_text(encoding="utf-8")
        # No ``[^...]`` lines in the output.
        assert "[^" not in text or "[^plain]" not in text

    def test_no_assets_file_passed_safely(self, tmp_path: Path) -> None:
        """Writer without ``assets_file`` does not crash (graceful omission)."""
        node = _make_node(
            node_id="safe",
            title="Safe",
            content="References ¶ tab-nonexistent here.",
        )
        wiki = _make_wiki([node])
        writer = OkfWriter()  # no assets_file
        writer.write(wiki, tmp_path)
        text = (tmp_path / "concepts" / "safe.md").read_text(encoding="utf-8")
        # No footnotes — store not loaded.
        assert "[^tab-nonexistent]" not in text


# ── Registration ────────────────────────────────────────────────────────


def test_okf_writer_registered() -> None:
    """OkfWriter must be discoverable via BaseOutputWriter.get('okf')."""
    cls = BaseOutputWriter.get("okf")
    assert cls is OkfWriter
    assert "okf" in BaseOutputWriter.available_formats()


# ── Frontmatter conformance (§11 item 2) ────────────────────────────────


def test_every_page_has_type_frontmatter(tmp_path: Path) -> None:
    """OKF §11.2: every frontmatter must contain a non-empty ``type`` field."""
    wiki = _make_wiki(
        [
            _make_node("alpha", "Alpha", "Content A"),
            _make_node("beta", "Beta", "Content B", node_type="entity"),
        ]
    )
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    for md_file in tmp_path.rglob("*.md"):
        if md_file.name in ("index.md", "log.md"):
            continue
        text = md_file.read_text(encoding="utf-8")
        fm = _parse_frontmatter(text)
        assert fm is not None, f"{md_file.name} missing frontmatter"
        assert "type" in fm, f"{md_file.name} missing 'type' field"
        assert fm["type"], f"{md_file.name} has empty 'type' field"


def test_type_maps_from_node_type(tmp_path: Path) -> None:
    """KNOX node_type values map to OKF type strings."""
    wiki = _make_wiki(
        [
            _make_node("c", "Concept", "x", node_type="concept"),
            _make_node("e", "Entity", "x", node_type="entity"),
            _make_node("o", "Overview", "x", node_type="overview"),
        ]
    )
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    concept_fm = _parse_frontmatter((tmp_path / "concepts" / "c.md").read_text())
    entity_fm = _parse_frontmatter((tmp_path / "concepts" / "e.md").read_text())
    overview_fm = _parse_frontmatter((tmp_path / "o.md").read_text())

    assert concept_fm is not None
    assert entity_fm is not None
    assert overview_fm is not None
    assert concept_fm["type"] == "Concept"
    assert entity_fm["type"] == "Entity"
    assert overview_fm["type"] == "Overview"


def test_okf_recommended_fields_present(tmp_path: Path) -> None:
    """OKF v0.2 recommended fields: title, description, status, generated."""
    wiki = _make_wiki([_make_node("test", "Test Page", "A test description.", provenance=["src.pdf"])])
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    text = (tmp_path / "concepts" / "test.md").read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    assert fm is not None
    assert fm["title"] == "Test Page"
    assert "description" in fm
    # v0.2 lifecycle field (§5.4).
    assert fm["status"] == "stable"
    # v0.2 trust field supersedes the v0.1 ``timestamp`` (§13.1).
    assert "timestamp" not in fm, "v0.1 timestamp must not be emitted under v0.2"
    assert "generated" in fm
    assert "by" in fm["generated"]
    assert "at" in fm["generated"]
    assert fm["generated"]["by"].startswith("knox/")


def test_tags_not_overloaded_with_source_files(tmp_path: Path) -> None:
    """OKF v0.2 §4.1: ``tags`` are topic tags, not provenance filename stems.

    Source-filename provenance lives in the KNOX ``source_files`` extension,
    freeing ``tags`` (and avoiding the ``sources`` shape collision of §5.1).
    """
    wiki = _make_wiki([_make_node("test", "Test", "Content.", provenance=["paper.pdf", "report.pdf"])])
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    fm = _parse_frontmatter((tmp_path / "concepts" / "test.md").read_text())
    assert fm is not None
    # Provenance stems appear in source_files, not tags.
    assert fm.get("source_files") == ["paper", "report"]
    tags = fm.get("tags") or []
    assert "paper" not in tags
    assert "report" not in tags


def test_sources_are_structured_objects(tmp_path: Path) -> None:
    """OKF v0.2 §5.1: ``sources`` is a list of objects, never bare strings."""
    wiki = _make_wiki([_make_node("test", "Test", "Content.", provenance=["paper.pdf", "https://example.com/article"])])
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    fm = _parse_frontmatter((tmp_path / "concepts" / "test.md").read_text())
    assert fm is not None
    sources = fm.get("sources")
    assert sources is not None, "structured sources must be emitted when provenance exists"
    assert isinstance(sources, list)
    for entry in sources:
        assert isinstance(entry, dict), "each sources entry must be an object (§5.1)"
        # Each entry has an id (footnote key) and a resource.
        assert "id" in entry
        assert "resource" in entry


def test_generated_actor_convention(tmp_path: Path) -> None:
    """OKF v0.2 §7: ``generated.by`` follows the ``<producer>/<version>`` convention."""
    wiki = _make_wiki([_make_node("test", "Test", "Content.")])
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    fm = _parse_frontmatter((tmp_path / "concepts" / "test.md").read_text())
    assert fm is not None
    by = fm["generated"]["by"]
    # ``knox/<version>`` per §7 actor convention.
    assert re.match(r"^knox/.+$", by), f"generated.by '{by}' not a §7 producer/version actor"


# ── Cross-link conformance (§6) ─────────────────────────────────────────


def test_wikilinks_converted_to_absolute_markdown_links(tmp_path: Path) -> None:
    """OKF §6: cross-links use standard markdown, not [[wikilinks]]."""
    wiki = _make_wiki(
        [
            _make_node("alpha", "Alpha", "See [[beta]] for more.", node_type="concept"),
            _make_node("beta", "Beta", "See [[alpha]] for context.", node_type="concept"),
        ]
    )
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    alpha = (tmp_path / "concepts" / "alpha.md").read_text(encoding="utf-8")
    beta = (tmp_path / "concepts" / "beta.md").read_text(encoding="utf-8")

    # No raw [[wikilinks]] in concept body (excluding ## Related section)
    assert "[[beta]]" not in alpha
    assert "[[alpha]]" not in beta

    # Absolute bundle-relative links present
    assert "[beta](/concepts/beta.md)" in alpha
    assert "[alpha](/concepts/alpha.md)" in beta


def test_related_section_uses_markdown_links(tmp_path: Path) -> None:
    """Related section links must be standard markdown, not wikilinks."""
    wiki = _make_wiki(
        [
            _make_node("a", "Alpha", "x", node_type="concept"),
            _make_node("b", "Beta", "x", node_type="concept"),
        ],
        edges=[{"source": "a", "target": "b", "type": "related", "confidence": 0.9}],
    )
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    alpha = (tmp_path / "concepts" / "a.md").read_text(encoding="utf-8")
    assert "[[Beta]]" not in alpha
    assert "[Beta](/concepts/b.md)" in alpha


def test_convert_wikilinks_unknown_target_left_as_text() -> None:
    """OKF §6.1: broken links are valid — unknown targets become plain text."""
    result = convert_wikilinks("See [[nonexistent]] here", {})
    assert "[[nonexistent]]" not in result
    assert "nonexistent" in result


def test_convert_wikilinks_with_alias() -> None:
    """Aliased wikilinks [[target|alias]] render with alias text."""
    slug_map = {"target": "/concepts/target.md"}
    result = convert_wikilinks("See [[target|Custom Title]]", slug_map)
    assert "[Custom Title](/concepts/target.md)" in result


# ── Index conformance (§8) ──────────────────────────────────────────────


def test_index_has_okf_version_frontmatter(tmp_path: Path) -> None:
    """OKF §12: root index.md MAY declare okf_version in frontmatter."""
    writer = OkfWriter()
    writer.write(_make_wiki([_make_node()]), tmp_path)

    index_text = (tmp_path / "index.md").read_text(encoding="utf-8")
    fm = _parse_frontmatter(index_text)
    assert fm is not None
    assert fm.get("okf_version") == "0.2"


def test_index_has_grouped_sections(tmp_path: Path) -> None:
    """OKF §8: index uses grouped sections with descriptions."""
    wiki = _make_wiki(
        [
            _make_node("c1", "Concept 1", "First concept.", node_type="concept"),
            _make_node("c2", "Concept 2", "Second concept.", node_type="concept"),
            _make_node("e1", "Entity 1", "An entity.", node_type="entity"),
        ]
    )
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    index_text = (tmp_path / "index.md").read_text(encoding="utf-8")
    # Concept section heading
    assert "# Concept" in index_text
    # Entity section heading
    assert "# Entity" in index_text or "# Concepts" in index_text
    # Entries with descriptions
    assert "Concept 1" in index_text
    assert "First concept." in index_text


# ── Log conformance (§9) ────────────────────────────────────────────────


def test_log_uses_iso_date_headings(tmp_path: Path) -> None:
    """OKF §9: log.md uses ISO 8601 YYYY-MM-DD date headings."""
    writer = OkfWriter()
    writer.write(_make_wiki([_make_node()]), tmp_path)

    log_text = (tmp_path / "log.md").read_text(encoding="utf-8")
    # Must contain a ## YYYY-MM-DD heading
    match = _ISO_DATE_HEADING_RE.search(log_text)
    assert match is not None, "log.md missing ISO date heading"


def test_log_has_bold_action_prefix(tmp_path: Path) -> None:
    """OKF §9 convention: log entries use bold action prefix."""
    writer = OkfWriter()
    writer.write(_make_wiki([_make_node()]), tmp_path)

    log_text = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "**Creation**" in log_text or "**Update**" in log_text


# ── Per-claim attribution (§5.1) ────────────────────────────────────────


def test_source_footnotes_emitted_for_provenance(tmp_path: Path) -> None:
    """OKF §5.1: per-claim attribution via footnote definitions keyed to sources[].id.

    Replaces the v0.1 body ``# Citations`` numbered list (§13.1).
    """
    wiki = _make_wiki([_make_node("test", "Test", "Content.", provenance=["paper.pdf", "https://example.com/article"])])
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    text = (tmp_path / "concepts" / "test.md").read_text(encoding="utf-8")
    # v0.1 numbered Citations section must NOT appear.
    assert "# Citations" not in text
    # Footnote definitions (``[^id]: label``) for each source id.
    fm = _parse_frontmatter(text)
    assert fm is not None
    for src in fm["sources"]:
        sid = src["id"]
        assert f"[^{sid}]:" in text, f"missing footnote definition for source id '{sid}'"


def test_no_attribution_section_without_provenance(tmp_path: Path) -> None:
    """Pages without provenance emit no sources and no footnote definitions."""
    wiki = _make_wiki([_make_node("test", "Test", "Content.")])
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    text = (tmp_path / "concepts" / "test.md").read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    assert fm is not None
    assert "sources" not in fm, "sources must be omitted (not empty) when no provenance"
    assert "[^" not in text


# ── YAML alias prevention ───────────────────────────────────────────────


def test_no_yaml_anchors_in_frontmatter(tmp_path: Path) -> None:
    """YAML must not emit anchors/aliases for repeated list values."""
    wiki = _make_wiki([_make_node("test", "Test", "Content.", provenance=["a.pdf", "b.pdf"])])
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    text = (tmp_path / "concepts" / "test.md").read_text(encoding="utf-8")
    assert "&id" not in text
    assert "*id" not in text


def _make_wiki(nodes: list[WikiNode] | None = None, edges: list[dict] | None = None) -> CompiledWiki:
    if nodes is None:
        nodes = [_make_node()]
    return CompiledWiki(nodes=nodes, edges=edges or [], toc=[])


def _make_node(
    node_id: str = "test",
    title: str = "Test Page",
    content: str = "This is test content.",
    node_type: str = "concept",
    provenance: list[str] | None = None,
) -> WikiNode:
    metadata = {}
    if provenance:
        metadata[MetadataField.PROVENANCE] = provenance
    return WikiNode(
        id=node_id,
        title=title,
        content=content,
        node_type=node_type,
        source_chunks=[],
        metadata=metadata,
    )


# ── Helpers ─────────────────────────────────────────────────────────────


def _parse_frontmatter(text: str) -> dict | None:
    """Extract and parse the YAML frontmatter block from a markdown file."""
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    return yaml.safe_load(parts[1])
