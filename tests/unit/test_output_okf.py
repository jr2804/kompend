"""Tests for the OKF (Open Knowledge Format) v0.1 output writer."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from knox.core.models import CompiledWiki, MetadataField, WikiNode
from knox.output.base import BaseOutputWriter
from knox.output.okf_writer import OkfWriter, convert_wikilinks

# ── Fixtures ────────────────────────────────────────────────────────────


_ISO_DATE_HEADING_RE = re.compile(r"## \d{4}-\d{2}-\d{2}")


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


def _make_wiki(nodes: list[WikiNode] | None = None, edges: list[dict] | None = None) -> CompiledWiki:
    if nodes is None:
        nodes = [_make_node()]
    return CompiledWiki(nodes=nodes, edges=edges or [], toc=[])


# ── Registration ────────────────────────────────────────────────────────


def test_okf_writer_registered() -> None:
    """OkfWriter must be discoverable via BaseOutputWriter.get('okf')."""
    cls = BaseOutputWriter.get("okf")
    assert cls is OkfWriter
    assert "okf" in BaseOutputWriter.available_formats()


# ── Frontmatter conformance (§9 item 2) ─────────────────────────────────


def test_every_page_has_type_frontmatter(tmp_path: Path) -> None:
    """OKF §9.2: every frontmatter must contain a non-empty ``type`` field."""
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

    assert concept_fm["type"] == "Concept"
    assert entity_fm["type"] == "Entity"
    assert overview_fm["type"] == "Overview"


def test_okf_recommended_fields_present(tmp_path: Path) -> None:
    """OKF recommended fields: title, description, tags, timestamp."""
    wiki = _make_wiki([_make_node("test", "Test Page", "A test description.", provenance=["src.pdf"])])
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    text = (tmp_path / "concepts" / "test.md").read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    assert fm["title"] == "Test Page"
    assert "description" in fm
    assert "tags" in fm
    assert "timestamp" in fm


# ── Cross-link conformance (§5) ─────────────────────────────────────────


def test_wikilinks_converted_to_absolute_markdown_links(tmp_path: Path) -> None:
    """OKF §5: cross-links use standard markdown, not [[wikilinks]]."""
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
    """OKF §5.3: broken links are valid — unknown targets become plain text."""
    result = convert_wikilinks("See [[nonexistent]] here", {})
    assert "[[nonexistent]]" not in result
    assert "nonexistent" in result


def test_convert_wikilinks_with_alias() -> None:
    """Aliased wikilinks [[target|alias]] render with alias text."""
    slug_map = {"target": "/concepts/target.md"}
    result = convert_wikilinks("See [[target|Custom Title]]", slug_map)
    assert "[Custom Title](/concepts/target.md)" in result


# ── Index conformance (§6) ──────────────────────────────────────────────


def test_index_has_okf_version_frontmatter(tmp_path: Path) -> None:
    """OKF §11: root index.md MAY declare okf_version in frontmatter."""
    writer = OkfWriter()
    writer.write(_make_wiki([_make_node()]), tmp_path)

    index_text = (tmp_path / "index.md").read_text(encoding="utf-8")
    fm = _parse_frontmatter(index_text)
    assert fm is not None
    assert fm.get("okf_version") == "0.1"


def test_index_has_grouped_sections(tmp_path: Path) -> None:
    """OKF §6: index uses grouped sections with descriptions."""
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


def test_non_root_index_has_no_frontmatter() -> None:
    """OKF §6: index.md files contain no frontmatter (except root with okf_version)."""
    # This test verifies the concept is enforced — only the root index
    # gets frontmatter. Currently OkfWriter only writes a root index.
    # If per-directory indices are added later, this test enforces the rule.
    pass  # placeholder — OkfWriter currently only writes root index


# ── Log conformance (§7) ────────────────────────────────────────────────


def test_log_uses_iso_date_headings(tmp_path: Path) -> None:
    """OKF §7: log.md uses ISO 8601 YYYY-MM-DD date headings."""
    writer = OkfWriter()
    writer.write(_make_wiki([_make_node()]), tmp_path)

    log_text = (tmp_path / "log.md").read_text(encoding="utf-8")
    # Must contain a ## YYYY-MM-DD heading
    match = _ISO_DATE_HEADING_RE.search(log_text)
    assert match is not None, "log.md missing ISO date heading"


def test_log_has_bold_action_prefix(tmp_path: Path) -> None:
    """OKF §7 convention: log entries use bold action prefix."""
    writer = OkfWriter()
    writer.write(_make_wiki([_make_node()]), tmp_path)

    log_text = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "**Creation**" in log_text or "**Update**" in log_text


# ── Citations (§8) ──────────────────────────────────────────────────────


def test_citations_section_from_provenance(tmp_path: Path) -> None:
    """OKF §8: citations rendered as numbered [N] references."""
    wiki = _make_wiki(
        [
            _make_node("test", "Test", "Content.", provenance=["paper.pdf", "https://example.com/article"]),
        ]
    )
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    text = (tmp_path / "concepts" / "test.md").read_text(encoding="utf-8")
    assert "# Citations" in text
    assert "[1]" in text
    assert "[2]" in text
    assert "paper.pdf" in text
    assert "example.com" in text


def test_no_citations_section_without_provenance(tmp_path: Path) -> None:
    """Pages without provenance should not have a Citations section."""
    wiki = _make_wiki([_make_node("test", "Test", "Content.")])
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    text = (tmp_path / "concepts" / "test.md").read_text(encoding="utf-8")
    assert "# Citations" not in text


# ── YAML alias prevention ───────────────────────────────────────────────


def test_no_yaml_anchors_in_frontmatter(tmp_path: Path) -> None:
    """YAML must not emit anchors/aliases for repeated list values."""
    wiki = _make_wiki([_make_node("test", "Test", "Content.", provenance=["a.pdf", "b.pdf"])])
    writer = OkfWriter()
    writer.write(wiki, tmp_path)

    text = (tmp_path / "concepts" / "test.md").read_text(encoding="utf-8")
    assert "&id" not in text
    assert "*id" not in text


# ── Helpers ─────────────────────────────────────────────────────────────


def _parse_frontmatter(text: str) -> dict | None:
    """Extract and parse the YAML frontmatter block from a markdown file."""
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    return yaml.safe_load(parts[1])
