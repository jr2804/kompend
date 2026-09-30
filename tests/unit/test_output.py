"""Tests for KNOX output writers (Milestone 5)."""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knox.core.models import CompiledWiki, GraphEdge, TocEntry, WikiNode
from knox.core.models.assets import AssetRecord
from knox.core.models.enums import AssetKind
from knox.core.models.references import Reference
from knox.core.models.structured import StructuredFigure, StructuredTable
from knox.core.paths import attachment_slug
from knox.core.pipeline.assets import AssetStore
from knox.core.pipeline.references import ReferenceStore
from knox.core.store import open_store
from knox.core.store.schema import VERIFICATION_TABLE
from knox.output.json_writer import JsonWriter
from knox.output.markdown import MarkdownWriter
from knox.output.typst import TypstWriter
from knox.output.vault import VaultWriter

# ---------------------------------------------------------------------------
# MarkdownWriter Tests
# ---------------------------------------------------------------------------


class TestMarkdownWriter:
    def test_writes_valid_markdown(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = MarkdownWriter(vault_mode=False)
        result_file = writer.write(sample_wiki, temp_output)

        assert result_file == temp_output / "wiki.md"
        assert result_file.exists()

        content = result_file.read_text(encoding="utf-8")
        assert "# Test Wiki" in content
        assert "# Introduction" in content  # heading_level=1 -> prefix "#"
        assert "## Core Concepts" in content  # heading_level=2 -> prefix "##"
        assert "# How to Use" in content  # default heading_level=1 -> prefix "#"
        assert "This is the introduction section." in content
        assert "Understanding the fundamental ideas." in content
        assert "Step-by-step instructions." in content

    def test_handles_empty_wiki(self, empty_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = MarkdownWriter(vault_mode=False)
        result_file = writer.write(empty_wiki, temp_output)

        assert result_file.exists()
        content = result_file.read_text(encoding="utf-8")
        assert "# Compiled Wiki" in content
        assert "Table of Contents" not in content

    def test_creates_output_dir(self, sample_wiki: CompiledWiki, tmp_path: Path) -> None:
        deep_dir = tmp_path / "a" / "b" / "c"
        writer = MarkdownWriter(vault_mode=False)
        result_file = writer.write(sample_wiki, deep_dir)

        assert result_file.exists()
        assert deep_dir.exists()

    def test_vault_mode_writes_one_file_per_node(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = MarkdownWriter(vault_mode=True)
        result_file = writer.write(sample_wiki, temp_output)

        assert result_file == temp_output / "index.md"
        assert result_file.exists()
        # overview/section/procedure stay at root; concept goes to concepts/.
        assert (temp_output / "intro.md").exists()
        assert (temp_output / "concepts" / "concept-1.md").exists()
        assert (temp_output / "procedure-1.md").exists()

    def test_vault_mode_index_contains_wikilinks(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = MarkdownWriter(vault_mode=True)
        writer.write(sample_wiki, temp_output)

        index = (temp_output / "index.md").read_text(encoding="utf-8")
        assert "# Test Wiki" in index
        assert "Introduction" in index
        assert "Core Concepts" in index
        assert "How to Use" in index

    def test_vault_mode_related_section(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = MarkdownWriter(vault_mode=True)
        writer.write(sample_wiki, temp_output)

        intro = (temp_output / "intro.md").read_text(encoding="utf-8")
        assert "## Related" in intro
        assert "[Core Concepts](concepts/concept-1.md)" in intro

    def test_vault_mode_typed_relation_annotation(self, temp_output: Path) -> None:
        wiki = CompiledWiki(
            nodes=[
                WikiNode(id="a", title="Page A", content="A.", node_type="section"),
                WikiNode(id="b", title="Page B", content="B.", node_type="section"),
            ],
            edges=[
                {"source": "a", "target": "b", "type": "uses", "confidence": 0.91},
            ],
            toc=[],
        )
        writer = MarkdownWriter(vault_mode=True)
        writer.write(wiki, temp_output)

        page_a = (temp_output / "a.md").read_text(encoding="utf-8")
        assert "## Related" in page_a
        assert "[Page B](b.md) (uses) — confidence 0.91" in page_a

    def test_vault_mode_empty_wiki(self, empty_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = MarkdownWriter(vault_mode=True)
        result_file = writer.write(empty_wiki, temp_output)

        assert result_file == temp_output / "index.md"
        index = (temp_output / "index.md").read_text(encoding="utf-8")
        assert "# Knowledge Bundle" in index

    def test_vault_mode_single_node(self, tmp_path: Path) -> None:
        wiki = CompiledWiki(
            nodes=[
                WikiNode(id="test", title="Test Page", content="Hello.", node_type="concept"),
            ],
            edges=[],
            toc=[],
        )
        writer = MarkdownWriter(vault_mode=True)
        result_file = writer.write(wiki, tmp_path)

        assert result_file == tmp_path / "index.md"
        assert (tmp_path / "concepts" / "test.md").exists()
        content = (tmp_path / "concepts" / "test.md").read_text(encoding="utf-8")
        assert "# Test Page" in content
        assert "Hello." in content
        assert "## Related" not in content


class TestStaleFileClearing:
    """knox-h7bm: a recompile must not leave stale files from a prior run."""

    def test_vault_clears_stale_root_markdown(self, sample_wiki: CompiledWiki, tmp_path: Path) -> None:
        writer = MarkdownWriter(vault_mode=True)
        # First compile writes node files + index.md.
        writer.write(sample_wiki, tmp_path)
        stale_root = tmp_path / "old-page.md"
        stale_root.write_text("# Stale\n\nfrom a previous run", encoding="utf-8")
        assert stale_root.exists()

        # Recompile: stale root .md gone, current nodes present at their routes.
        writer.write(sample_wiki, tmp_path)
        assert not stale_root.exists()
        assert (tmp_path / "intro.md").exists()
        assert (tmp_path / "concepts" / "concept-1.md").exists()

    def test_vault_clears_stale_concept_subfolder(self, sample_wiki: CompiledWiki, tmp_path: Path) -> None:
        """knox-r2kp: recompile also clears stale .md in the concepts/ subfolder."""
        writer = MarkdownWriter(vault_mode=True)
        writer.write(sample_wiki, tmp_path)
        concepts_dir = tmp_path / "concepts"
        stale_concept = concepts_dir / "old-concept.md"
        stale_concept.write_text("# Stale concept", encoding="utf-8")

        writer.write(sample_wiki, tmp_path)

        assert not stale_concept.exists()
        assert (concepts_dir / "concept-1.md").exists()

    def test_vault_preserves_user_subfolder(self, sample_wiki: CompiledWiki, tmp_path: Path) -> None:
        writer = MarkdownWriter(vault_mode=True)
        # Non-recursive clear must leave subfolders (e.g. _distilled/) intact.
        distilled_dir = tmp_path / "_distilled"
        distilled_dir.mkdir(parents=True)
        distilled_file = distilled_dir / "note.md"
        distilled_file.write_text("# User note", encoding="utf-8")

        writer.write(sample_wiki, tmp_path)

        assert distilled_file.exists()
        assert (tmp_path / "intro.md").exists()


class TestSubfolderLayout:
    """knox-r2kp: synthesized pages route into subfolders by kind; root and
    non-synthesized kinds stay at the vault root; index links still resolve.
    """

    def test_concepts_routed_to_concepts_folder(self, tmp_path: Path) -> None:
        wiki = CompiledWiki(
            nodes=[
                WikiNode(id="c1", title="Concept A", content="Body.", node_type="concept"),
                WikiNode(id="c2", title="Concept B", content="Body.", node_type="comparison"),
                WikiNode(id="c3", title="Term X", content="Body.", node_type="entity"),
            ],
            edges=[],
            toc=[],
            metadata={"title": "Test"},
        )
        MarkdownWriter(vault_mode=True).write(wiki, tmp_path)
        assert (tmp_path / "concepts" / "c1.md").exists()
        assert (tmp_path / "concepts" / "c2.md").exists()
        assert (tmp_path / "entities" / "c3.md").exists()

    def test_overview_and_unknown_kinds_stay_at_root(self, tmp_path: Path) -> None:
        wiki = CompiledWiki(
            nodes=[
                WikiNode(id="ov", title="Overview", content="Body.", node_type="overview"),
                WikiNode(id="sec", title="Section", content="Body.", node_type="section"),
            ],
            edges=[],
            toc=[],
            metadata={"title": "Test"},
        )
        MarkdownWriter(vault_mode=True).write(wiki, tmp_path)
        assert (tmp_path / "ov.md").exists()
        assert (tmp_path / "sec.md").exists()
        assert not (tmp_path / "concepts").exists()

    def test_stale_entity_pages_in_typed_subfolders_are_pruned(self, tmp_path: Path) -> None:
        """Recompiling with a changed entity type removes the previous page.

        Entity pages live in ``entities/<type>/``.  A non-recursive stale
        sweep left the old type's pages behind — producing orphans with
        broken concept links, and a duplicate whenever the type was merely
        renamed (regression: ``event/`` → ``conference/``).
        """

        def _wiki(entity_type: str) -> CompiledWiki:
            return CompiledWiki(
                nodes=[
                    WikiNode(
                        id="daga-2023",
                        title="DAGA 2023",
                        content="Body.",
                        node_type="entity",
                        metadata={"entity_type": entity_type},
                    )
                ],
                edges=[],
                toc=[],
                metadata={"title": "Test"},
            )

        writer = MarkdownWriter(vault_mode=True)
        writer.write(_wiki("event"), tmp_path)
        assert (tmp_path / "entities" / "event" / "daga-2023.md").exists()

        # Same page, entity type corrected to "conference".
        writer.write(_wiki("conference"), tmp_path)

        assert (tmp_path / "entities" / "conference" / "daga-2023.md").exists()
        assert not (tmp_path / "entities" / "event" / "daga-2023.md").exists()
        # The emptied typed folder goes too — no cruft accumulation.
        assert not (tmp_path / "entities" / "event").exists()

    def test_index_lists_all_pages_regardless_of_folder(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        MarkdownWriter(vault_mode=True).write(sample_wiki, temp_output)
        index = (temp_output / "index.md").read_text(encoding="utf-8")
        # Portable relative links to every page regardless of folder.
        assert "[Introduction](intro.md)" in index
        assert "concept-1.md" in index
        assert "procedure-1.md" in index


class TestFrontmatterEmission:
    """knox-wgt.1 + knox-rxod: Every compiled page must have YAML frontmatter."""

    def test_vault_pages_have_title_frontmatter(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        """Each .md file in vault mode starts with frontmatter containing title."""
        writer = MarkdownWriter(vault_mode=True)
        writer.write(sample_wiki, temp_output)

        for node in sample_wiki.nodes:
            slug = node.id
            folder = MarkdownWriter._folder_for_kind(node.node_type)
            page_file = (temp_output / folder / f"{slug}.md") if folder else (temp_output / f"{slug}.md")
            content = page_file.read_text(encoding="utf-8")
            assert content.startswith("---\n"), f"{slug}.md missing frontmatter"
            assert "title:" in content.split("---\n")[1]

    def test_vault_pages_have_kind_and_generated(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        """OKF v0.2: every compiled page carries kind and generated.{by,at}."""
        writer = MarkdownWriter(vault_mode=True)
        writer.write(sample_wiki, temp_output)

        for node in sample_wiki.nodes:
            folder = MarkdownWriter._folder_for_kind(node.node_type)
            page_file = (temp_output / folder / f"{node.id}.md") if folder else (temp_output / f"{node.id}.md")
            fm_block = page_file.read_text(encoding="utf-8").split("---\n")[1]
            assert "kind:" in fm_block, f"{node.id}.md missing kind"
            assert "generated:" in fm_block, f"{node.id}.md missing generated"
            assert "by:" in fm_block, f"{node.id}.md missing generated.by"
            assert "at:" in fm_block, f"{node.id}.md missing generated.at"

    def test_vault_pages_source_files_from_provenance(self, temp_output: Path) -> None:
        """OKF v0.2: source_files extension is derived from node provenance."""
        wiki = CompiledWiki(
            nodes=[
                WikiNode(
                    id="c1",
                    title="Concept",
                    content="Body.",
                    node_type="concept",
                    metadata={"provenance": ["sources/paper_a.pdf", "sources/paper_b.pdf"]},
                )
            ],
            edges=[],
            toc=[],
        )
        MarkdownWriter(vault_mode=True).write(wiki, temp_output)
        content = (temp_output / "concepts" / "c1.md").read_text(encoding="utf-8")
        assert "source_files:" in content
        assert "paper_a" in content
        assert "paper_b" in content

    def test_index_has_okf_frontmatter(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        """index.md has OKF §12 frontmatter (okf_version, no title field)."""
        writer = MarkdownWriter(vault_mode=True)
        writer.write(sample_wiki, temp_output)

        index = (temp_output / "index.md").read_text(encoding="utf-8")
        assert index.startswith("---\n")
        assert 'okf_version: "0.2"' in index

    def test_frontmatter_defaults_preserved(self, temp_output: Path) -> None:
        """frontmatter_defaults from config are included alongside title."""
        wiki = CompiledWiki(
            nodes=[WikiNode(id="n1", title="Page", content="Body.", node_type="section")],
            edges=[],
            toc=[],
        )
        writer = MarkdownWriter(vault_mode=True, frontmatter_defaults={"lang": "en"})
        writer.write(wiki, temp_output)

        content = (temp_output / "n1.md").read_text(encoding="utf-8")
        assert "title: Page" in content
        assert "lang: en" in content

    def test_node_metadata_frontmatter_respected(self, temp_output: Path) -> None:
        """node.metadata['frontmatter'] values override defaults."""
        wiki = CompiledWiki(
            nodes=[
                WikiNode(
                    id="n1",
                    title="Original",
                    content="Body.",
                    node_type="section",
                    metadata={"frontmatter": {"title": "Custom Title", "kind": "overview"}},
                ),
            ],
            edges=[],
            toc=[],
        )
        writer = MarkdownWriter(vault_mode=True)
        writer.write(wiki, temp_output)

        content = (temp_output / "n1.md").read_text(encoding="utf-8")
        assert "title: Custom Title" in content
        assert "kind: overview" in content


# ---------------------------------------------------------------------------


class TestJsonWriter:
    def test_writes_valid_json(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = JsonWriter()
        result_file = writer.write(sample_wiki, temp_output)

        assert result_file == temp_output / "wiki.json"
        assert result_file.exists()

        data = json.loads(result_file.read_text(encoding="utf-8"))
        assert len(data["nodes"]) == 3
        assert data["nodes"][0]["id"] == "intro"
        assert data["nodes"][0]["title"] == "Introduction"
        assert data["nodes"][0]["content"] == "This is the introduction section."
        assert data["nodes"][0]["node_type"] == "section"
        assert data["nodes"][0]["source_chunks"] == ["chunk-1", "chunk-2"]
        assert data["nodes"][0]["metadata"]["heading_level"] == 1

        assert len(data["edges"]) == 2
        assert data["edges"][0]["source"] == "intro"
        assert data["edges"][0]["target"] == "concept-1"

        assert len(data["toc"]) == 3
        assert data["toc"][1]["title"] == "Core Concepts"

        assert data["metadata"]["title"] == "Test Wiki"
        assert data["quality_score"] == 0.0
        assert data["iteration"] == 0

    def test_handles_empty_wiki(self, empty_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = JsonWriter()
        result_file = writer.write(empty_wiki, temp_output)

        data = json.loads(result_file.read_text(encoding="utf-8"))
        assert data["nodes"] == []
        assert data["edges"] == []
        assert data["toc"] == []
        assert data["metadata"] == {}

    def test_creates_output_dir(self, sample_wiki: CompiledWiki, tmp_path: Path) -> None:
        writer = JsonWriter()
        result_file = writer.write(sample_wiki, tmp_path / "deep" / "dir")
        assert result_file.exists()


# ---------------------------------------------------------------------------
# TypstWriter Tests
# ---------------------------------------------------------------------------


class TestTypstWriter:
    def test_writes_valid_typst_source(self, sample_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = TypstWriter()
        result_file = writer.write(sample_wiki, temp_output)

        assert result_file == temp_output / "wiki.typ"
        assert result_file.exists()

        content = result_file.read_text(encoding="utf-8")
        # Document setup
        assert '#set page("a4")' in content
        assert "#set text(size: 11pt)" in content

        # Title page
        assert "#align(center + top)" in content
        assert "Test Wiki" in content

        # TOC
        assert "= Table of Contents" in content
        assert "#outline()" in content

        # Node sections with labels
        assert "#label(<intro>)" in content
        assert "== Introduction" in content  # heading_level=1 -> prefix "=="
        assert "#label(<concept-1>)" in content
        assert "=== Core Concepts" in content  # heading_level=2 -> prefix "==="
        assert "#label(<procedure-1>)" in content

        # Content
        assert "This is the introduction section." in content
        assert "Step-by-step instructions." in content

        # Source chunk references
        assert "*Source: 2 chunk(s)*" in content

        # Back links
        assert "Back to Table of Contents" in content
        assert "#pagebreak()" in content

    def test_handles_empty_wiki(self, empty_wiki: CompiledWiki, temp_output: Path) -> None:
        writer = TypstWriter()
        result_file = writer.write(empty_wiki, temp_output)

        assert result_file.exists()
        content = result_file.read_text(encoding="utf-8")
        assert '#set page("a4")' in content
        assert "Compiled Wiki" in content  # fallback title
        assert "#pagebreak()" in content

    def test_wiki_without_metadata_title(self, temp_output: Path) -> None:
        """Fallback title 'Compiled Wiki' when no metadata title."""
        wiki = CompiledWiki(
            nodes=[WikiNode(id="n1", title="Node", content="Content", node_type="section")],
            edges=[],
            toc=[],
            metadata={},
        )
        writer = TypstWriter()
        result_file = writer.write(wiki, temp_output)
        content = result_file.read_text(encoding="utf-8")
        assert "Compiled Wiki" in content

    def test_creates_output_dir(self, sample_wiki: CompiledWiki, tmp_path: Path) -> None:
        writer = TypstWriter()
        result_file = writer.write(sample_wiki, tmp_path / "nested" / "path")
        assert result_file.exists()


# ---------------------------------------------------------------------------
# Copy-on-embed: wiki pages never reference .knox/ (WP3)
# ---------------------------------------------------------------------------
class TestPortableImages:
    def test_knox_image_copied_to_attachments(self, tmp_path: Path) -> None:
        base = tmp_path / "vault"
        img_dir = base / ".knox" / "cache" / "images"
        img_dir.mkdir(parents=True)
        img_file = img_dir / "report_img_1.png"
        img_file.write_bytes(b"png-bytes")

        wiki = CompiledWiki(
            nodes=[
                WikiNode(
                    id="concept-1",
                    title="Core Concepts",
                    content="Understanding the fundamental ideas.",
                    node_type="concept",
                    source_chunks=["chunk-3"],
                    metadata={
                        "provenance": ["sources/report.pdf"],
                        "images": [{"path": ".knox/cache/images/report_img_1.png", "caption": "A figure", "element_type": "image"}],
                    },
                ),
            ],
            edges=[],
            toc=[],
        )
        writer = MarkdownWriter(vault_mode=True, base_dir=base)
        writer.write(wiki, base / "wiki")

        page = (base / "wiki" / "concepts" / "concept-1.md").read_text(encoding="utf-8")
        copied = base / "attachments" / "report" / "report_img_1.png"
        assert copied.read_bytes() == b"png-bytes"
        assert "](../../attachments/report/report_img_1.png)" in page
        assert ".knox" not in page

    def test_external_image_and_data_uri_pass_through(self, tmp_path: Path) -> None:
        base = tmp_path / "vault"
        writer = MarkdownWriter(vault_mode=True, base_dir=base)
        wiki = CompiledWiki(
            nodes=[
                WikiNode(
                    id="concept-1",
                    title="Core Concepts",
                    content="Body.",
                    node_type="concept",
                    metadata={
                        "provenance": ["sources/report.pdf"],
                        "images": [
                            {"path": "attachments/report/keep.png", "caption": "", "element_type": "image"},
                            {"path": "data:image/png;base64,AAAA", "caption": "inline", "element_type": "image"},
                        ],
                    },
                ),
            ],
            edges=[],
            toc=[],
        )
        writer.write(wiki, base / "wiki")
        page = (base / "wiki" / "concepts" / "concept-1.md").read_text(encoding="utf-8")
        assert "](attachments/report/keep.png)" in page or "](../../attachments/report/keep.png)" in page
        assert "data:image/png;base64,AAAA" in page

    def test_no_base_dir_no_rewrite(self, tmp_path: Path) -> None:
        wiki = CompiledWiki(
            nodes=[
                WikiNode(
                    id="concept-1",
                    title="Core Concepts",
                    content="Body.",
                    node_type="concept",
                    metadata={
                        "provenance": ["sources/report.pdf"],
                        "images": [{"path": ".knox/cache/images/x.png", "caption": "", "element_type": "image"}],
                    },
                ),
            ],
            edges=[],
            toc=[],
        )
        writer = MarkdownWriter(vault_mode=True)
        docs = writer._wiki_to_documents(wiki)
        assert ".knox/cache/images/x.png" in docs[0].body


# ---------------------------------------------------------------------------
# WP2: per-source summary pages (wiki/summaries/)
# ---------------------------------------------------------------------------
class TestSourceSummaries:
    def test_summary_page_written(self, tmp_path: Path) -> None:
        base = self._vault(tmp_path)
        wiki = CompiledWiki(
            nodes=[
                WikiNode(
                    id="concept-1",
                    title="Core Concepts",
                    content="Body.",
                    node_type="concept",
                    metadata={"provenance": ["sources/report.pdf"]},
                ),
            ],
            edges=[],
            toc=[],
        )
        writer = MarkdownWriter(vault_mode=True, base_dir=base)
        writer.write(wiki, base / "wiki")

        summary = base / "wiki" / "summaries" / "report.md"
        assert summary.exists()
        text = summary.read_text(encoding="utf-8")
        assert "type: Source" in text
        assert "sources/report.pdf" in text
        assert "Core Concepts" in text  # backlink to feeding concept

    def test_summary_assets_grouped_title_pretty(self, tmp_path: Path) -> None:
        """Assets render as kind counts (+ figure captions); title is readable."""
        base = self._vault(tmp_path)
        store = AssetStore(base / ".knox" / "assets.json")
        store.put(
            AssetRecord(
                id="tab-aaa",
                kind=AssetKind.TABLE,
                source="sources/report.pdf",
                data=StructuredTable(columns=["a"], rows=[{"a": "1"}]),
                summary="1 cols x 1 rows",
            )
        )
        store.put(
            AssetRecord(
                id="fig-bbb",
                kind=AssetKind.FIGURE,
                source="sources/report.pdf",
                data=StructuredFigure(caption="SNR curve"),
            )
        )
        store.save()

        wiki = CompiledWiki(nodes=[], edges=[], toc=[])
        MarkdownWriter(vault_mode=True, base_dir=base).write(wiki, base / "wiki")

        text = (base / "wiki" / "summaries" / "report.md").read_text(encoding="utf-8")
        assert "# Source: report" in text
        assert "**Tables:** 1" in text
        assert "**Figures:** 1" in text
        assert "SNR curve" in text
        assert "`tab-aaa" not in text  # no raw asset-id dump

    def test_stale_summaries_pruned(self, tmp_path: Path) -> None:
        base = self._vault(tmp_path)
        summaries = base / "wiki" / "summaries"
        summaries.mkdir(parents=True)
        (summaries / "gone.md").write_text("stale", encoding="utf-8")

        wiki = CompiledWiki(nodes=[], edges=[], toc=[])
        writer = MarkdownWriter(vault_mode=True, base_dir=base)
        writer.write(wiki, base / "wiki")
        assert not (summaries / "gone.md").exists()

    def test_same_stem_sources_get_distinct_slugs(self, tmp_path: Path) -> None:
        """Same-stem sources (foo.doc vs foo.docx) must not overwrite each other."""
        base = tmp_path / "vault"
        (base / "sources").mkdir(parents=True)
        (base / "sources" / "foo.doc").write_bytes(b"fake")
        (base / "sources" / "foo.docx").write_bytes(b"fake")

        wiki = CompiledWiki(nodes=[], edges=[], toc=[])
        writer = MarkdownWriter(vault_mode=True, base_dir=base)
        writer.write(wiki, base / "wiki")

        summaries = base / "wiki" / "summaries"
        files = {p.name for p in summaries.glob("*.md")}
        assert "foo-doc.md" in files
        assert "foo-docx.md" in files
        assert "foo.md" not in files
        # Each page names its own source file.
        assert "sources/foo.doc" in (summaries / "foo-doc.md").read_text(encoding="utf-8")
        assert "sources/foo.docx" in (summaries / "foo-docx.md").read_text(encoding="utf-8")

    def test_no_base_dir_no_summaries(self, tmp_path: Path) -> None:
        wiki = CompiledWiki(nodes=[], edges=[], toc=[])
        writer = MarkdownWriter(vault_mode=True)
        out = tmp_path / "wiki"
        writer.write(wiki, out)
        assert not (out / "summaries").exists()

    def test_index_links_summaries_and_references(self, tmp_path: Path) -> None:
        ref_file = tmp_path / ".knox" / "references.json"
        ref_file.parent.mkdir(parents=True)
        store = ReferenceStore(ref_file)
        store.save()
        base = self._vault(tmp_path)
        wiki = CompiledWiki(
            nodes=[
                WikiNode(
                    id="concept-1",
                    title="Core Concepts",
                    content="Body.",
                    node_type="concept",
                    metadata={"provenance": ["sources/report.pdf"]},
                ),
            ],
            edges=[],
            toc=[],
        )
        writer = MarkdownWriter(
            vault_mode=True,
            base_dir=base,
            references_file=ref_file,
        )
        writer.write(wiki, base / "wiki")

        index = (base / "wiki" / "index.md").read_text(encoding="utf-8")
        assert "## Sources" in index
        assert "(summaries/report.md)" in index

    def test_summary_links_existing_export(self, tmp_path: Path) -> None:
        base = self._vault(tmp_path)
        (base / "exported").mkdir()
        (base / "exported" / "report.md").write_text("# full export", encoding="utf-8")

        wiki = CompiledWiki(nodes=[], edges=[], toc=[])
        writer = MarkdownWriter(vault_mode=True, base_dir=base)
        writer.write(wiki, base / "wiki")

        text = (base / "wiki" / "summaries" / "report.md").read_text(encoding="utf-8")
        assert "## Full export" in text
        assert "(../../exported/report.md)" in text

    def _vault(self, tmp_path: Path) -> Path:
        base = tmp_path / "vault"
        (base / "sources").mkdir(parents=True)
        (base / "sources" / "report.pdf").write_bytes(b"%PDF-1.4 fake")
        return base


# ---------------------------------------------------------------------------
# WP5: quality page (wiki/quality.md) + per-page verification frontmatter
# ---------------------------------------------------------------------------
class TestQualityPage:
    def test_quality_page_written(self, tmp_path: Path) -> None:
        base = self._vault(tmp_path)
        self._verification(base)
        wiki = CompiledWiki(
            nodes=[
                WikiNode(id="concept-1", title="Alpha", content="A.", node_type="concept"),
                WikiNode(id="concept-2", title="Beta", content="B.", node_type="concept"),
            ],
            edges=[],
            toc=[],
        )
        writer = MarkdownWriter(vault_mode=True, base_dir=base)
        writer.write(wiki, base / "wiki")

        quality = base / "wiki" / "quality.md"
        assert quality.exists()
        text = quality.read_text(encoding="utf-8")
        assert "type: Quality" in text
        assert "Claims checked: **3**" in text
        assert "| Alpha | 2 | 1 |" in text
        # Index links it.
        index = (base / "wiki" / "index.md").read_text(encoding="utf-8")
        assert "(quality.md)" in index

    def test_no_verification_no_page(self, tmp_path: Path) -> None:
        base = self._vault(tmp_path)
        wiki = CompiledWiki(nodes=[], edges=[], toc=[])
        writer = MarkdownWriter(vault_mode=True, base_dir=base)
        writer.write(wiki, base / "wiki")
        assert not (base / "wiki" / "quality.md").exists()

    def test_frontmatter_quality_keys(self, tmp_path: Path) -> None:
        base = self._vault(tmp_path)
        self._verification(base)
        wiki = CompiledWiki(
            nodes=[WikiNode(id="concept-1", title="Alpha", content="A.", node_type="concept")],
            edges=[],
            toc=[],
        )
        writer = MarkdownWriter(vault_mode=True, base_dir=base)
        docs = writer._wiki_to_documents(wiki)
        fm = docs[0].frontmatter
        assert fm["verified_claims"] == "1/2"
        assert fm["quality_score"] == 0.55

    def _vault(self, tmp_path: Path) -> Path:
        base = tmp_path / "vault"
        (base / "sources").mkdir(parents=True)
        return base

    def _verification(self, base: Path) -> None:
        claims = json.dumps(
            [
                {"node_id": "concept-1", "verdict": "supported", "confidence": 0.9},
                {"node_id": "concept-1", "verdict": "unsupported", "confidence": 0.2},
                {"node_id": "concept-2", "verdict": "supported", "confidence": 1.0},
            ]
        )
        db = base / ".knox" / "knox.db"
        with open_store(db) as conn:
            conn.execute(
                f"INSERT INTO {VERIFICATION_TABLE} (written_at, total, verified, unsupported, average_confidence, claims) VALUES (?, ?, ?, ?, ?, ?)",  # noqa: S608
                ("2025-01-01T00:00:00+00:00", 3, 2, 1, 0.7, claims),
            )


class TestReferencesPage:
    """references.md is always generated — an empty bibliography is information too."""

    def test_written_even_without_store(self, sample_wiki: CompiledWiki, tmp_path: Path) -> None:
        writer = MarkdownWriter(vault_mode=True, base_dir=tmp_path)
        writer.write(sample_wiki, tmp_path)
        page = tmp_path / "references.md"
        assert page.exists()
        content = page.read_text(encoding="utf-8")
        assert "## Corpus sources" in content
        assert "## Cited works not in corpus" in content
        assert "_(none extracted yet)_" in content

    def test_lists_corpus_sources_and_cited_works(self, sample_wiki: CompiledWiki, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()
        (sources_dir / "a.pdf").write_bytes(b"%PDF-")
        ref_file = tmp_path / ".knox" / "references.json"
        ref_file.parent.mkdir()
        store = ReferenceStore(ref_file)
        store.add_source(
            "sources/a.pdf",
            [Reference(id="smith2020", title="On Speech Quality", raw="Smith (2020) On Speech Quality.")],
        )
        store.save()
        writer = MarkdownWriter(vault_mode=True, base_dir=tmp_path, references_file=ref_file)
        writer.write(sample_wiki, tmp_path)
        content = (tmp_path / "references.md").read_text(encoding="utf-8")
        assert "- sources/a.pdf" in content
        assert "smith2020" in content

    def test_corpus_link_when_summary_page_exists(self, sample_wiki: CompiledWiki, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()
        (sources_dir / "a.pdf").write_bytes(b"%PDF-")
        summaries = tmp_path / "summaries"
        summaries.mkdir()
        slug = attachment_slug("a.pdf")
        (summaries / f"{slug}.md").write_text("# Source: A")
        writer = MarkdownWriter(vault_mode=True, base_dir=tmp_path)
        writer.write(sample_wiki, tmp_path)
        content = (tmp_path / "references.md").read_text(encoding="utf-8")
        assert f"[sources/a.pdf](summaries/{slug}.md)" in content


class TestOkfDescription:
    """okf_description truncates to the first sentence.

    Index entries are link + one-line summary, not a duplicate of the page
    body.
    """

    def test_first_sentence_only(self) -> None:
        node = WikiNode(
            id="x",
            title="X",
            content="The Bark scale is non-linear. It models frequency resolution.",
            node_type="concept",
        )
        assert VaultWriter.okf_description(node) == "The Bark scale is non-linear."

    def test_abbreviation_not_split(self) -> None:
        node = WikiNode(
            id="x",
            title="X",
            content="Defined in ITU-T Rec. P.863 for speech quality.",
            node_type="concept",
        )
        # ". 8" is not a sentence boundary; whole line returned.
        assert VaultWriter.okf_description(node) == "Defined in ITU-T Rec. P.863 for speech quality."

    def test_long_line_capped(self) -> None:
        node = WikiNode(id="x", title="X", content="word " * 80, node_type="concept")
        desc = VaultWriter.okf_description(node)
        assert len(desc) <= 201
        assert desc.endswith("…")

    def test_skips_leading_heading_lines(self) -> None:
        """Regression: p863 wiki pages started with '## Definition', so the
        index description was a heading fragment. Must skip heading lines.
        """
        node = WikiNode(
            id="x",
            title="X",
            content="## Definition\nAbsolute Category Rating is a discrete rating scale. It uses five points.",
            node_type="concept",
        )
        assert VaultWriter.okf_description(node) == "Absolute Category Rating is a discrete rating scale."

    def test_skips_multiple_leading_headings(self) -> None:
        node = WikiNode(
            id="x",
            title="X",
            content="# Title\n## Definition\n## Methodology\nBody sentence starts here.",
            node_type="concept",
        )
        assert VaultWriter.okf_description(node) == "Body sentence starts here."

    def test_all_headings_returns_empty(self) -> None:
        node = WikiNode(
            id="x",
            title="X",
            content="# H1\n## H2\n### H3",
            node_type="concept",
        )
        assert VaultWriter.okf_description(node) == ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_wiki() -> CompiledWiki:
    """A CompiledWiki with a few nodes, edges, and TOC entries."""
    nodes = [
        WikiNode(
            id="intro",
            title="Introduction",
            content="This is the introduction section.",
            node_type="section",
            source_chunks=["chunk-1", "chunk-2"],
            metadata={"heading_level": 1},
        ),
        WikiNode(
            id="concept-1",
            title="Core Concepts",
            content="Understanding the fundamental ideas.",
            node_type="concept",
            source_chunks=["chunk-3"],
            metadata={"heading_level": 2},
        ),
        WikiNode(
            id="procedure-1",
            title="How to Use",
            content="Step-by-step instructions.",
            node_type="procedure",
            source_chunks=None,
            metadata={},
        ),
    ]
    edges: list[GraphEdge] = [
        {"source": "intro", "target": "concept-1", "type": "contains"},
        {"source": "concept-1", "target": "procedure-1", "type": "prerequisite"},
    ]
    toc: list[TocEntry] = [
        {"id": "intro", "title": "Introduction", "level": 1},
        {"id": "concept-1", "title": "Core Concepts", "level": 1},
        {"id": "procedure-1", "title": "How to Use", "level": 1},
    ]
    metadata = {"title": "Test Wiki", "description": "A wiki for testing"}
    return CompiledWiki(nodes=nodes, edges=edges, toc=toc, metadata=metadata)


@pytest.fixture
def empty_wiki() -> CompiledWiki:
    """A CompiledWiki with no nodes, edges, or TOC."""
    return CompiledWiki(nodes=[], edges=[], toc=[], metadata={})


@pytest.fixture
def temp_output(tmp_path: Path) -> Path:
    """Temporary output directory."""
    return tmp_path / "output"
