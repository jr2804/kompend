"""`portablize_images`: slug from the image's source, drop what cannot be attributed.

Covers the two writer-side regressions from the hearing-model vault:

- the slug came from ``provenance[0]``, so an agent-reported label that names no
  source (``Proc. DAGA``, ``J. Acoust. Soc. Am.``) created an ``attachments/``
  directory of its own, filled with copies of unrelated figures;
- nothing stopped an image whose provenance matched no entry, which would have
  left a ``.kompend/`` link in a page (the markdown-only invariant forbids that).
"""

from __future__ import annotations

from pathlib import Path

from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.models.enums import NodeType
from kompend.core.paths import VaultPaths, attachment_slug, attachment_slugs_for_source
from kompend.core.pipeline.state import SourceState, TierState, TierStateStore
from kompend.lint.base import LintRuleOutcome
from kompend.lint.structural import OrphanAttachmentsRule
from kompend.output.markdown import MarkdownWriter

_SOURCE_A = "ECMA-418-2_4th_edition_june_2025.pdf"
_SOURCE_B = "ISO226.pdf"


def test_a_split_childs_figure_lands_in_the_parents_directory(tmp_path: Path) -> None:
    """A child is virtual: its figure is the parent's image and belongs to the parent.

    The child slug (``packet_p005-009``) matches no file under ``sources/``, so
    the orphan rule would delete that directory on the next ``lint --fix`` and
    leave the page pointing at nothing. The figure carries ``attachment_owner``
    for exactly this reason; attribution (``source_file``) stays the child.
    """
    child = "packet#p005-009.pdf"
    fig = _seed_image(tmp_path, "packet__fig9.png")
    wiki = _wiki(
        _node(
            "page",
            [child],
            [
                {
                    "path": fig,
                    "caption": "",
                    "element_type": "image",
                    "source_file": f"sources/{child}",
                    "attachment_owner": "sources/packet.pdf",
                }
            ],
        )
    )

    MarkdownWriter(vault_mode=True, base_dir=tmp_path).write(wiki, tmp_path / "wiki")

    assert (tmp_path / "attachments" / "packet" / "packet__fig9.png").is_file()
    assert not (tmp_path / "attachments" / "packet_p005-009").exists()
    page = (tmp_path / "wiki" / "concepts" / "page.md").read_text(encoding="utf-8")
    # Attribution is unchanged: the child's page still embeds the figure.
    assert "../../attachments/packet/packet__fig9.png" in page


def test_an_absolute_owner_path_slugs_the_same_directory(tmp_path: Path) -> None:
    """The pipeline stamps the real source path, which is absolute."""
    child = "packet#p005-009.pdf"
    fig = _seed_image(tmp_path, "packet__fig9.png")
    wiki = _wiki(
        _node(
            "page",
            [child],
            [
                {
                    "path": fig,
                    "caption": "",
                    "element_type": "image",
                    "source_file": f"sources/{child}",
                    "attachment_owner": str(tmp_path / "sources" / "packet.pdf"),
                }
            ],
        )
    )

    MarkdownWriter(vault_mode=True, base_dir=tmp_path).write(wiki, tmp_path / "wiki")

    assert (tmp_path / "attachments" / "packet" / "packet__fig9.png").is_file()
    assert not (tmp_path / "attachments" / "packet_p005-009").exists()


def test_an_ordinary_source_still_slugs_its_own_directory(tmp_path: Path) -> None:
    """No owner key: the image's own source keeps deciding the directory."""
    fig = _seed_image(tmp_path, "report__fig1.png")
    wiki = _wiki(
        _node(
            "page",
            ["report"],
            [{"path": fig, "caption": "", "element_type": "image", "source_file": "sources/report.pdf"}],
        )
    )

    MarkdownWriter(vault_mode=True, base_dir=tmp_path).write(wiki, tmp_path / "wiki")

    assert (tmp_path / "attachments" / "report" / "report__fig1.png").is_file()


def test_the_parents_directory_survives_lint_fix(tmp_path: Path) -> None:
    """The whole point: after a compile, ``lint --fix`` must not eat the figure.

    The orphan rule derives its live slugs from real files under ``sources/``.
    The parent's slug is live; the child's synthetic slug is not — which is why
    the writer may not use it.
    """
    (tmp_path / "sources").mkdir(parents=True)
    (tmp_path / "sources" / "packet.pdf").write_bytes(b"%PDF fake")
    _seed_compile_state(tmp_path, "packet")
    child = "packet#p005-009.pdf"
    fig = _seed_image(tmp_path, "packet__fig9.png")
    wiki = _wiki(
        _node(
            "page",
            [child],
            [
                {
                    "path": fig,
                    "caption": "",
                    "element_type": "image",
                    "source_file": f"sources/{child}",
                    "attachment_owner": "sources/packet.pdf",
                }
            ],
        )
    )
    MarkdownWriter(vault_mode=True, base_dir=tmp_path).write(wiki, tmp_path / "wiki")

    rule = OrphanAttachmentsRule()
    results = rule.check(CompiledWiki(nodes=[], edges=[], toc=[]), sources_dir=str(tmp_path / "sources"))

    # The rule really ran (a compile happened, an attachments/ tree exists) and
    # found the parent's directory live — the figure is not an orphan.
    assert [r.outcome for r in results] == [LintRuleOutcome.PASS]
    # Running the destructive fix over that outcome removes nothing.
    rule.fix(CompiledWiki(nodes=[], edges=[], toc=[]), results, sources_dir=str(tmp_path / "sources"))
    assert (tmp_path / "attachments" / "packet" / "packet__fig9.png").is_file()


def _seed_compile_state(base_dir: Path, stem: str) -> None:
    """A compile ran: without it the orphan rule declines to judge anything."""
    TierStateStore(VaultPaths(base_dir).compile_state_file).save(
        TierState(entries={stem: SourceState(stem=stem, source_file="sources/packet.pdf", content_hash="h")})
    )


def test_a_childs_own_slug_is_not_a_live_source_slug(tmp_path: Path) -> None:
    """Why the writer cannot keep the matched entry's slug for a child's figure."""
    (tmp_path / "sources").mkdir(parents=True)
    (tmp_path / "sources" / "packet.pdf").write_bytes(b"%PDF fake")

    live = {slug for source in (tmp_path / "sources").iterdir() for slug in attachment_slugs_for_source(source.relative_to(tmp_path / "sources").as_posix())}

    assert "packet" in live
    assert "packet#p005-009" not in live
    assert attachment_slug("packet#p005-009.pdf") not in live


def test_slug_comes_from_the_images_source_not_provenance_order(tmp_path: Path) -> None:
    """Each source's figures land in that source's directory, whatever the order."""
    a = _seed_image(tmp_path, "a_img_0.png")
    b = _seed_image(tmp_path, "b_img_0.png")
    wiki = _wiki(
        _node(
            "page",
            # The agent lists an unrelated venue first — the old code slugged this.
            ["Proc. DAGA", _SOURCE_A, _SOURCE_B],
            [
                {"path": a, "caption": "", "element_type": "image", "source_file": f"sources/{_SOURCE_A}"},
                {"path": b, "caption": "", "element_type": "image", "source_file": f"sources/{_SOURCE_B}"},
            ],
        )
    )

    MarkdownWriter(vault_mode=True, base_dir=tmp_path).write(wiki, tmp_path / "wiki")

    assert (tmp_path / "attachments" / "ecma-418-2_4th_edition_june_2025" / "a_img_0.png").is_file()
    assert (tmp_path / "attachments" / "iso226" / "b_img_0.png").is_file()
    assert not (tmp_path / "attachments" / "proc_daga").exists()


def test_page_embeds_relative_links_into_attachments(tmp_path: Path) -> None:
    a = _seed_image(tmp_path, "a_img_0.png")
    wiki = _wiki(
        _node(
            "page",
            [_SOURCE_A],
            [{"path": a, "caption": "cap", "element_type": "image", "source_file": f"sources/{_SOURCE_A}"}],
        )
    )

    MarkdownWriter(vault_mode=True, base_dir=tmp_path).write(wiki, tmp_path / "wiki")

    page = (tmp_path / "wiki" / "concepts" / "page.md").read_text(encoding="utf-8")
    assert "../../attachments/ecma-418-2_4th_edition_june_2025/a_img_0.png" in page
    assert ".kompend/" not in page


def test_unattributable_image_is_dropped_not_linked(tmp_path: Path) -> None:
    """No matching provenance entry → no image, no invented directory, no .kompend link."""
    stray = _seed_image(tmp_path, "stray_img_0.png")
    wiki = _wiki(
        _node(
            "page",
            ["Proc. DAGA"],
            [{"path": stray, "caption": "", "element_type": "image", "source_file": "sources/Unknown.pdf"}],
        )
    )

    MarkdownWriter(vault_mode=True, base_dir=tmp_path).write(wiki, tmp_path / "wiki")

    page = (tmp_path / "wiki" / "concepts" / "page.md").read_text(encoding="utf-8")
    assert "attachments/" not in page
    assert ".kompend/" not in page
    assert not (tmp_path / "attachments").exists()


def _seed_image(base_dir: Path, name: str) -> str:
    """Write a fake extracted figure where the parser would put it."""
    images = base_dir / ".kompend" / "cache" / "images"
    images.mkdir(parents=True, exist_ok=True)
    path = images / name
    path.write_bytes(f"png:{name}".encode())
    return f".kompend/cache/images/{name}"


def _wiki(*nodes: WikiNode) -> CompiledWiki:
    return CompiledWiki(nodes=list(nodes), edges=[], toc=[], metadata={"title": "T"})


def _node(node_id: str, provenance: list[str], images: list[dict[str, str]]) -> WikiNode:
    return WikiNode(
        id=node_id,
        title=node_id.title(),
        content=f"Body of {node_id}.",
        node_type=NodeType.CONCEPT,
        metadata={"provenance": provenance, "images": images},
    )
