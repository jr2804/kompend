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
from kompend.output.markdown import MarkdownWriter

_SOURCE_A = "ECMA-418-2_4th_edition_june_2025.pdf"
_SOURCE_B = "ISO226.pdf"


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
