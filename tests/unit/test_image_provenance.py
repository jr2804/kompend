"""Per-source image scoping (`image_source_key`, `images_for_provenance`).

A page embeds the figures of the sources it was generated from. The regression
this pins: the compile stamped the *run-global* image list onto every page, so
each page embedded all 259 figures of the corpus and the writer copied all of
them into one ``attachments/<slug>/`` directory per provenance value — 24
directories holding the same file set (401 MB) on the hearing-model vault.
"""

from __future__ import annotations

from kompend.core.models import ImageRef, image_source_key, images_for_provenance


def test_key_normalizes_label_path_and_absolute_path() -> None:
    """The three shapes provenance and chunk metadata arrive in must agree."""
    label = "ECMA-418-2_4th_edition_june_2025"
    relative = f"sources/{label}.pdf"
    absolute = f"D:/vault/sources/{label}.pdf"

    assert image_source_key(label) == image_source_key(relative) == image_source_key(absolute)
    assert image_source_key(relative) == label.casefold()


def test_key_is_extension_aware_for_multi_dot_names() -> None:
    """ITU names keep their dots (Path.stem would chop at '.10')."""
    assert image_source_key("T-REC-P.10-201906-I!Amd1!PDF-E.pdf") == "t-rec-p.10-201906-i!amd1!pdf-e"


def test_only_the_pages_own_sources_figures_are_selected() -> None:
    images = [_img("a1.png", "sources/A.pdf"), _img("a2.png", "sources/A.pdf"), _img("b1.png", "sources/B.pdf")]

    assert [i["path"] for i in images_for_provenance(images, ["A"])] == ["a1.png", "a2.png"]
    assert [i["path"] for i in images_for_provenance(images, ["B"])] == ["b1.png"]
    assert [i["path"] for i in images_for_provenance(images, ["A", "B"])] == ["a1.png", "a2.png", "b1.png"]


def test_invented_provenance_contributes_no_images() -> None:
    """Agents report venues and titles as provenance ('Proc. DAGA').

    Those match no source, so the page embeds no figures — the alternative was
    embedding every figure in the run under a directory named after the label.
    """
    images = [_img("a1.png", "sources/A.pdf")]
    assert images_for_provenance(images, ["Proc. DAGA"]) == []
    assert images_for_provenance(images, []) == []


def test_image_without_source_is_never_selected() -> None:
    """A figure with no source attribution cannot be claimed by a page."""
    images = [_img("x.png", "")]
    assert images_for_provenance(images, ["A"]) == []
    assert images_for_provenance(images, [""]) == []


def _img(path: str, source: str) -> ImageRef:
    return ImageRef(path=path, caption="", element_type="image", source_file=source)
