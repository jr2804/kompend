"""`unreferenced-attachments`: writer copies no page links any more.

The vault writer copies every page-referenced figure into
``attachments/<source-slug>/`` and never removes a copy, so a page that stops
referencing an image leaves the file behind. Two facts make this rule's scope
precise:

- a directory whose slug maps to no live source is ``orphan-attachments``
  territory, so this rule skips it (the two rules never report the same file);
- auto-adoption registers writer copies in the manifest too (it cannot tell
  them from a manually dropped file), so "registered" does not mean "user
  file" — writer output is recognized by being byte-identical to an extracted
  image in ``.kompend/cache/images/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.models.attachments import Attachment, AttachmentManifest
from kompend.core.models.enums import NodeType
from kompend.core.paths import VaultPaths
from kompend.lint.base import LintRuleOutcome
from kompend.lint.runner import LintRunner
from kompend.lint.structural import UnreferencedAttachmentsRule, unreferenced_attachment_files

_STEM = "report"


def test_referenced_writer_copy_is_kept(tmp_path: Path) -> None:
    _source(tmp_path)
    link = _writer_copy(tmp_path, _STEM, "report_img_0.png")
    _page(tmp_path, link)
    assert unreferenced_attachment_files(tmp_path) == []


def test_unreferenced_writer_copy_is_flagged_and_deleted(tmp_path: Path) -> None:
    _source(tmp_path)
    link = _writer_copy(tmp_path, _STEM, "report_img_0.png")
    stale = _writer_copy(tmp_path, _STEM, "report_img_1.png")  # no longer embedded
    _page(tmp_path, link)

    assert unreferenced_attachment_files(tmp_path) == [stale]

    rule = UnreferencedAttachmentsRule()
    sources_dir = str(VaultPaths(tmp_path).sources_dir)
    results = rule.check(_wiki(), sources_dir=sources_dir)
    assert [r.outcome for r in results] == [LintRuleOutcome.FAIL]
    fixed = rule.fix(_wiki(), results)

    assert fixed[0].fix_applied is True
    assert not (tmp_path / stale).exists()
    assert (tmp_path / link).is_file()


def test_adopted_writer_copy_is_still_pruned_and_tombstoned(tmp_path: Path) -> None:
    """Auto-adoption registers writer output, so a manifest row is not ownership."""
    _source(tmp_path)
    stale = _writer_copy(tmp_path, _STEM, "report_img_2.png")
    _register(tmp_path, stale)  # exactly what adopt_orphan_attachments() does
    _page(tmp_path, None)

    assert unreferenced_attachment_files(tmp_path) == [stale]

    rule = UnreferencedAttachmentsRule()
    sources_dir = str(VaultPaths(tmp_path).sources_dir)
    results = rule.check(_wiki(), sources_dir=sources_dir)
    rule.fix(_wiki(), results)

    assert not (tmp_path / stale).exists()
    # The row survives as a tombstone rather than pointing at a missing file.
    manifest = AttachmentManifest(VaultPaths(tmp_path).attachments_manifest_file)
    assert stale[len("attachments/") :] in manifest.deleted_paths()


def test_registered_user_file_survives(tmp_path: Path) -> None:
    """A hand-added file is input, not writer output — keep it."""
    _source(tmp_path)
    kept = _user_file(tmp_path, _STEM, "handwritten.csv")
    _register(tmp_path, kept)
    _page(tmp_path, None)

    assert unreferenced_attachment_files(tmp_path) == []


def _register(base_dir: Path, rel: str) -> None:
    manifest = AttachmentManifest(VaultPaths(base_dir).attachments_manifest_file)
    inner = rel[len("attachments/") :]
    manifest.append(Attachment(path=inner, source=inner.split("/", 1)[0], ingestible=False))


def test_unregistered_user_file_survives(tmp_path: Path) -> None:
    """Not a cache copy → not writer output → not this rule's business."""
    _source(tmp_path)
    _user_file(tmp_path, _STEM, "dropped-in.pdf")
    _page(tmp_path, None)

    assert unreferenced_attachment_files(tmp_path) == []


def _user_file(base_dir: Path, slug: str, name: str, payload: bytes = b"notes") -> str:
    path = base_dir / "attachments" / slug / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return f"attachments/{slug}/{name}"


def test_dead_slug_directory_is_left_to_orphan_attachments(tmp_path: Path) -> None:
    """No live source for the slug → orphan-attachments owns it, not this rule."""
    _source(tmp_path)
    _writer_copy(tmp_path, "sources_report", "report_img_3.png")  # stale full-path slug
    _page(tmp_path, None)

    assert unreferenced_attachment_files(tmp_path) == []


def test_rule_is_destructive_so_plain_fix_is_gated(tmp_path: Path) -> None:
    """The lint runner must skip it unless --fix-destructive is passed.

    This is the guard from the p863 roundup: a destructive rule that ran on
    plain ``--fix`` emptied a vault's attachments once already.
    """
    _source(tmp_path)
    stale = _writer_copy(tmp_path, _STEM, "report_img_4.png")
    _page(tmp_path, None)
    sources_dir = str(VaultPaths(tmp_path).sources_dir)

    plain = LintRunner(rules=["unreferenced-attachments"], fix=True, fix_destructive=False, sources_dir=sources_dir)
    plain.run(_wiki())
    assert (tmp_path / stale).is_file(), "plain --fix must not delete attachments"

    opted_in = LintRunner(rules=["unreferenced-attachments"], fix=True, fix_destructive=True, sources_dir=sources_dir)
    opted_in.run(_wiki())
    assert not (tmp_path / stale).exists(), "--fix-destructive must delete them"


def test_a_linked_file_whose_name_contains_a_parenthesis_survives(tmp_path: Path) -> None:
    """A figure name may contain ")" — the link must not be cut at the first one.

    The p863 wiki links figures named
    ``Perceptual_Objective_Listening_Quality_Assessment_(POLQA),_…__0b467ca03243dedc.jpg``;
    a link target read up to the first ")" resolves to nothing, so the file
    looks unreferenced and the destructive rule deletes a page's own figure.
    """
    _source(tmp_path)
    name = "Perceptual_Objective_Listening_Quality_Assessment_(POLQA),_Standard__0b467ca03243dedc.jpg"
    link = _writer_copy(tmp_path, _STEM, name)
    _page(tmp_path, link)

    assert unreferenced_attachment_files(tmp_path) == []

    rule = UnreferencedAttachmentsRule()
    sources_dir = str(VaultPaths(tmp_path).sources_dir)
    results = rule.check(_wiki(), sources_dir=sources_dir)
    rule.fix(_wiki(), results)

    assert (tmp_path / link).is_file()


@pytest.mark.parametrize(
    "form",
    [
        '![fig]({link} "Figure 1")',
        "![fig](<{link}>)",
        "[figure]: {link}\n\n![fig][figure]",
        "[figure]: {link}\n",
        '<img src="{link}" alt="fig">',
        "[fig]({link})",
    ],
    ids=["title", "angle-brackets", "reference-used", "reference-declared", "html-img", "plain-link"],
)
def test_every_link_form_keeps_the_figure(tmp_path: Path, form: str) -> None:
    """One case per markdown form that used to resolve to nothing.

    A titled link, an angle-bracket destination, a reference definition and an
    HTML ``src`` each left the target unparsed, so the file looked unreferenced
    and ``--fix-destructive`` deleted a page's own figure.
    """
    _source(tmp_path)
    link = _writer_copy(tmp_path, _STEM, "report_img_9.png")
    _page_raw(tmp_path, form.format(link=f"../../{link}"))

    assert unreferenced_attachment_files(tmp_path) == []

    rule = UnreferencedAttachmentsRule()
    results = rule.check(_wiki(), sources_dir=str(VaultPaths(tmp_path).sources_dir))
    fixed = rule.fix(_wiki(), results)

    assert [r.outcome for r in fixed if r.fix_applied] == []
    assert (tmp_path / link).is_file()


@pytest.mark.parametrize(
    "form",
    ["```\n![fig]({link})\n```", "`![fig]({link})`"],
    ids=["fenced-code", "inline-code"],
)
def test_a_code_example_does_not_protect_a_file(tmp_path: Path, form: str) -> None:
    """Code is example text, not a published reference — the file stays prunable."""
    _source(tmp_path)
    stale = _writer_copy(tmp_path, _STEM, "report_img_10.png")
    _page_raw(tmp_path, form.format(link=f"../../{stale}"))

    assert unreferenced_attachment_files(tmp_path) == [stale]


def test_an_unreadable_page_does_not_condemn_the_figure_it_links(tmp_path: Path) -> None:
    """A page nobody can decode links *something* — never treat it as linking nothing.

    The scanner used to swallow ``UnicodeDecodeError``, so a page with one bad
    byte sequence read as "links nothing", its figure looked stale and
    ``--fix-destructive`` deleted it without a trace.
    """
    _source(tmp_path)
    stale = _writer_copy(tmp_path, _STEM, "report_img_11.png")
    _page(tmp_path, stale)
    broken = tmp_path / "wiki" / "concepts" / "broken.md"
    broken.write_bytes(b"# Broken\n\n\xff\xfe not utf-8\n")

    assert unreferenced_attachment_files(tmp_path) == []

    rule = UnreferencedAttachmentsRule()
    results = rule.check(_wiki(), sources_dir=str(VaultPaths(tmp_path).sources_dir))

    assert [r.outcome for r in results] == [LintRuleOutcome.SKIP]
    assert results[0].details["unreadable_pages"] == ["wiki/concepts/broken.md"]
    assert "could not be read" in results[0].message
    rule.fix(_wiki(), results)
    assert (tmp_path / stale).is_file()


def test_an_unreadable_page_is_reported_at_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The undecodable page must be visible in the log, not silently skipped."""
    _source(tmp_path)
    _writer_copy(tmp_path, _STEM, "report_img_12.png")
    broken = tmp_path / "wiki" / "concepts" / "broken.md"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"\xff\xfe")

    with caplog.at_level(logging.WARNING, logger="kompend.lint.structural.attachments"):
        unreferenced_attachment_files(tmp_path)

    assert any("wiki/concepts/broken.md" in record.getMessage() for record in caplog.records)


def test_a_readable_page_does_not_disable_the_rule(tmp_path: Path) -> None:
    """The fail-closed path is for unreadable pages only — a clean vault still prunes."""
    _source(tmp_path)
    stale = _writer_copy(tmp_path, _STEM, "report_img_13.png")
    _page(tmp_path, None)

    rule = UnreferencedAttachmentsRule()
    results = rule.check(_wiki(), sources_dir=str(VaultPaths(tmp_path).sources_dir))

    assert [r.outcome for r in results] == [LintRuleOutcome.FAIL]
    rule.fix(_wiki(), results)
    assert not (tmp_path / stale).exists()


def test_an_unreadable_cache_entry_keeps_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A cache probe that fails must not read as "not a writer copy" — keep the file.

    ``_is_cached_extraction`` turned ``OSError`` into ``False``, which is the
    answer that lets the destructive rule delete. A file the rule cannot
    compare is a file it cannot prove expendable.
    """
    _source(tmp_path)
    kept = _writer_copy(tmp_path, _STEM, "report_img_14.png")
    _page(tmp_path, None)
    cache_file = VaultPaths(tmp_path).images_dir / "report_img_14.png"
    real_read_bytes = Path.read_bytes

    def _refuse(self: Path) -> bytes:
        # The cache copy is readable; the attachment copy is not.
        if self == tmp_path / kept:
            raise OSError(5, "I/O error", str(self))
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _refuse)
    with caplog.at_level(logging.WARNING, logger="kompend.lint.structural.attachments"):
        flagged = unreferenced_attachment_files(tmp_path)

    assert flagged == []
    assert any("report_img_14.png" in record.getMessage() for record in caplog.records)
    assert cache_file.is_file()


def _source(base_dir: Path) -> None:
    """A live source, so the attachments dir counts as a live slug."""
    sources = VaultPaths(base_dir).sources_dir
    sources.mkdir(parents=True, exist_ok=True)
    (sources / f"{_STEM}.pdf").write_bytes(b"%PDF-1.7")


def _writer_copy(base_dir: Path, slug: str, name: str) -> str:
    path = base_dir / "attachments" / slug / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_cached_image(base_dir, name))
    return f"attachments/{slug}/{name}"


def _cached_image(base_dir: Path, name: str) -> bytes:
    """An extracted figure in the parse cache — the copy source."""
    images = VaultPaths(base_dir).images_dir
    images.mkdir(parents=True, exist_ok=True)
    payload = f"png:{name}".encode()
    (images / name).write_bytes(payload)
    return payload


def _page(base_dir: Path, link: str | None) -> None:
    page = base_dir / "wiki" / "concepts" / "one.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    embed = f"\n![fig](../../{link})\n" if link else "\nNo media here.\n"
    page.write_text(f"---\ntitle: One\n---\n\n# One\n{embed}", encoding="utf-8")


def _page_raw(base_dir: Path, body: str) -> None:
    """A page whose body is written verbatim (one link form per test)."""
    page = base_dir / "wiki" / "concepts" / "one.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(f"---\ntitle: One\n---\n\n# One\n\n{body}\n", encoding="utf-8")


def _wiki() -> CompiledWiki:
    return CompiledWiki(nodes=[WikiNode(id="one", title="One", content="Body.", node_type=NodeType.CONCEPT)], edges=[], toc=[])
