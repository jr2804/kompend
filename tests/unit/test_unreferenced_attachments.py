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
  image in ``.knox/cache/images/``.
"""

from __future__ import annotations

from pathlib import Path

from knox.core.models import CompiledWiki, WikiNode
from knox.core.models.attachments import Attachment, AttachmentManifest
from knox.core.paths import VaultPaths
from knox.lint.base import LintRuleOutcome
from knox.lint.runner import LintRunner
from knox.lint.structural import UnreferencedAttachmentsRule, unreferenced_attachment_files

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


def _wiki() -> CompiledWiki:
    return CompiledWiki(nodes=[WikiNode(id="one", title="One", content="Body.", node_type="concept")], edges=[], toc=[])
