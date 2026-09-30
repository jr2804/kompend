"""`unreferenced-attachments`: published media no page links any more.

The vault writer copies every page-referenced figure into
``attachments/<source-slug>/`` and never removes a copy, so a page that stops
referencing an image (or a slug that changes, e.g. the old full-path form
``sources_report`` → ``report``) leaves the file behind. The rule reports those
files and, only under ``--fix-destructive``, deletes them — manifest-registered
files are user input and survive.
"""

from __future__ import annotations

from pathlib import Path

from knox.core.models import CompiledWiki, WikiNode
from knox.core.models.attachments import Attachment, AttachmentManifest
from knox.core.paths import VaultPaths
from knox.lint.base import LintRuleOutcome
from knox.lint.runner import LintRunner
from knox.lint.structural import UnreferencedAttachmentsRule, unreferenced_attachment_files


def test_referenced_file_is_not_flagged(tmp_path: Path) -> None:
    link = _attachment(tmp_path, "report", "report_img_0.png")
    _page(tmp_path, link)
    assert unreferenced_attachment_files(tmp_path) == []


def test_unreferenced_file_is_flagged_and_deleted(tmp_path: Path) -> None:
    link = _attachment(tmp_path, "report", "report_img_0.png")
    _page(tmp_path, link)
    stale = _attachment(tmp_path, "sources_report", "report_img_1.png")  # old slug form

    assert unreferenced_attachment_files(tmp_path) == [stale]

    rule = _rule(tmp_path)
    results = rule.check(_wiki(), sources_dir=str(VaultPaths(tmp_path).sources_dir))
    assert [r.outcome for r in results] == [LintRuleOutcome.FAIL]
    fixed = rule.fix(_wiki(), results)

    assert fixed[0].fix_applied is True
    assert not (tmp_path / stale).exists()
    # The emptied slug directory goes with it; the referenced file survives.
    assert not (tmp_path / "attachments" / "sources_report").exists()
    assert (tmp_path / link).is_file()


def _page(base_dir: Path, link: str) -> None:
    page = base_dir / "wiki" / "concepts" / "one.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    link_from_page = link.replace("attachments/", "../../attachments/", 1)
    page.write_text(f"---\ntitle: One\n---\n\n# One\n\n![fig]({link_from_page})\n", encoding="utf-8")


def _rule(base_dir: Path) -> UnreferencedAttachmentsRule:
    rule = UnreferencedAttachmentsRule()
    sources_dir = str(VaultPaths(base_dir).sources_dir)
    rule.check(_wiki(), sources_dir=sources_dir)  # populates _last_sources_dir for fix()
    return rule


def test_manifest_registered_file_survives(tmp_path: Path) -> None:
    """A user-added attachment is input, not writer output — keep it."""
    _attachment(tmp_path, "report", "report_img_0.png")
    _attachment(tmp_path, "report", "handwritten.csv")
    manifest = AttachmentManifest(VaultPaths(tmp_path).attachments_manifest_file)
    manifest.append(Attachment(path="report/handwritten.csv", source="report", ingestible=False))

    stale = unreferenced_attachment_files(tmp_path)
    # The registered file is input and stays; the unregistered copy is stale.
    assert stale == ["attachments/report/report_img_0.png"]


def test_rule_is_destructive_so_plain_fix_is_gated(tmp_path: Path) -> None:
    """The lint runner must skip it unless --fix-destructive is passed.

    This is the guard from the p863 roundup: a destructive rule that ran on
    plain ``--fix`` emptied a vault's attachments once already.
    """
    stale = _attachment(tmp_path, "report", "report_img_0.png")
    sources_dir = str(VaultPaths(tmp_path).sources_dir)

    plain = LintRunner(rules=["unreferenced-attachments"], fix=True, fix_destructive=False, sources_dir=sources_dir)
    plain.run(_wiki())
    assert (tmp_path / stale).is_file(), "plain --fix must not delete attachments"

    opted_in = LintRunner(rules=["unreferenced-attachments"], fix=True, fix_destructive=True, sources_dir=sources_dir)
    opted_in.run(_wiki())
    assert not (tmp_path / stale).exists(), "--fix-destructive must delete them"


def _attachment(base_dir: Path, slug: str, name: str) -> str:
    path = base_dir / "attachments" / slug / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")
    return f"attachments/{slug}/{name}"


def _wiki() -> CompiledWiki:
    return CompiledWiki(nodes=[WikiNode(id="one", title="One", content="Body.", node_type="concept")], edges=[], toc=[])
