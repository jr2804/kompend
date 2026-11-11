"""`orphan-attachments`: a page link outranks the slug.

A vault carries attachment directories whose names come from an older slug
rule (the rule changed when sources moved into subdirectories, so the live
slug grew a path prefix). The compiled pages still link those directories,
and the parent source is alive — only the directory name is history.

The rule used to flag every file in such a directory as an orphan, and
``--fix-destructive` deleted them: 371 files and 1391 broken image links in
the p863 wiki (2026-11-11). A file a page links to is now kept and reported
as one PASS summary; only files no page links to stay in the FAIL set.
"""

from __future__ import annotations

from pathlib import Path

from kompend.core.models.attachments import adopt_orphan_attachments
from kompend.core.models.wiki import CompiledWiki
from kompend.core.pipeline.state import SourceState, TierState, TierStateStore
from kompend.lint.base import LintRuleOutcome
from kompend.lint.structural import OrphanAttachmentsRule, UnreferencedAttachmentsRule

_DEAD_SLUG = "polqaoemlibmanualv3"


def test_linked_legacy_file_is_not_an_orphan(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    linked = _legacy_file(vault, "PolqaOemLibManualV3.18__28e07839dcc55cae.png")
    _page(vault, linked.name)

    rule = OrphanAttachmentsRule()
    results = rule.check(_wiki(), sources_dir=str(vault / "sources"))

    assert [r.outcome for r in results] == [LintRuleOutcome.PASS]
    assert "still linked" in results[0].message
    assert results[0].details["kept_linked"] == 1


def test_the_destructive_fix_keeps_a_linked_legacy_file(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    linked = _legacy_file(vault, "PolqaOemLibManualV3.18__28e07839dcc55cae.png")
    _page(vault, linked.name)

    rule = OrphanAttachmentsRule()
    results = rule.check(_wiki(), sources_dir=str(vault / "sources"))
    rule.fix(_wiki(), results, sources_dir=vault / "sources")

    assert linked.is_file()


def test_an_unlinked_legacy_file_is_still_an_orphan(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    linked = _legacy_file(vault, "PolqaOemLibManualV3.18__28e07839dcc55cae.png")
    dead = _legacy_file(vault, "PolqaOemLibManualV3.18__06362fdd532bce28.png")
    _page(vault, linked.name)

    rule = OrphanAttachmentsRule()
    results = rule.check(_wiki(), sources_dir=str(vault / "sources"))
    fails = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
    assert [r.details["path"] for r in fails] == [dead.relative_to(vault).as_posix()]

    rule.fix(_wiki(), results, sources_dir=vault / "sources")
    assert not dead.exists()
    assert linked.is_file()


def test_an_unreadable_page_keeps_every_legacy_file(tmp_path: Path) -> None:
    """An undecodable page may link any legacy file — report nothing, delete nothing."""
    vault = _vault(tmp_path)
    dead = _legacy_file(vault, "PolqaOemLibManualV3.18__06362fdd532bce28.png")
    broken = vault / "wiki" / "concepts" / "broken.md"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"# Broken\n\n\xff\xfe not utf-8\n")

    rule = OrphanAttachmentsRule()
    results = rule.check(_wiki(), sources_dir=str(vault / "sources"))

    assert [r.outcome for r in results] == [LintRuleOutcome.SKIP]
    assert results[0].details["unreadable_pages"] == ["wiki/concepts/broken.md"]
    rule.fix(_wiki(), results, sources_dir=vault / "sources")
    assert dead.is_file()


def test_a_protected_sources_figure_is_kept_by_the_orphan_rule(tmp_path: Path) -> None:
    """A captured note owns its figures: the slug is not an orphan.

    The slug scan skipped protected sources but the liveness helper did not, so
    this vault used to report (and, opted in, delete) the figure of a note the
    capture lifecycle still owns.
    """
    vault = _captured_vault(tmp_path)
    figure = _captured_figure(vault)
    (vault / "wiki" / "concepts").mkdir(parents=True, exist_ok=True)

    rule = OrphanAttachmentsRule()
    results = rule.check(_wiki(), sources_dir=str(vault / "sources"))

    assert [r.outcome for r in results] == [LintRuleOutcome.PASS]
    rule.fix(_wiki(), results, sources_dir=vault / "sources")
    assert figure.is_file()


def test_both_attachment_rules_agree_on_a_protected_slug(tmp_path: Path) -> None:
    """Neither destructive rule touches a protected source's figures.

    This is the regression the three-copy slug computation hid: one rule said
    "orphan, delete it", the other said "live, keep it".
    """
    vault = _captured_vault(tmp_path)
    figure = _captured_figure(vault)

    orphan_rule = OrphanAttachmentsRule()
    orphan_results = orphan_rule.check(_wiki(), sources_dir=str(vault / "sources"))
    unreferenced_rule = UnreferencedAttachmentsRule()
    unreferenced_results = unreferenced_rule.check(_wiki(), sources_dir=str(vault / "sources"))

    assert [r.outcome for r in orphan_results] == [LintRuleOutcome.PASS]
    assert [r.outcome for r in unreferenced_results] == [LintRuleOutcome.PASS]
    orphan_rule.fix(_wiki(), orphan_results, sources_dir=vault / "sources")
    unreferenced_rule.fix(_wiki(), unreferenced_results, sources_dir=vault / "sources")
    assert figure.is_file()


def test_adoption_and_the_rules_share_the_live_slug_answer(tmp_path: Path) -> None:
    """Auto-adoption answers "is this slug live?" the same way both rules do.

    Adoption is the third copy of that question. It must agree: a protected
    slug is not adopted, an ordinary one is.
    """
    vault = _captured_vault(tmp_path)
    _captured_figure(vault)
    (vault / "sources" / "report.pdf").write_bytes(b"%PDF-1.7")
    ordinary = vault / "attachments" / "report" / "hand.txt"
    ordinary.parent.mkdir(parents=True, exist_ok=True)
    ordinary.write_text("x", encoding="utf-8")

    adopted = adopt_orphan_attachments(vault)

    assert [a.path for a in adopted] == ["report/hand.txt"]
    assert not any(a.path.startswith("captured_") for a in adopted)


def _captured_vault(tmp_path: Path) -> Path:
    """A vault whose only source sits under the protected ``sources/_captured/``."""
    captured = tmp_path / "sources" / "_captured"
    captured.mkdir(parents=True)
    (captured / "quick-note.md").write_text("# note\n", encoding="utf-8")
    store = TierStateStore(tmp_path / ".kompend" / "compile_state.json")
    store.save(TierState(entries={"captured_quick-note": SourceState(stem="quick-note", source_file="sources/_captured/quick-note.md", content_hash="h")}))
    return tmp_path


def _captured_figure(vault: Path) -> Path:
    """The figure that belongs to the captured note, in its slug directory."""
    figure = vault / "attachments" / "captured_quick-note" / "screenshot.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes(b"\x89PNG\r\n\x1a\n")
    return figure


def _vault(tmp_path: Path) -> Path:
    """A vault with one live source and a compile state the rule accepts."""
    (tmp_path / "sources" / "misc").mkdir(parents=True)
    (tmp_path / "sources" / "misc" / "PolqaOemLibManualV3.18.pdf").write_bytes(b"%PDF-1.7 source")
    store = TierStateStore(tmp_path / ".kompend" / "compile_state.json")
    store.save(
        TierState(
            entries={
                "misc_polqaoemlibmanualv3_18": SourceState(
                    stem="PolqaOemLibManualV3.18", source_file="sources/misc/PolqaOemLibManualV3.18.pdf", content_hash="h"
                )
            }
        )
    )
    return tmp_path


def _legacy_file(vault: Path, name: str) -> Path:
    att_dir = vault / "attachments" / _DEAD_SLUG
    att_dir.mkdir(parents=True, exist_ok=True)
    f = att_dir / name
    f.write_bytes(b"png bytes")
    return f


def _page(vault: Path, name: str) -> None:
    page = vault / "wiki" / "concepts" / "polqa.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        f"# POLQA\n\n![image](../../attachments/{_DEAD_SLUG}/{name})\n",
        encoding="utf-8",
    )


def _wiki() -> CompiledWiki:
    return CompiledWiki(nodes=[], edges=[], toc=[])
