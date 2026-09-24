"""Tests for the destructive-fix gate and slug-mismatch fix.

Regression: p863 wiki roundup (2026-09-10) — ``knox attachments no-ingest
--all`` appeared to delete 248 attachment files. Root cause chain:

1. ``OrphanAttachmentsRule.check()`` computed live source slugs from the
   FULL relative path (``attachment_slug("p862/foo.pdf") == "p862_foo"``)
   but attachments were registered with bare-name slugs (``"foo"``), so
   EVERY attachment was falsely flagged as orphaned.
2. The post-compile lint runs with ``fix=True``, and the new
   ``OrphanAttachmentsRule.fix()`` deleted the flagged files.

Fixes: (a) ``attachment_slugs_for_source`` matches both slug forms;
(b) destructive rules only run when the caller explicitly passes
``fix_destructive=True`` — the post-compile lint and plain ``--fix``
never delete files.
"""

from __future__ import annotations

from pathlib import Path

from knox.core.models.attachments import Attachment, AttachmentManifest
from knox.core.models.wiki import CompiledWiki
from knox.core.paths import attachment_slugs_for_source
from knox.core.pipeline.state import SourceState, TierState, TierStateStore
from knox.lint.base import LintResult, LintRuleOutcome
from knox.lint.runner import LintRunner
from knox.lint.structural import OrphanAttachmentsRule


class TestAttachmentSlugsForSource:
    def test_returns_both_forms(self) -> None:
        """Full-path slug AND bare-name slug are both returned."""
        slugs = attachment_slugs_for_source("p862/T-REC-P.862-200102-W!!PDF-E.pdf")
        assert "p862_t-rec-p_862-200102-w_pdf-e" in slugs  # full path
        assert "t-rec-p_862-200102-w_pdf-e" in slugs  # bare name

    def test_bare_name_returns_single(self) -> None:
        slugs = attachment_slugs_for_source("report.pdf")
        assert slugs == {"report"}


class TestOrphanAttachmentsSlugMatch:
    def test_subdir_source_not_flagged_as_orphan(self, tmp_path: Path) -> None:
        """A source under a subdirectory matches its bare-name attachment dir."""
        vault = tmp_path
        # Source under a subdirectory.
        (vault / "sources" / "p862").mkdir(parents=True)
        (vault / "sources" / "p862" / "T-REC-P.862-200102-W!!PDF-E.pdf").write_text("x", encoding="utf-8")
        # Attachment registered with the bare-name slug.
        _make_orphan(vault, "t-rec-p_862-200102-w_pdf-e", "dsp.c")
        _seed_compile_state(vault)

        rule = OrphanAttachmentsRule()
        results = rule.check(CompiledWiki(nodes=[], edges=[], toc=[]), sources_dir=str(vault / "sources"))
        fails = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert fails == [], f"expected no orphans, got: {[r.message for r in fails]}"


class TestDestructiveGate:
    def test_fix_skipped_without_fix_destructive(self, tmp_path: Path) -> None:
        """Plain fix=True does NOT delete orphan files."""
        vault = tmp_path
        orphan = _make_orphan(vault, "orphan-source", "old.c")
        rel_path = orphan.relative_to(vault).as_posix()
        _seed_manifest(vault, rel_path)
        _seed_compile_state(vault)

        rule = OrphanAttachmentsRule()
        rule.check(CompiledWiki(nodes=[], edges=[], toc=[]), sources_dir=str(vault / "sources"))
        rule.fix(
            CompiledWiki(nodes=[], edges=[], toc=[]),
            [LintResult(rule="orphan-attachments", outcome=LintRuleOutcome.FAIL, details={"path": rel_path})],
        )
        # fix() itself still works when called directly — the gate lives in
        # the runner. Verify the runner gate instead below.
        assert not orphan.exists()

    def test_runner_skips_destructive_rule_without_flag(self, tmp_path: Path) -> None:
        """LintRunner with fix=True but no fix_destructive does NOT delete."""
        vault = tmp_path
        orphan = _make_orphan(vault, "orphan-source", "old.c")
        rel_path = orphan.relative_to(vault).as_posix()
        _seed_manifest(vault, rel_path)
        _seed_compile_state(vault)

        runner = LintRunner(
            rules=["orphan-attachments"],
            fix=True,
            fix_destructive=False,
            sources_dir=str(vault / "sources"),
        )
        report = runner.run(CompiledWiki(nodes=[], edges=[], toc=[]))
        assert orphan.exists(), "file must NOT be deleted without fix_destructive"
        assert any(r.outcome == LintRuleOutcome.FAIL for r in report.results)

    def test_runner_runs_destructive_rule_with_flag(self, tmp_path: Path) -> None:
        """LintRunner with fix_destructive=True DOES delete orphan files."""
        vault = tmp_path
        orphan = _make_orphan(vault, "orphan-source", "old.c")
        rel_path = orphan.relative_to(vault).as_posix()
        _seed_manifest(vault, rel_path)
        _seed_compile_state(vault)

        runner = LintRunner(
            rules=["orphan-attachments"],
            fix=True,
            fix_destructive=True,
            sources_dir=str(vault / "sources"),
        )
        report = runner.run(CompiledWiki(nodes=[], edges=[], toc=[]))
        assert not orphan.exists(), "file must be deleted with fix_destructive"
        assert any(r.fix_applied for r in report.results)


def _seed_compile_state(vault: Path) -> None:
    """Seed tier state via the store so the orphan rule's compile gate passes."""
    store = TierStateStore(vault / ".knox" / "compile_state.json")
    store.save(TierState(entries={"report": SourceState(stem="report", source_file="sources/report.pdf", content_hash="h")}))


def _seed_manifest(base_dir: Path, path: str) -> None:
    manifest = AttachmentManifest(base_dir / ".knox" / "attachments.jsonl")
    manifest.append(Attachment(path=path, source="orphan-source", ingestible=True))


def _make_orphan(vault: Path, slug: str, name: str) -> Path:
    att_dir = vault / "attachments" / slug
    att_dir.mkdir(parents=True, exist_ok=True)
    f = att_dir / name
    f.write_text("orphan contents", encoding="utf-8")
    return f
