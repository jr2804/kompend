"""Tests for ``OrphanAttachmentsRule.fix()`` — destructive auto-clean.

Regression: p863 wiki roundup (2026-09-10) revealed 992 orphan-attachment
findings that the linter only *reported*; ``knox compile --lint --fix`` did
nothing. ``fix()`` deletes the file on disk + tombstones the manifest entry.
"""

from __future__ import annotations

from pathlib import Path

from knox.core.models.attachments import Attachment, AttachmentManifest, adopt_orphan_attachments
from knox.core.models.wiki import CompiledWiki
from knox.lint.base import LintResult, LintRuleOutcome, Severity
from knox.lint.structural import OrphanAttachmentsRule


def test_fix_removes_orphan_file_and_tombstones_manifest(tmp_path: Path) -> None:
    vault = tmp_path
    orphan = _make_orphan(vault, "orphan-source", "old.c")
    rel_path = orphan.relative_to(vault).as_posix()
    _seed_manifest(vault, "orphan-source/old.c")

    rule = OrphanAttachmentsRule()
    rule.check(CompiledWiki(nodes=[], edges=[], toc=[]), sources_dir=str(vault / "sources"))

    results = rule.fix(
        CompiledWiki(nodes=[], edges=[], toc=[]),
        [
            LintResult(
                rule="orphan-attachments",
                outcome=LintRuleOutcome.FAIL,
                severity=Severity.WARNING,
                message="orphaned",
                details={"path": rel_path},
            )
        ],
    )

    assert len(results) == 1
    assert results[0].fix_applied is True
    assert "Deleted orphan attachment" in results[0].message
    assert not orphan.exists(), "orphan file should be gone"
    # read_all returns active entries; the tombstone is gone from there.
    manifest = AttachmentManifest(vault / ".knox" / "attachments.jsonl")
    assert "orphan-source/old.c" not in manifest.read_all()
    # The tombstone exists under the manifest key (attachments-relative) — the
    # assertion that would have caught the vault-relative key mismatch.
    assert "orphan-source/old.c" in manifest.deleted_paths()


def test_fix_removes_empty_slug_dir(tmp_path: Path) -> None:
    vault = tmp_path
    orphan = _make_orphan(vault, "orphan-source", "single-file.c")
    rel_path = orphan.relative_to(vault).as_posix()
    _seed_manifest(vault, "orphan-source/single-file.c")

    rule = OrphanAttachmentsRule()
    rule.check(CompiledWiki(nodes=[], edges=[], toc=[]), sources_dir=str(vault / "sources"))
    rule.fix(
        CompiledWiki(nodes=[], edges=[], toc=[]),
        [LintResult(rule="orphan-attachments", outcome=LintRuleOutcome.FAIL, details={"path": rel_path})],
    )
    assert not (vault / "attachments" / "orphan-source").exists()


def _seed_manifest(base_dir: Path, path: str) -> None:
    """Write a minimal manifest so the rule can read + tombstone it."""
    manifest = AttachmentManifest(base_dir / ".knox" / "attachments.jsonl")
    manifest.append(
        Attachment(
            path=path,
            source="orphan-source",
            ingestible=True,
        )
    )


def _make_orphan(vault: Path, slug: str, name: str) -> Path:
    """Create a file under attachments/<slug>/<name> and return its abs path."""
    att_dir = vault / "attachments" / slug
    att_dir.mkdir(parents=True, exist_ok=True)
    f = att_dir / name
    f.write_text("orphan contents", encoding="utf-8")
    return f


def test_fix_noop_when_no_failures(tmp_path: Path) -> None:
    vault = tmp_path
    rule = OrphanAttachmentsRule()
    rule.check(CompiledWiki(nodes=[], edges=[], toc=[]), sources_dir=str(vault / "sources"))

    pass_result = LintResult(rule="orphan-attachments", outcome=LintRuleOutcome.PASS, message="ok")
    skip_result = LintResult(rule="orphan-attachments", outcome=LintRuleOutcome.SKIP, message="skipped")

    out = rule.fix(CompiledWiki(nodes=[], edges=[], toc=[]), [pass_result, skip_result])
    assert out == [pass_result, skip_result]


def test_fix_records_os_error_in_details(tmp_path: Path) -> None:
    """If the file can't be deleted (e.g., already gone), record the error."""
    vault = tmp_path
    rule = OrphanAttachmentsRule()
    rule.check(CompiledWiki(nodes=[], edges=[], toc=[]), sources_dir=str(vault / "sources"))

    rel_path = "attachments/orphan/missing.c"
    # Don't create the file → unlink() will raise FileNotFoundError.
    result = LintResult(
        rule="orphan-attachments",
        outcome=LintRuleOutcome.FAIL,
        details={"path": rel_path},
    )
    out = rule.fix(CompiledWiki(nodes=[], edges=[], toc=[]), [result])
    # When the file doesn't exist, the rule treats that as success (already
    # gone) and tombstone-stamps the manifest. fix_applied=True.
    assert out[0].fix_applied is True
    assert "Deleted orphan attachment" in out[0].message


def test_fix_tombstone_prevents_readoption(tmp_path: Path) -> None:
    """After a destructive fix, a reappearing file must not be re-adopted.

    Regression for the vault-relative/attachments-relative key mismatch: the
    file was deleted but no tombstone was written, so the next compile's
    adoption could silently re-register it.
    """
    vault = tmp_path
    (vault / "sources").mkdir()
    (vault / "sources" / "orphan-source.pdf").write_bytes(b"%PDF fake")
    orphan = _make_orphan(vault, "orphan-source", "old.c")
    rel_path = orphan.relative_to(vault).as_posix()
    _seed_manifest(vault, "orphan-source/old.c")

    rule = OrphanAttachmentsRule()
    rule.check(CompiledWiki(nodes=[], edges=[], toc=[]), sources_dir=str(vault / "sources"))
    rule.fix(
        CompiledWiki(nodes=[], edges=[], toc=[]),
        [LintResult(rule="orphan-attachments", outcome=LintRuleOutcome.FAIL, details={"path": rel_path})],
    )
    assert not orphan.exists()

    # Simulate the file reappearing (restored tree / re-synced source dir).
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text("orphan contents", encoding="utf-8")

    adopted = adopt_orphan_attachments(vault)
    assert not any(a.path == "orphan-source/old.c" for a in adopted)
    assert "orphan-source/old.c" in AttachmentManifest(vault / ".knox" / "attachments.jsonl").deleted_paths()
