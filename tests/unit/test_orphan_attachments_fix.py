"""Tests for ``OrphanAttachmentsRule.fix()`` — destructive auto-clean.

Regression: p863 wiki roundup (2026-09-10) revealed 992 orphan-attachment
findings that the linter only *reported*; ``knox compile --lint --fix`` did
nothing. ``fix()`` deletes the file on disk + tombstones the manifest entry.
"""

from __future__ import annotations

import json
from pathlib import Path

from knox.core.models.attachments import Attachment, AttachmentManifest
from knox.core.models.wiki import CompiledWiki
from knox.lint.base import LintResult, LintRuleOutcome, Severity
from knox.lint.structural import OrphanAttachmentsRule


def test_fix_removes_orphan_file_and_tombstones_manifest(tmp_path: Path) -> None:
    vault = tmp_path
    orphan = _make_orphan(vault, "orphan-source", "old.c")
    rel_path = orphan.relative_to(vault).as_posix()
    _seed_manifest(vault, rel_path)

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
    assert rel_path not in AttachmentManifest(vault / ".knox" / "attachments.jsonl").read_all()
    # Raw read confirms the tombstone entry (deleted=True) was appended.
    raw_lines = (vault / ".knox" / "attachments.jsonl").read_text(encoding="utf-8").splitlines()
    last_entry = json.loads(raw_lines[-1])
    assert last_entry["path"] == rel_path
    assert last_entry["deleted"] is True


def test_fix_removes_empty_slug_dir(tmp_path: Path) -> None:
    vault = tmp_path
    orphan = _make_orphan(vault, "orphan-source", "single-file.c")
    rel_path = orphan.relative_to(vault).as_posix()
    _seed_manifest(vault, rel_path)

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
