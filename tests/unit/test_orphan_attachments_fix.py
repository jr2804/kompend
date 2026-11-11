"""Tests for ``OrphanAttachmentsRule.fix()`` — destructive auto-clean.

Regression: p863 wiki roundup (2026-09-10) revealed 992 orphan-attachment
findings that the linter only *reported*; ``kompend compile --lint --fix`` did
nothing. ``fix()`` deletes the file on disk + tombstones the manifest entry.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from kompend.core.models.attachments import Attachment, AttachmentManifest, adopt_orphan_attachments
from kompend.core.models.wiki import CompiledWiki
from kompend.core.pipeline.state import SourceState, TierState, TierStateStore
from kompend.lint.base import LintResult, LintRuleOutcome, Severity
from kompend.lint.structural import OrphanAttachmentsRule


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
        sources_dir=vault / "sources",
    )

    assert len(results) == 1
    assert results[0].fix_applied is True
    assert "Deleted orphan attachment" in results[0].message
    assert not orphan.exists(), "orphan file should be gone"
    # read_all returns active entries; the tombstone is gone from there.
    manifest = AttachmentManifest(vault / ".kompend" / "attachments.jsonl")
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
        sources_dir=vault / "sources",
    )
    assert not (vault / "attachments" / "orphan-source").exists()


def test_fix_noop_when_no_failures(tmp_path: Path) -> None:
    vault = tmp_path
    rule = OrphanAttachmentsRule()
    rule.check(CompiledWiki(nodes=[], edges=[], toc=[]), sources_dir=str(vault / "sources"))

    pass_result = LintResult(rule="orphan-attachments", outcome=LintRuleOutcome.PASS, message="ok")
    skip_result = LintResult(rule="orphan-attachments", outcome=LintRuleOutcome.SKIP, message="skipped")

    out = rule.fix(CompiledWiki(nodes=[], edges=[], toc=[]), [pass_result, skip_result], sources_dir=vault / "sources")
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
    out = rule.fix(CompiledWiki(nodes=[], edges=[], toc=[]), [result], sources_dir=vault / "sources")
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
        sources_dir=vault / "sources",
    )
    assert not orphan.exists()

    # Simulate the file reappearing (restored tree / re-synced source dir).
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text("orphan contents", encoding="utf-8")

    adopted = adopt_orphan_attachments(vault)
    assert not any(a.path == "orphan-source/old.c" for a in adopted)
    assert "orphan-source/old.c" in AttachmentManifest(vault / ".kompend" / "attachments.jsonl").deleted_paths()


def test_fix_survives_an_empty_dir_it_cannot_remove(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """An undeletable empty directory is logged, not reported as removed.

    ``contextlib.suppress(OSError)`` hid this: the run said "directory cleaned"
    while the directory stayed, and the next run said it again.
    """
    vault = tmp_path
    (vault / "sources").mkdir()
    orphan = _make_orphan(vault, "orphan-source", "locked.c")
    rel_path = orphan.relative_to(vault).as_posix()
    slug_dir = orphan.parent
    real_rmdir = Path.rmdir

    def _refuse(self: Path, *args: object, **kwargs: object) -> None:
        if self == slug_dir:
            raise OSError(13, "Permission denied", str(self))
        real_rmdir(self, *args, **kwargs)

    rule = OrphanAttachmentsRule()
    rule.check(CompiledWiki(nodes=[], edges=[], toc=[]), sources_dir=str(vault / "sources"))
    with monkeypatch.context() as m, caplog.at_level(logging.DEBUG, logger="kompend.lint.structural.attachments"):
        m.setattr(Path, "rmdir", _refuse)
        results = rule.fix(
            CompiledWiki(nodes=[], edges=[], toc=[]),
            [LintResult(rule="orphan-attachments", outcome=LintRuleOutcome.FAIL, details={"path": rel_path})],
            sources_dir=vault / "sources",
        )

    assert results[0].fix_applied is True
    assert not orphan.exists()
    assert slug_dir.is_dir(), "the directory could not be removed, so it is still there"
    assert any(str(slug_dir) in record.getMessage() for record in caplog.records)


def test_fix_acts_on_the_directory_the_caller_names(tmp_path: Path) -> None:
    """Two vaults, one check pass each: the fix must act on the one it is told.

    The rules used to remember whatever ``check()`` saw on the instance,
    which is the module-level singleton the runner holds. A fix for vault A
    issued after a scan of vault B therefore deleted vault B's figures — the
    caller could not say which vault it meant. The directory is a parameter
    now, so the second call acts on the second vault.
    """
    first = tmp_path / "first"
    second = tmp_path / "second"
    orphan_first = _make_orphan(first, "orphan-source", "first.c")
    orphan_second = _make_orphan(second, "orphan-source", "second.c")
    _seed_compile_state(first, "first")
    _seed_compile_state(second, "second")

    rule = OrphanAttachmentsRule()
    results_first = rule.check(CompiledWiki(nodes=[], edges=[], toc=[]), sources_dir=str(first / "sources"))
    rule.check(CompiledWiki(nodes=[], edges=[], toc=[]), sources_dir=str(second / "sources"))

    rule.fix(
        CompiledWiki(nodes=[], edges=[], toc=[]),
        [r for r in results_first if r.outcome == LintRuleOutcome.FAIL],
        sources_dir=first / "sources",
    )

    assert not orphan_first.exists()
    assert orphan_second.is_file(), "the second vault was never the subject of this fix"

    rule.fix(
        CompiledWiki(nodes=[], edges=[], toc=[]),
        [
            LintResult(
                rule="orphan-attachments",
                outcome=LintRuleOutcome.FAIL,
                details={"path": orphan_second.relative_to(second).as_posix()},
            )
        ],
        sources_dir=second / "sources",
    )

    assert not orphan_second.exists()


def _seed_compile_state(vault: Path, stem: str) -> None:
    """Seed tier state so the orphan rule's "has a compile run" gate passes."""
    sources = vault / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    (sources / f"{stem}.md").write_text("# note\n", encoding="utf-8")
    store = TierStateStore(vault / ".kompend" / "compile_state.json")
    store.save(TierState(entries={stem: SourceState(stem=stem, source_file=f"sources/{stem}.md", content_hash="h")}))


def _seed_manifest(base_dir: Path, path: str) -> None:
    """Write a minimal manifest so the rule can read + tombstone it."""
    manifest = AttachmentManifest(base_dir / ".kompend" / "attachments.jsonl")
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
