"""Tests for ``knox attachments verify`` and ``knox attachments adopt``.

``verify`` is a read-only pre-compile check (no CompiledWiki required) that
shares the orphan-slug logic with ``OrphanAttachmentsRule``. ``adopt`` runs the
same adoption the compile hook runs, on demand.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from knox.cli.app import app
from knox.core.models.attachments import Attachment, AttachmentManifest
from knox.core.models.wiki import CompiledWiki
from knox.core.paths import VaultPaths
from knox.core.pipeline.state import SourceState, TierState, TierStateStore
from knox.lint.base import LintRuleOutcome
from knox.lint.structural import OrphanAttachmentsRule, orphan_attachment_dirs

_cli_runner = CliRunner()


# ── verify ───────────────────────────────────────────────────────────────


def test_verify_clean_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path)
    _add_source(base)
    att = base / "attachments" / "report" / "keep.txt"
    att.parent.mkdir(parents=True)
    att.write_text("x", encoding="utf-8")
    _manifest(base).append(Attachment(path="report/keep.txt", source="report"))

    monkeypatch.chdir(base)
    result = _cli_runner.invoke(app, ["attachments", "verify"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_verify_orphan_dir_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path)
    _add_source(base)
    ghost = base / "attachments" / "ghost" / "x.txt"
    ghost.parent.mkdir(parents=True)
    ghost.write_text("x", encoding="utf-8")

    monkeypatch.chdir(base)
    result = _cli_runner.invoke(app, ["attachments", "verify"])
    assert result.exit_code == 1, result.output
    assert "ghost" in result.output
    assert "1 orphan attachment(s)" in result.output


def test_verify_missing_manifest_file_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path)
    _add_source(base)
    keep = base / "attachments" / "report" / "keep.txt"
    keep.parent.mkdir(parents=True)
    keep.write_text("x", encoding="utf-8")
    manifest = _manifest(base)
    manifest.append(Attachment(path="report/keep.txt", source="report"))
    manifest.append(Attachment(path="report/missing.txt", source="report"))

    monkeypatch.chdir(base)
    result = _cli_runner.invoke(app, ["attachments", "verify"])
    assert result.exit_code == 1, result.output
    assert "report/missing.txt" in result.output
    assert "1 missing file(s)" in result.output


def test_verify_agrees_with_orphan_rule(tmp_path: Path) -> None:
    """The command must NOT reimplement the slug logic (single source of truth)."""
    base = _vault(tmp_path)
    _add_source(base)
    for slug in ("ghost", "phantom"):
        d = base / "attachments" / slug
        d.mkdir(parents=True)
        (d / "x.txt").write_text("x", encoding="utf-8")
    store = TierStateStore(VaultPaths(base).compile_state_file)
    store.save(TierState(entries={"report": SourceState(stem="report", source_file="sources/report.pdf", content_hash="h")}))

    rule = OrphanAttachmentsRule()
    results = rule.check(CompiledWiki(nodes=[], edges=[], toc=[]), sources_dir=str(base / "sources"))
    rule_paths = {(r.details or {}).get("path") for r in results if r.outcome == LintRuleOutcome.FAIL}

    assert rule_paths == set(orphan_attachment_dirs(base, base / "sources"))
    assert rule_paths == {"attachments/ghost/x.txt", "attachments/phantom/x.txt"}


# ── adopt ────────────────────────────────────────────────────────────────


def test_adopt_registers_unregistered_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default config: adopted files are registered but not ingestible."""
    base = _vault(tmp_path)
    _add_source(base)
    f = base / "attachments" / "report" / "new.txt"
    f.parent.mkdir(parents=True)
    f.write_text("x", encoding="utf-8")

    monkeypatch.chdir(base)
    result = _cli_runner.invoke(app, ["attachments", "adopt"])
    assert result.exit_code == 0, result.output
    assert "Adopted 1 attachment(s)" in result.output

    row = _manifest(base).read_all()["report/new.txt"]
    assert row.source == "report"
    assert row.ingestible is False

    again = _cli_runner.invoke(app, ["attachments", "adopt"])
    assert again.exit_code == 0, again.output
    assert "Nothing to adopt" in again.output


def test_adopt_respects_ingest_adopted_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path)
    _add_source(base)
    (base / ".knox" / "config.toml").write_text("[attachments]\ningest_adopted = true\n", encoding="utf-8")
    f = base / "attachments" / "report" / "new.txt"
    f.parent.mkdir(parents=True)
    f.write_text("x", encoding="utf-8")

    monkeypatch.chdir(base)
    result = _cli_runner.invoke(app, ["attachments", "adopt"])
    assert result.exit_code == 0, result.output
    assert _manifest(base).read_all()["report/new.txt"].ingestible is True


def test_adopt_ignores_tombstoned_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _vault(tmp_path)
    _add_source(base)
    f = base / "attachments" / "report" / "gone.txt"
    f.parent.mkdir(parents=True)
    f.write_text("x", encoding="utf-8")
    manifest = _manifest(base)
    manifest.append(Attachment(path="report/gone.txt", source="report"))
    assert manifest.remove("report/gone.txt") is True

    monkeypatch.chdir(base)
    result = _cli_runner.invoke(app, ["attachments", "adopt"])
    assert result.exit_code == 0, result.output
    assert "Nothing to adopt" in result.output


def _vault(tmp_path: Path) -> Path:
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".knox").mkdir(parents=True)
    (base / ".knox" / "config.toml").write_text("", encoding="utf-8")
    return base


def _add_source(base: Path, name: str = "report.pdf") -> None:
    (base / "sources" / name).write_bytes(b"%PDF fake")


def _manifest(base: Path) -> AttachmentManifest:
    return AttachmentManifest(VaultPaths(base).attachments_manifest_file)
