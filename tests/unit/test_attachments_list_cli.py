"""``knox attachments list`` reads the store, not the legacy JSONL.

Regression: the ``--all`` branch of ``list_attachments`` read
``.knox/attachments.jsonl`` directly. The store migration stopped writing that
file, so ``knox attachments list --all`` raised ``FileNotFoundError``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from knox.cli.app import app
from knox.core.models.attachments import Attachment, AttachmentManifest
from knox.core.paths import VaultPaths

_cli_runner = CliRunner()


def test_list_all_includes_tombstones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = _cli_runner.invoke(app, ["attachments", "list", "--all"])
    assert result.exit_code == 0, result.output
    assert "doc/live.txt" in result.output
    assert "doc/gone.txt" in result.output
    assert "deleted" in result.output


def test_list_without_all_excludes_tombstones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = _cli_runner.invoke(app, ["attachments", "list"])
    assert result.exit_code == 0, result.output
    assert "doc/live.txt" in result.output
    assert "doc/gone.txt" not in result.output


def test_list_all_does_not_open_legacy_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader must not touch ``.knox/attachments.jsonl`` as a file."""
    _seed(tmp_path)
    monkeypatch.chdir(tmp_path)
    legacy = VaultPaths(tmp_path).attachments_manifest_file
    assert not legacy.exists()  # never written by the store

    real_read_text = Path.read_text

    def _guard(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if self == legacy:
            raise AssertionError("legacy attachments.jsonl was read as a file")
        return real_read_text(self, encoding, errors)

    monkeypatch.setattr(Path, "read_text", _guard)
    result = _cli_runner.invoke(app, ["attachments", "list", "--all"])
    assert result.exit_code == 0, result.output
    assert "doc/gone.txt" in result.output


def _seed(base: Path) -> AttachmentManifest:
    """Vault with a live row and a tombstoned row, both in the store."""
    (base / "sources").mkdir(parents=True, exist_ok=True)
    (base / ".knox").mkdir(parents=True, exist_ok=True)
    (base / ".knox" / "config.toml").write_text("", encoding="utf-8")
    manifest = AttachmentManifest(VaultPaths(base).attachments_manifest_file)
    manifest.append(Attachment(path="doc/live.txt", source="doc", ingestible=True))
    manifest.append(Attachment(path="doc/gone.txt", source="doc", ingestible=True))
    assert manifest.remove("doc/gone.txt") is True
    return manifest
