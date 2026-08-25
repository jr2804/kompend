"""Tests for vault-path resolution — fail-fast on non-vault directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from knox.core.paths import VAULT_MARKERS, vault_paths


class TestVaultPathsFailFast:
    def test_raises_on_non_vault_dir(self, tmp_path: Path) -> None:
        """A directory without any vault marker is rejected.

        Silent operation on arbitrary directories masks misconfiguration.
        """
        with pytest.raises(LookupError, match="not a KNOX vault root"):
            vault_paths(tmp_path)

    def test_accepts_vault_with_knox_config(self, tmp_path: Path) -> None:
        knox_dir = tmp_path / ".knox"
        knox_dir.mkdir()
        (knox_dir / "config.toml").write_text("", encoding="utf-8")
        assert vault_paths(tmp_path).base_dir == tmp_path.resolve()

    def test_accepts_project_marker(self, tmp_path: Path) -> None:
        (tmp_path / VAULT_MARKERS[0]).write_text("", encoding="utf-8")
        assert vault_paths(tmp_path).base_dir == tmp_path.resolve()

    def test_require_vault_false_skips_check(self, tmp_path: Path) -> None:
        """Config-introspection paths may opt out explicitly."""
        vp = vault_paths(tmp_path, require_vault=False)
        assert vp.base_dir == tmp_path.resolve()
