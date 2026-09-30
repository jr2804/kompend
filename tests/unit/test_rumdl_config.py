"""Tests for ``resolve_rumdl_config`` — vault-side config lifecycle.

Materializes a fresh ``.knox/rumdl.toml`` from the shipped default on first
use; refreshes stale copies (without a current hash marker) when the
shipped default changes; preserves user edits below the ``# knox-managed``
marker.
"""

from __future__ import annotations

from pathlib import Path

from knox.core.config.rumdl import resolve_rumdl_config
from knox.core.paths import RUMDL_CONFIG_FILE


def test_first_use_materializes_shipped_default(tmp_path: Path) -> None:
    """No on-disk copy → ship the default with the managed marker."""
    _seed_vault(tmp_path)
    p = resolve_rumdl_config(tmp_path)
    assert p is not None
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "# knox-managed:" in text
    assert "MD025" in text  # the shipped default disables MD025 via front-matter-title


def test_idempotent_when_hash_matches(tmp_path: Path) -> None:
    """Repeated calls don't rewrite when the marker hash matches shipped."""
    _seed_vault(tmp_path)
    p1 = resolve_rumdl_config(tmp_path)
    p2 = resolve_rumdl_config(tmp_path)
    assert p1 is not None
    assert p2 is not None
    content1 = p1.read_bytes()
    content2 = p2.read_bytes()
    assert content1 == content2


def test_stale_file_is_refreshed(tmp_path: Path) -> None:
    """An on-disk copy WITHOUT a current-hash marker is refreshed."""
    _seed_vault(tmp_path)
    stale = tmp_path / RUMDL_CONFIG_FILE
    stale.write_text("# old partial copy without marker\n", encoding="utf-8")
    p = resolve_rumdl_config(tmp_path)
    assert p is not None
    text = p.read_text(encoding="utf-8")
    assert "MD025" in text  # refreshed config has the disable rule
    assert "# knox-managed:" in text


def test_user_edits_below_marker_are_preserved(tmp_path: Path) -> None:
    """User-added content below the marker survives a stale refresh."""
    _seed_vault(tmp_path)
    stale = tmp_path / RUMDL_CONFIG_FILE
    user_section = "[MD041]\nlevel = 2\n# my custom rule\n"
    stale.write_text(
        "# shipped stuff\n# knox-managed: 5ff9d402824e9008\n" + user_section,
        encoding="utf-8",
    )
    p = resolve_rumdl_config(tmp_path)
    assert p is not None
    text = p.read_text(encoding="utf-8")
    assert "[MD041]" in text
    assert "my custom rule" in text


def _seed_vault(vault: Path) -> None:
    (vault / ".knox").mkdir(parents=True, exist_ok=True)


def test_explicit_config_path_wins(tmp_path: Path) -> None:
    """A configured path bypasses the default materialization."""
    custom = tmp_path / "my-rumdl.toml"
    custom.write_text("# my config\n", encoding="utf-8")
    p = resolve_rumdl_config(tmp_path, configured=custom)
    assert p == custom


def test_relative_config_path_resolves_against_base_dir(tmp_path: Path) -> None:
    custom_rel = "subdir/r.toml"
    (tmp_path / "subdir").mkdir()
    (tmp_path / custom_rel).write_text("# subdir\n", encoding="utf-8")
    p = resolve_rumdl_config(tmp_path, configured=custom_rel)
    assert p == (tmp_path / custom_rel).resolve()
