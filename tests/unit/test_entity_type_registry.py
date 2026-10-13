"""Tests for ``EntityTypeRegistry`` — stable per-concept entity-kind registry.

Regression: p863 wiki roundup (2026-09-10) showed entity folder assignments
flickered between runs because the model returned different ``type`` values
for the same concept on different compilations. The registry persists the
first-seen kind and uses it on every subsequent compile.
"""

from __future__ import annotations

from pathlib import Path

from kompend.core.pipeline.entity_types import EntityTypeRegistry
from kompend.core.store import vault_db_path


def test_get_returns_none_for_missing_key(tmp_path: Path) -> None:
    reg = EntityTypeRegistry(tmp_path / "entity_types.json")
    assert reg.get("polqa") is None


def test_record_and_get(tmp_path: Path) -> None:
    reg = EntityTypeRegistry(tmp_path / "entity_types.json")
    assert reg.record("polqa", "standard") is False  # first write, not an override
    assert reg.get("polqa") == "standard"


def test_record_returns_true_on_override(tmp_path: Path) -> None:
    """When a concept's type changes, ``record`` returns True (override)."""
    reg = EntityTypeRegistry(tmp_path / "entity_types.json")
    reg.record("polqa", "standard")
    assert reg.record("polqa", "standard-model") is True  # override detected


def test_record_ignores_empty_inputs(tmp_path: Path) -> None:
    reg = EntityTypeRegistry(tmp_path / "entity_types.json")
    assert reg.record("", "standard") is False
    assert reg.record("polqa", "") is False
    assert reg.get("polqa") is None


def test_record_same_value_returns_false(tmp_path: Path) -> None:
    reg = EntityTypeRegistry(tmp_path / "entity_types.json")
    reg.record("polqa", "standard")
    assert reg.record("polqa", "standard") is False


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "entity_types.json"
    reg = EntityTypeRegistry(path)
    reg.record("polqa", "standard")
    reg.record("itu-t", "organization")
    reg.save()

    reg2 = EntityTypeRegistry(path)
    assert reg2.all() == {"polqa": "standard", "itu-t": "organization"}
    assert reg2.get("polqa") == "standard"
    assert reg2.get("itu-t") == "organization"
    assert len(reg2) == 2


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    reg = EntityTypeRegistry(tmp_path / "does-not-exist.json")
    assert reg.all() == {}
    assert len(reg) == 0


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "subdir" / "entity_types.json"
    reg = EntityTypeRegistry(path)
    reg.record("polqa", "standard")
    reg.save()
    assert vault_db_path(path).exists()


def test_save_is_durable(tmp_path: Path) -> None:
    """A saved registry is visible to a fresh instance (committed, not cached)."""
    path = tmp_path / "entity_types.json"
    reg = EntityTypeRegistry(path)
    reg.record("polqa", "standard")
    reg.save()
    assert EntityTypeRegistry(path).get("polqa") == "standard"


def test_all_returns_copy(tmp_path: Path) -> None:
    """Mutating the returned dict does not affect internal state."""
    reg = EntityTypeRegistry(tmp_path / "entity_types.json")
    reg.record("polqa", "standard")
    snapshot = reg.all()
    snapshot["polqa"] = "mutated"
    assert reg.get("polqa") == "standard"
