"""Unit tests for Attachment model and AttachmentManifest."""

from __future__ import annotations

from pathlib import Path

from knox.core.models.attachments import (
    IMAGE_EXTENSIONS,
    INGESTIBLE_EXTENSIONS,
    TEXT_EXTENSIONS,
    Attachment,
    AttachmentManifest,
    adopt_orphan_attachments,
)

# ── Attachment model ────────────────────────────────────────────────────


def test_attachment_basic_fields() -> None:
    """Attachment stores source, path, ingestible, embed, deleted."""
    att = Attachment(path="report/photo.jpg", source="report", ingestible=True)
    assert att.source == "report"
    assert att.path == "report/photo.jpg"
    assert att.filename == "photo.jpg"
    assert att.extension == ".jpg"
    assert att.ingestible is True
    assert att.embed is None
    assert att.deleted is False


def test_attachment_to_dict_roundtrip() -> None:
    """to_dict / from_dict roundtrip preserves all fields."""
    att = Attachment(
        path="report/data.csv",
        source="report",
        ingestible=True,
        embed=False,
    )
    d = att.to_dict()
    assert d["path"] == "report/data.csv"
    assert d["source"] == "report"
    assert d["ingestible"] is True
    assert d["embed"] is False
    assert d["deleted"] is False

    restored = Attachment.from_dict(d)
    assert restored.path == att.path
    assert restored.source == att.source
    assert restored.ingestible == att.ingestible
    assert restored.embed == att.embed


def test_attachment_embed_override() -> None:
    """embed=None means inherit, True/False overrides."""
    inherit = Attachment(path="a/b.txt", source="a")
    assert inherit.effective_embed(global_embed_attachments=True) is True
    assert inherit.effective_embed(global_embed_attachments=False) is False

    force = Attachment(path="a/b.txt", source="a", embed=True)
    assert force.effective_embed(global_embed_attachments=False) is True

    suppress = Attachment(path="a/b.txt", source="a", embed=False)
    assert suppress.effective_embed(global_embed_attachments=True) is False


def test_attachment_is_image() -> None:
    """is_image returns True for image extensions."""
    assert Attachment(path="r/photo.jpg", source="r").is_image is True
    assert Attachment(path="r/photo.png", source="r").is_image is True
    assert Attachment(path="r/data.csv", source="r").is_image is False


def test_attachment_extension_case_insensitive() -> None:
    """Extension matching is case-insensitive."""
    assert Attachment(path="r/PHOTO.JPG", source="r").is_image is True
    assert Attachment(path="r/PHOTO.JPG", source="r").extension == ".jpg"


def test_ingestible_extensions_union() -> None:
    """INGESTIBLE_EXTENSIONS = TEXT_EXTENSIONS ∪ IMAGE_EXTENSIONS."""
    assert INGESTIBLE_EXTENSIONS == TEXT_EXTENSIONS | IMAGE_EXTENSIONS
    assert ".jpg" in INGESTIBLE_EXTENSIONS
    assert ".md" in INGESTIBLE_EXTENSIONS
    assert ".pdf" not in INGESTIBLE_EXTENSIONS
    assert ".exe" not in INGESTIBLE_EXTENSIONS


# ── AttachmentManifest ──────────────────────────────────────────────────


def test_manifest_read_empty(tmp_path: Path) -> None:
    """Missing manifest returns empty dict."""
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    assert not manifest.exists()
    assert manifest.read_all() == {}


def test_manifest_append_and_read(tmp_path: Path) -> None:
    """Appended attachments are readable."""
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    att = Attachment(path="report/photo.jpg", source="report", ingestible=True)
    manifest.append(att)

    entries = manifest.read_all()
    assert "report/photo.jpg" in entries
    assert entries["report/photo.jpg"].ingestible is True


def test_manifest_remove_tombstone(tmp_path: Path) -> None:
    """remove() appends a tombstone; read_all() excludes it."""
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    att = Attachment(path="report/photo.jpg", source="report", ingestible=True)
    manifest.append(att)
    assert len(manifest.read_all()) == 1

    assert manifest.remove("report/photo.jpg") is True
    assert manifest.read_all() == {}

    # Removing non-existent returns False.
    assert manifest.remove("nonexistent") is False


def test_manifest_latest_entry_wins(tmp_path: Path) -> None:
    """When the same path is appended twice, the latest wins."""
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    manifest.append(Attachment(path="r/a.txt", source="r", ingestible=False))
    manifest.append(Attachment(path="r/a.txt", source="r", ingestible=True))

    entries = manifest.read_all()
    assert entries["r/a.txt"].ingestible is True


def test_manifest_set_ingestible_updates_flag(tmp_path: Path) -> None:
    """set_ingestible flips the per-attachment flag (append-only update)."""
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    manifest.append(Attachment(path="r/a.txt", source="r", ingestible=True, embed=True))

    assert manifest.set_ingestible("r/a.txt", False) is True
    att = manifest.read_all()["r/a.txt"]
    assert att.ingestible is False
    assert att.embed is True  # unrelated fields preserved

    assert manifest.set_ingestible("r/a.txt", True) is True
    assert manifest.read_all()["r/a.txt"].ingestible is True

    # Unknown path returns False.
    assert manifest.set_ingestible("r/missing.txt", True) is False


def test_manifest_set_ingestible_empty_manifest(tmp_path: Path) -> None:
    """set_ingestible on an empty manifest returns False."""
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    assert manifest.set_ingestible("r/a.txt", True) is False


def test_manifest_read_for_source(tmp_path: Path) -> None:
    """read_for_source filters by source stem."""
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    manifest.append(Attachment(path="r1/a.txt", source="r1"))
    manifest.append(Attachment(path="r2/b.jpg", source="r2"))

    r1 = manifest.read_for_source("r1")
    assert len(r1) == 1
    assert r1[0].source == "r1"


def test_manifest_list_sources(tmp_path: Path) -> None:
    """list_sources returns sorted unique source stems."""
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    manifest.append(Attachment(path="b/x.txt", source="b"))
    manifest.append(Attachment(path="a/y.txt", source="a"))
    manifest.append(Attachment(path="a/z.txt", source="a"))

    assert manifest.list_sources() == ["a", "b"]


def test_manifest_tombstone_then_readd(tmp_path: Path) -> None:
    """A deleted attachment can be re-added."""
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    manifest.append(Attachment(path="r/a.txt", source="r"))
    manifest.remove("r/a.txt")
    manifest.append(Attachment(path="r/a.txt", source="r", ingestible=True))

    entries = manifest.read_all()
    assert "r/a.txt" in entries
    assert entries["r/a.txt"].ingestible is True


def test_manifest_roundtrips_through_store(tmp_path: Path) -> None:
    """Manifest rows round-trip through the vault store."""
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    manifest.append(Attachment(path="r/a.txt", source="r"))
    manifest.append(Attachment(path="r/b.txt", source="r"))

    entries = manifest.read_all()
    assert set(entries) == {"r/a.txt", "r/b.txt"}
    assert entries["r/a.txt"].source == "r"


# ── AttachmentManifest edge cases ───────────────────────────────────────


def test_manifest_absent_store_reads_empty(tmp_path: Path) -> None:
    """A manifest whose store does not exist yet reads as empty."""
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    assert manifest.read_all() == {}
    assert manifest.deleted_paths() == set()
    assert manifest.exists() is False


def test_tombstone_survives_new_manifest_instance(tmp_path: Path) -> None:
    """A removal is durable: a fresh manifest on the same store honours it.

    The tombstone is the only record of an explicit ``knox attachments remove``.
    If it did not survive the storage move, auto-adoption would silently
    re-adopt the removed file.
    """
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    manifest.append(Attachment(path="r/keep.txt", source="r"))
    manifest.append(Attachment(path="r/gone.txt", source="r"))
    assert manifest.remove("r/gone.txt") is True

    reopened = AttachmentManifest(tmp_path / "attachments.jsonl")
    assert set(reopened.read_all()) == {"r/keep.txt"}
    assert reopened.deleted_paths() == {"r/gone.txt"}


def test_adopt_registers_orphaned_files(tmp_path: Path) -> None:
    """Files under attachments/<live-slug>/ without manifest entry get adopted."""
    base = _make_vault(tmp_path)
    slug_dir = base / "attachments" / "report"
    slug_dir.mkdir()
    (slug_dir / "notes.txt").write_text("hello")
    (slug_dir / "diagram.png").write_bytes(b"\x89PNG")

    adopted = adopt_orphan_attachments(base)
    assert {a.path for a in adopted} == {"report/notes.txt", "report/diagram.png"}
    # Text and image extensions are ingestible.
    by_path = {a.path: a for a in adopted}
    assert by_path["report/notes.txt"].ingestible is True
    assert by_path["report/diagram.png"].ingestible is True
    # embed inherits global config.
    assert all(a.embed is None for a in adopted)


def test_adopt_skips_binary_and_unknown_slug(tmp_path: Path) -> None:
    """Binaries are non-ingestible; slugs with no live source are not adopted."""
    base = _make_vault(tmp_path)
    report_dir = base / "attachments" / "report"
    report_dir.mkdir()
    (report_dir / "data.bin").write_bytes(b"\x00\x01")
    ghost = base / "attachments" / "ghost-source"
    ghost.mkdir()
    (ghost / "x.txt").write_text("no matching source")

    adopted = adopt_orphan_attachments(base)
    assert [a.path for a in adopted] == ["report/data.bin"]
    assert adopted[0].ingestible is False


def test_adopt_idempotent_and_respects_tombstone(tmp_path: Path) -> None:
    """Already-registered paths are skipped; tombstoned paths never return."""
    base = _make_vault(tmp_path)
    slug_dir = base / "attachments" / "report"
    slug_dir.mkdir()
    (slug_dir / "notes.txt").write_text("hello")
    (slug_dir / "removed.txt").write_text("gone")

    manifest = AttachmentManifest(base / ".knox" / "attachments.jsonl")
    manifest.append(Attachment(path="report/notes.txt", source="report", ingestible=True))
    manifest.append(Attachment(path="report/removed.txt", source="report"))
    manifest.remove("report/removed.txt")

    adopted = adopt_orphan_attachments(base)
    assert adopted == []

    # Recovery scenario: the store is deleted entirely — tombstone knowledge is
    # gone with it, so both files are re-adopted.
    _delete_store(base)
    adopted = adopt_orphan_attachments(base)
    assert {a.path for a in adopted} == {"report/notes.txt", "report/removed.txt"}


def test_adopt_no_attachments_dir(tmp_path: Path) -> None:
    """Missing attachments dir is a no-op."""
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    assert adopt_orphan_attachments(base) == []


def test_manifest_deleted_paths(tmp_path: Path) -> None:
    """deleted_paths returns tombstoned paths minus later re-registrations."""
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    manifest.append(Attachment(path="r/a.txt", source="r"))
    manifest.remove("r/a.txt")
    assert manifest.deleted_paths() == {"r/a.txt"}

    # Re-registration clears the tombstone.
    manifest.append(Attachment(path="r/a.txt", source="r", ingestible=True))
    assert manifest.deleted_paths() == set()


def test_adopt_ingest_adopted_flag_gates_ingestible(tmp_path: Path) -> None:
    """ingest_adopted=False registers files without making them ingestible."""
    base = _make_vault(tmp_path)
    slug_dir = base / "attachments" / "report"
    slug_dir.mkdir()
    (slug_dir / "notes.txt").write_text("hello")

    adopted = adopt_orphan_attachments(base, ingest_adopted=False)
    assert [a.path for a in adopted] == ["report/notes.txt"]
    assert adopted[0].ingestible is False


# ── Auto-adoption of orphaned attachments ───────────────────────────────


def _make_vault(tmp_path: Path) -> Path:
    """Vault with one source and an attachments dir."""
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / "sources" / "report.pdf").write_bytes(b"%PDF fake")
    (base / "attachments").mkdir()
    return base


def _delete_store(base: Path) -> None:
    """Delete the vault store (and WAL sidecars) to simulate manifest loss."""
    db = base / ".knox" / "knox.db"
    for suffix in ("", "-wal", "-shm"):
        db.with_name(db.name + suffix).unlink(missing_ok=True)


# ── Tombstone-inclusive reads (CLI ``--all``) ───────────────────────────


def test_read_all_includes_deleted_on_request(tmp_path: Path) -> None:
    """``read_all`` excludes tombstones by default; ``include_deleted`` returns them."""
    manifest = AttachmentManifest(tmp_path / "attachments.jsonl")
    manifest.append(Attachment(path="doc/live.txt", source="doc", ingestible=True))
    manifest.append(Attachment(path="doc/gone.txt", source="doc", ingestible=True))
    assert manifest.remove("doc/gone.txt") is True

    active = manifest.read_all()
    assert set(active) == {"doc/live.txt"}
    assert active["doc/live.txt"].deleted is False

    every = manifest.read_all(include_deleted=True)
    assert set(every) == {"doc/live.txt", "doc/gone.txt"}
    assert every["doc/gone.txt"].deleted is True
    assert every["doc/live.txt"].deleted is False
