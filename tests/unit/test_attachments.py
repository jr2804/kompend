"""Unit tests for Attachment model and AttachmentManifest."""

from __future__ import annotations

import json
from pathlib import Path

from knox.core.models.attachments import (
    IMAGE_EXTENSIONS,
    INGESTIBLE_EXTENSIONS,
    TEXT_EXTENSIONS,
    Attachment,
    AttachmentManifest,
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


def test_manifest_jsonl_format(tmp_path: Path) -> None:
    """Manifest is valid JSONL (one JSON object per line)."""
    f = tmp_path / "attachments.jsonl"
    manifest = AttachmentManifest(f)
    manifest.append(Attachment(path="r/a.txt", source="r"))
    manifest.append(Attachment(path="r/b.txt", source="r"))

    lines = f.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "path" in obj
        assert "source" in obj


# ── AttachmentManifest edge cases ───────────────────────────────────────


def test_manifest_corrupt_line_skipped(tmp_path: Path) -> None:
    """Corrupt JSON lines are silently skipped."""
    f = tmp_path / "attachments.jsonl"
    f.write_text(
        json.dumps({"path": "r/a.txt", "source": "r", "ingestible": True}) + "\nTHIS IS NOT JSON\n" + json.dumps({"path": "r/b.txt", "source": "r"}) + "\n",
    )
    manifest = AttachmentManifest(f)
    entries = manifest.read_all()
    assert len(entries) == 2
    assert "r/a.txt" in entries
    assert "r/b.txt" in entries


def test_manifest_blank_lines_skipped(tmp_path: Path) -> None:
    """Blank lines in the manifest are ignored."""
    f = tmp_path / "attachments.jsonl"
    f.write_text(
        "\n" + json.dumps({"path": "r/a.txt", "source": "r"}) + "\n\n",
    )
    manifest = AttachmentManifest(f)
    assert len(manifest.read_all()) == 1
