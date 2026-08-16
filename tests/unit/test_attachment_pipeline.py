"""Unit tests for pipeline attachment ingestion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from knox.core.models import DocumentElement, ElementType, ParsedDocument
from knox.core.models.attachments import Attachment, AttachmentManifest
from knox.core.paths import VaultPaths
from knox.core.pipeline.attachments import ingest_attachment, iter_ingestible_attachments


def test_iter_ingestible_attachments_empty_manifest(vault: VaultPaths) -> None:
    """No manifest → empty list."""
    assert iter_ingestible_attachments(vault) == []


def test_iter_ingestible_attachments_filters_non_ingestible(vault: VaultPaths) -> None:
    """Only ingestible attachments are returned."""
    manifest = AttachmentManifest(vault.attachments_manifest_file)
    manifest.append(Attachment(path="r/a.txt", source="r", ingestible=False))
    manifest.append(Attachment(path="r/b.txt", source="r", ingestible=True))

    # Create the files on disk
    (vault.attachments_dir / "r").mkdir(parents=True)
    (vault.attachments_dir / "r" / "a.txt").write_text("a")
    (vault.attachments_dir / "r" / "b.txt").write_text("b")

    result = iter_ingestible_attachments(vault)
    assert len(result) == 1
    assert result[0].path == "r/b.txt"


def test_iter_ingestible_attachments_skips_missing(vault: VaultPaths) -> None:
    """Attachments missing on disk are skipped."""
    manifest = AttachmentManifest(vault.attachments_manifest_file)
    manifest.append(Attachment(path="r/ghost.txt", source="r", ingestible=True))

    assert iter_ingestible_attachments(vault) == []


def test_ingest_text_attachment(vault: VaultPaths) -> None:
    """Text attachments are parsed via the router."""
    att = Attachment(path="r/notes.md", source="r", ingestible=True)
    att_dir = vault.attachments_dir / "r"
    att_dir.mkdir(parents=True)
    att_file = att_dir / "notes.md"
    att_file.write_text("# Title\n\nSome content.")

    manifest = AttachmentManifest(vault.attachments_manifest_file)
    manifest.append(att)

    mock_router = MagicMock()
    expected_doc = ParsedDocument(
        source_file=att_file,
        title="notes",
        elements=[DocumentElement(type=ElementType.TEXT, content="Some content.")],
    )
    mock_router.parse.return_value = expected_doc

    result = ingest_attachment(vault, att, mock_router)
    assert result is not None
    assert result.metadata.get("source_asset") is True
    assert result.metadata.get("parent_source") == "r"
    assert result.metadata.get("asset_file") == "r/notes.md"

    # Element metadata should also be stamped.
    assert result.elements[0].metadata.get("source_asset") is True


def test_ingest_image_attachment_no_vlm(vault: VaultPaths) -> None:
    """Image attachment without VLM captioner returns None."""
    att = Attachment(path="r/photo.jpg", source="r", ingestible=True)
    att_dir = vault.attachments_dir / "r"
    att_dir.mkdir(parents=True)
    (att_dir / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0")  # JPEG header

    mock_router = MagicMock()

    result = ingest_attachment(vault, att, mock_router, vlm_captioner=None)
    assert result is None


def test_ingest_image_attachment_with_vlm(vault: VaultPaths) -> None:
    """Image attachment with VLM captioner produces a synthetic doc."""
    att = Attachment(path="r/photo.jpg", source="r", ingestible=True)
    att_dir = vault.attachments_dir / "r"
    att_dir.mkdir(parents=True)
    (att_dir / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0")

    mock_router = MagicMock()
    mock_captioner = MagicMock()
    mock_captioner.run.return_value = "A landscape photo."

    result = ingest_attachment(vault, att, mock_router, vlm_captioner=mock_captioner)
    assert result is not None
    assert result.metadata.get("asset_type") == "image_caption"
    content = result.elements[0].content
    assert isinstance(content, str)
    assert "landscape" in content.lower()


def test_ingest_unsupported_type(vault: VaultPaths) -> None:
    """Binary files return None."""
    att = Attachment(path="r/binary.exe", source="r", ingestible=True)
    att_dir = vault.attachments_dir / "r"
    att_dir.mkdir(parents=True)
    (att_dir / "binary.exe").write_bytes(b"\x00\x01")

    mock_router = MagicMock()

    result = ingest_attachment(vault, att, mock_router)
    assert result is None


def test_ingest_unknown_text_extension(vault: VaultPaths) -> None:
    """Non-binary files with arbitrary extensions (.c/.h/unknown) are ingested.

    Regression: previously a hardcoded allowlist in ingest_attachment() skipped
    source-code assets (.c/.cpp/.h) even though the CLI gate accepted them.
    """
    att = Attachment(path="r/module.c", source="r", ingestible=True)
    att_dir = vault.attachments_dir / "r"
    att_dir.mkdir(parents=True)
    (att_dir / "module.c").write_text("int main(void) { return 0; }\n")

    expected_doc = ParsedDocument(
        source_file=att_dir / "module.c",
        title="module.c",
        elements=[DocumentElement(type=ElementType.TEXT, content="int main(void) { return 0; }")],
    )
    mock_router = MagicMock()
    mock_router.parse.return_value = expected_doc

    result = ingest_attachment(vault, att, mock_router)
    assert result is not None
    mock_router.parse.assert_called_once()


@pytest.fixture
def vault(tmp_path: Path) -> VaultPaths:
    """VaultPaths with sources/ and attachments/ directories."""
    (tmp_path / "sources").mkdir()
    (tmp_path / "attachments").mkdir()
    (tmp_path / ".knox").mkdir()
    return VaultPaths(tmp_path)
