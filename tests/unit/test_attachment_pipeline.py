"""Unit tests for pipeline attachment ingestion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kompend.core.config import AttachmentMode
from kompend.core.models import CompiledWiki, DocumentElement, ElementType, MetadataField, NodeType, ParsedDocument, WikiNode
from kompend.core.models.attachments import Attachment, AttachmentManifest
from kompend.core.paths import CACHE_DIR, VaultPaths
from kompend.core.pipeline.attachments import ingest_attachment, iter_ingestible_attachments, stamp_attachment_mentions
from kompend.core.pipeline.stages.parse import _attachment_stub_nodes, _process_attachments
from kompend.core.reporting.ledger import StageLedger
from kompend.core.store import vault_db_path


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
    """Text attachments are parsed via the parser."""
    att = Attachment(path="r/notes.md", source="r", ingestible=True)
    att_dir = vault.attachments_dir / "r"
    att_dir.mkdir(parents=True)
    att_file = att_dir / "notes.md"
    att_file.write_text("# Title\n\nSome content.")

    manifest = AttachmentManifest(vault.attachments_manifest_file)
    manifest.append(att)

    mock_parser = MagicMock()
    expected_doc = ParsedDocument(
        source_file=att_file,
        title="notes",
        elements=[DocumentElement(type=ElementType.TEXT, content="Some content.")],
    )
    mock_parser.parse.return_value = expected_doc

    result = ingest_attachment(vault, att, mock_parser)
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

    mock_parser = MagicMock()

    result = ingest_attachment(vault, att, mock_parser, vlm_captioner=None)
    assert result is None


def test_ingest_image_attachment_with_vlm(vault: VaultPaths) -> None:
    """Image attachment with VLM captioner produces a synthetic doc."""
    att = Attachment(path="r/photo.jpg", source="r", ingestible=True)
    att_dir = vault.attachments_dir / "r"
    att_dir.mkdir(parents=True)
    (att_dir / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0")

    mock_parser = MagicMock()
    mock_captioner = MagicMock()
    mock_captioner.run.return_value = "A landscape photo."

    result = ingest_attachment(vault, att, mock_parser, vlm_captioner=mock_captioner)
    assert result is not None
    assert result.metadata.get("asset_type") == "image_caption"
    content = result.elements[0].content
    assert isinstance(content, str)
    assert "landscape" in content.lower()


def test_process_attachments_gate_off_skips_all(vault: VaultPaths) -> None:
    """attachments.mode=off: nothing is ingested even when entries are ingestible."""
    manifest = AttachmentManifest(vault.attachments_manifest_file)
    manifest.append(Attachment(path="r/a.txt", source="r", ingestible=True))
    (vault.attachments_dir / "r").mkdir(parents=True)
    (vault.attachments_dir / "r" / "a.txt").write_text("a")

    pipeline = MagicMock()
    pipeline.base_directory = str(vault.base_dir)
    pipeline.config.attachments_mode = AttachmentMode.OFF

    assert _process_attachments(pipeline, StageLedger.open(vault_db_path(vault.base_dir / CACHE_DIR))) == []
    pipeline.parser.parse.assert_not_called()


def test_attachment_stub_nodes_in_index_only_mode(vault: VaultPaths) -> None:
    """index-only: an ingestible text attachment becomes one stub page, not a tree."""
    manifest = AttachmentManifest(vault.attachments_manifest_file)
    manifest.append(Attachment(path="r/a.txt", source="r", ingestible=True))
    (vault.attachments_dir / "r").mkdir(parents=True)
    (vault.attachments_dir / "r" / "a.txt").write_text("annex content")

    pipeline = MagicMock()
    pipeline.base_directory = str(vault.base_dir)
    pipeline.config.attachments_mode = AttachmentMode.INDEX_ONLY
    pipeline.config.llm.default_model = None
    pipeline.config.llm.vlm_model = None
    pipeline.config.llm.request_ceiling_s = 120.0
    pipeline.parser = MagicMock()
    pipeline.parser.parse.return_value = ParsedDocument(
        source_file=vault.attachments_dir / "r" / "a.txt",
        elements=[DocumentElement(type=ElementType.TEXT, content="annex content")],
        title="a.txt",
    )

    stubs = _attachment_stub_nodes(pipeline, StageLedger.open(vault_db_path(vault.base_dir / CACHE_DIR)))

    assert len(stubs) == 1
    stub = stubs[0]
    assert isinstance(stub, WikiNode)
    assert stub.node_type is NodeType.PAGE
    assert stub.title == "a.txt"
    assert "annex content" in stub.content
    assert stub.metadata["source_asset"] is True
    assert stub.metadata[MetadataField.ASSET_STUB] is True
    assert stub.metadata["parent_source"] == "r"


def test_a_marginal_figure_is_indexed_in_its_parent_but_gets_no_stub(vault: VaultPaths) -> None:
    """index-only + a ``marginal`` verdict: no page of its own.

    The gate already answered "worth a corpus entry?"; a page per stock
    photo is exactly the pollution the stub quarantine was meant to avoid,
    so the verdict is honoured instead of being asked again.
    """
    manifest = AttachmentManifest(vault.attachments_manifest_file)
    manifest.append(Attachment(path="r/stock.jpg", source="r", ingestible=True))
    (vault.attachments_dir / "r").mkdir(parents=True)
    (vault.attachments_dir / "r" / "stock.jpg").write_bytes(b"JPEG mock data")

    pipeline = MagicMock()
    pipeline.base_directory = str(vault.base_dir)
    pipeline.config.attachments_mode = AttachmentMode.INDEX_ONLY
    pipeline.config.llm.default_model = "test/model"
    pipeline.config.llm.vlm_model = "test/vlm"
    pipeline.config.llm.request_ceiling_s = 120.0
    pipeline.config.decision = MagicMock()
    pipeline.parser = MagicMock()
    pipeline.parser.parse.return_value = ParsedDocument(
        source_file=vault.attachments_dir / "r" / "stock.jpg",
        elements=[DocumentElement(type=ElementType.TEXT, content="A generic office photo")],
        title="stock.jpg",
    )

    captioned = ParsedDocument(
        source_file=vault.attachments_dir / "r" / "stock.jpg",
        elements=[DocumentElement(type=ElementType.TEXT, content="A generic office photo")],
        title="stock.jpg",
        metadata={MetadataField.FIGURE_VERDICT: "marginal"},
    )
    ledger = StageLedger.open(vault_db_path(vault.base_dir / CACHE_DIR))
    with (
        patch("kompend.core.pipeline.stages.parse.VlmAgent"),
        patch("kompend.core.pipeline.stages.parse.build_figure_gate"),
        patch("kompend.core.pipeline.stages.parse.ingest_attachment", return_value=captioned),
    ):
        stubs = _attachment_stub_nodes(pipeline, ledger)

    assert stubs == []


def test_attachment_stub_nodes_skip_binaries(vault: VaultPaths) -> None:
    """Binary attachments have no honest text form — no stub, no error."""
    manifest = AttachmentManifest(vault.attachments_manifest_file)
    manifest.append(Attachment(path="r/binary.bin", source="r", ingestible=True))
    (vault.attachments_dir / "r").mkdir(parents=True)
    (vault.attachments_dir / "r" / "binary.bin").write_bytes(b"\x00\x01\x02")

    pipeline = MagicMock()
    pipeline.base_directory = str(vault.base_dir)
    pipeline.config.attachments_mode = AttachmentMode.INDEX_ONLY
    pipeline.config.llm.default_model = None
    pipeline.config.llm.vlm_model = None
    pipeline.config.llm.request_ceiling_s = 120.0
    pipeline.parser = MagicMock()

    stubs = _attachment_stub_nodes(pipeline, StageLedger.open(vault_db_path(vault.base_dir / CACHE_DIR)))

    assert stubs == []


def test_stamp_attachment_mentions_by_provenance_stem(vault: VaultPaths) -> None:
    """Nodes whose provenance names a source with attachments get the mention."""
    manifest = AttachmentManifest(vault.attachments_manifest_file)
    manifest.append(Attachment(path="r/a.txt", source="r", ingestible=False))
    manifest.append(Attachment(path="r/pic.png", source="r", ingestible=False))

    attached = WikiNode(id="p1", title="Report", content="body", node_type=NodeType.PAGE, metadata={"provenance": ["sources/r.pdf"]})
    other = WikiNode(id="p2", title="Other", content="body", node_type=NodeType.PAGE, metadata={"provenance": ["sources/other.pdf"]})

    wiki = CompiledWiki(nodes=[attached, other], edges=[], toc=[])
    stamped = stamp_attachment_mentions(vault.base_dir, wiki)

    assert stamped == 1
    assert attached.metadata["attachments"] == ["attachments/r/a.txt", "attachments/r/pic.png"]
    assert "attachments" not in other.metadata


def test_process_attachments_gate_on_uses_manifest(vault: VaultPaths) -> None:
    """In full-ingest mode, only manifest-ingestible attachments are processed."""
    manifest = AttachmentManifest(vault.attachments_manifest_file)
    manifest.append(Attachment(path="r/a.txt", source="r", ingestible=True))
    (vault.attachments_dir / "r").mkdir(parents=True)
    (vault.attachments_dir / "r" / "a.txt").write_text("a")

    pipeline = MagicMock()
    pipeline.base_directory = str(vault.base_dir)
    pipeline.config.attachments_mode = AttachmentMode.FULL_INGEST
    pipeline.config.llm.default_model = None
    pipeline.config.llm.vlm_model = None
    pipeline.config.llm.request_ceiling_s = 120.0
    pipeline.parser = MagicMock()
    pipeline.chunker = None

    # ingest_attachment returns None for a text file without a parser pass
    # result; the loop must still run (emit called) rather than early-return.
    trees = _process_attachments(pipeline, StageLedger.open(vault_db_path(vault.base_dir / CACHE_DIR)))
    assert trees == []
    pipeline.emit.assert_called()


def test_ingest_unsupported_type(vault: VaultPaths) -> None:
    """Binary files return None."""
    att = Attachment(path="r/binary.exe", source="r", ingestible=True)
    att_dir = vault.attachments_dir / "r"
    att_dir.mkdir(parents=True)
    (att_dir / "binary.exe").write_bytes(b"\x00\x01")

    mock_parser = MagicMock()

    result = ingest_attachment(vault, att, mock_parser)
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
    mock_parser = MagicMock()
    mock_parser.parse.return_value = expected_doc

    result = ingest_attachment(vault, att, mock_parser)
    assert result is not None
    mock_parser.parse.assert_called_once()


@pytest.fixture
def vault(tmp_path: Path) -> VaultPaths:
    """VaultPaths with sources/ and attachments/ directories."""
    (tmp_path / "sources").mkdir()
    (tmp_path / "attachments").mkdir()
    (tmp_path / ".kompend").mkdir()
    return VaultPaths(tmp_path)
