"""A figure must survive the whole pipeline, not only the parse.

The stages that carry a figure each live in their own module: the parser
stores the crop, the chunker keeps the reference, the resolve tier captions
it, the compilation attaches it to the page, and the writer copies it out of
the cache. Every one of those steps existed once and each of them lost the
figure at some point — that is why the chain is tested end to end here
instead of one stage at a time.

No model is called: the caption comes from a stub agent, so the test pins
the plumbing, not the model's taste.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

from kompend.chunking import HierarchicalChunker
from kompend.compilation._two_phase_core import extract_images_from_chunks
from kompend.core.config import EmbeddingConfig, Flavor, LLMConfig, OutputConfig
from kompend.core.models import (
    CompiledWiki,
    DocumentElement,
    ElementType,
    MetadataField,
    NodeType,
    ParsedDocument,
    WikiNode,
)
from kompend.core.paths import ATTACHMENTS_DIR_NAME, KOMP_DIR, VaultPaths
from kompend.core.pipeline.config import Compiler, ImageMode, PipelineConfig
from kompend.core.pipeline.orchestrator import build_pipeline
from kompend.core.pipeline.stages._resolve import resolve_chunks
from kompend.output.markdown import MarkdownWriter
from kompend.resolution.vlm_enricher import VLMEnricher


class _StubVlm:
    """A captioning agent that answers without a model."""

    def __init__(self, caption: str = "A block diagram of the two-stage pipeline.") -> None:
        self.caption = caption
        self.contexts: list[str] = []
        self.calls = 0

    def run(self, deps, **_kwargs) -> str:  # noqa: ANN001, ANN003 — stub signature
        self.calls += 1
        self.contexts.append(deps.context)
        return self.caption


class _BrokenVlm:
    """A captioning agent that fails the way a provider fails."""

    def run(self, deps, **_kwargs) -> str:  # noqa: ANN001, ANN003 — stub signature
        msg = "the provider is on fire"
        raise RuntimeError(msg)


def test_a_figure_reaches_the_page_with_its_caption(tmp_path: Path) -> None:
    """Parse → chunk → caption → page → vault, with the image in between."""
    source = tmp_path / "sources" / "report.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.4\n")

    image = VaultPaths(base_dir=tmp_path).images_dir / "report__fig1.png"
    image.parent.mkdir(parents=True)
    png = _png_bytes()
    image.write_bytes(png)

    document = ParsedDocument(
        source_file=source,
        elements=[
            DocumentElement(
                type=ElementType.HEADING,
                content="Architecture",
                heading_level=1,
                source_file=source,
            ),
            DocumentElement(
                type=ElementType.TEXT,
                content="The pipeline has two stages.",
                source_file=source,
            ),
            DocumentElement(
                type=ElementType.IMAGE,
                content=str(image),
                metadata={MetadataField.IMAGE_FILE: str(image)},
                source_file=source,
            ),
        ],
    )

    tree = HierarchicalChunker(max_tokens=512).chunk(document)
    image_chunks = [node.chunk for node in tree.leaves() if node.chunk and ElementType.IMAGE in node.chunk.element_types]
    assert image_chunks, "the chunker dropped the figure"
    # The reference travels with the chunk — that is the channel the whole
    # chain uses.
    assert image_chunks[0].metadata[MetadataField.IMAGE_FILE] == str(image)

    stub = _StubVlm()
    pipeline = build_pipeline(_make_config(tmp_path))
    pipeline.resolver = VLMEnricher(agent=stub)  # ty: ignore[invalid-argument-type]

    resolved = resolve_chunks(pipeline, [tree])

    # The caption came from the model, with the surroundings in the prompt.
    assert stub.calls == 1
    assert "The pipeline has two stages." in stub.contexts[0]

    images = extract_images_from_chunks(resolved)
    assert len(images) == 1
    assert images[0]["path"] == str(image)
    assert images[0]["caption"] == stub.caption

    node_metadata: dict[str, Any] = {
        MetadataField.IMAGES: images,
        MetadataField.PROVENANCE: ["report"],
    }
    wiki = CompiledWiki(
        nodes=[
            WikiNode(
                id="architecture",
                title="Architecture",
                content="The pipeline has two stages.",
                node_type=NodeType.CONCEPT,
                metadata=node_metadata,
            ),
        ],
        edges=[],
        toc=[],
    )

    writer = MarkdownWriter(flavor=Flavor.OBSIDIAN, vault_mode=True, base_dir=tmp_path)
    writer.write(wiki, tmp_path / "wiki")

    pages = [p for p in tmp_path.rglob("architecture.md") if KOMP_DIR not in p.parts]
    assert pages, "the writer produced no page"
    page = pages[0].read_text(encoding="utf-8")
    # The image left the cache: the writer copies it into attachments/ and
    # rewrites the link, because a page may not point into the cache.
    assert KOMP_DIR not in page
    copied = list((tmp_path / ATTACHMENTS_DIR_NAME).rglob("report__fig1.png"))
    assert copied, "the writer did not copy the figure into attachments/"
    assert copied[0].read_bytes() == png
    assert "report__fig1.png" in page
    # The caption is the alt text and the italic line under the image.
    assert "![A block diagram of the two-stage pipeline.]" in page


def test_a_failed_caption_is_not_published_as_a_caption(tmp_path: Path) -> None:
    """A broken figure must not become a page caption."""
    source = tmp_path / "sources" / "report.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.4\n")

    image = VaultPaths(base_dir=tmp_path).images_dir / "report__fig1.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(_png_bytes())

    document = ParsedDocument(
        source_file=source,
        elements=[
            DocumentElement(
                type=ElementType.IMAGE,
                content=str(image),
                metadata={MetadataField.IMAGE_FILE: str(image)},
                source_file=source,
            ),
        ],
    )
    tree = HierarchicalChunker(max_tokens=512).chunk(document)

    pipeline = build_pipeline(_make_config(tmp_path))
    enricher = VLMEnricher(agent=_BrokenVlm())  # ty: ignore[invalid-argument-type]
    pipeline.resolver = enricher

    resolved = resolve_chunks(pipeline, [tree])

    assert extract_images_from_chunks(resolved)[0]["caption"] == ""
    # The failure is recorded, not swallowed.
    assert enricher.failures.failures
    assert enricher.failures.failures[0].kind.value == "other"


def _png_bytes(width: int = 96, height: int = 96) -> bytes:
    """A PNG big enough to pass the enricher's size and dimension filter.

    Built with the standard library: the bytes are noise, so the file is
    well over the byte threshold, and the picture is big enough to have
    pixels. A 1x1 pixel would be filtered out as a decorative spacer
    before the captioning agent ever sees it.
    """
    rows = bytearray()
    seed = 12345
    for _row in range(height):
        rows.append(0)  # filter type 0 (none)
        for _col in range(width):
            seed = (1103515245 * seed + 12345) % (1 << 31)
            rows += bytes(((seed >> 16) & 0xFF, (seed >> 8) & 0xFF, seed & 0xFF))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + _chunk(b"IEND", b"")


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        source_dir=tmp_path,
        output_dir=tmp_path / "wiki",
        llm=LLMConfig(default_model="ollama/llama3.2", vlm_model="ollama/llama3.2-vision"),
        embedding=EmbeddingConfig(model=""),
        output=OutputConfig(),
        compiler=Compiler.GRAPH,
        image_mode=ImageMode.FILE,
    )
