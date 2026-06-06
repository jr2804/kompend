"""Tests for KNOX core data models."""

from pathlib import Path

from knox.core.models import Chunk, CompiledWiki, DocumentElement, ElementType, ParsedDocument, WikiNode


def test_element_type_enum() -> None:
    assert ElementType.TEXT.name == "TEXT"
    assert ElementType.TABLE.name == "TABLE"
    assert ElementType.IMAGE.name == "IMAGE"


def test_document_element_creation() -> None:
    elem = DocumentElement(
        type=ElementType.HEADING,
        content="# Introduction",
        page=1,
        reading_order=0,
        heading_level=1,
        metadata={"style": "Title"},
    )
    assert elem.type == ElementType.HEADING
    assert elem.heading_level == 1
    assert elem.metadata["style"] == "Title"


def test_parsed_document_structure() -> None:
    elems = [
        DocumentElement(
            type=ElementType.HEADING,
            content="# Title",
            page=1,
            reading_order=0,
            heading_level=1,
        ),
        DocumentElement(
            type=ElementType.TEXT,
            content="Body text here.",
            page=1,
            reading_order=1,
        ),
    ]
    doc = ParsedDocument(
        source_path=Path("test.pdf"),
        elements=elems,
        title="Test Doc",
        author="Test Author",
        page_count=2,
    )
    assert len(doc.elements) == 2
    assert doc.title == "Test Doc"
    assert doc.author == "Test Author"
    assert doc.elements[0].type == ElementType.HEADING


def test_chunk_creation() -> None:
    chunk = Chunk(
        id="chunk-1",
        text="Sample text",
        element_types=[ElementType.TEXT],
        source_elements=[0, 1],
        heading_context="Section 1",
    )
    assert chunk.id == "chunk-1"
    assert chunk.heading_context == "Section 1"


def test_wiki_node_creation() -> None:
    node = WikiNode(
        id="node-1",
        title="Introduction",
        content="The intro...",
        node_type="section",
    )
    assert node.node_type == "section"
    assert node.title == "Introduction"


def test_compiled_wiki_creation() -> None:
    node = WikiNode(
        id="node-1",
        title="Intro",
        content="...",
        node_type="section",
    )
    wiki = CompiledWiki(
        nodes=[node],
        edges=[],
        toc=[],
    )
    assert len(wiki.nodes) == 1
    assert len(wiki.edges) == 0
