"""Tests for KNOX core models."""

from __future__ import annotations

from pathlib import Path

from knox.core.models import Chunk, DocumentElement, ElementType, ParsedDocument


def test_element_type_enum() -> None:
    assert ElementType.TEXT.name == "TEXT"
    assert ElementType.TABLE.name == "TABLE"


def test_document_element_creation() -> None:
    el = DocumentElement(type=ElementType.TEXT, content="hello world")
    assert el.type == ElementType.TEXT
    assert el.content == "hello world"


def test_parsed_document_creation() -> None:
    doc = ParsedDocument(
        source_path=Path("/tmp/test.pdf"),
        elements=[DocumentElement(type=ElementType.TEXT, content="hello")],
    )
    assert doc.source_path == Path("/tmp/test.pdf")
    assert len(doc.elements) == 1


def test_chunk_creation() -> None:
    chunk = Chunk(
        id="c1",
        text="test chunk",
        element_types=[ElementType.TEXT],
        source_elements=[0],
    )
    assert chunk.text == "test chunk"
    assert chunk.element_types == [ElementType.TEXT]
