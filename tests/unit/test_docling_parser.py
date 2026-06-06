"""Tests for Docling parser."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knox.core.models import ElementType, ParsedDocument
from knox.preprocessing.parsers.docling_parser import DOCLING_AVAILABLE, DoclingParser


def test_docling_parser_import_guard() -> None:
    """DoclingParser raises ImportError when docling is not available."""
    if DOCLING_AVAILABLE:
        pytest.skip("docling is installed, skipping import guard test")
    with pytest.raises(ImportError):
        DoclingParser()


@pytest.mark.skipif(not DOCLING_AVAILABLE, reason="docling not installed")
def test_docling_parser_mocked() -> None:
    """Test DoclingParser with mocked DocumentConverter."""
    mock_doc = MagicMock()
    mock_doc.name = "TestDoc"
    mock_doc.pages = []

    mock_result = MagicMock()
    mock_result.document = mock_doc

    with patch("knox.preprocessing.parsers.docling_parser.DocumentConverter") as MockConv:
        instance = MockConv.return_value
        instance.convert.return_value = mock_result

        parser = DoclingParser()
        doc = parser.parse(Path("/tmp/test.pdf"))

        assert isinstance(doc, ParsedDocument)
        assert doc.title == "TestDoc"
        assert doc.metadata["parser"] == "docling"
