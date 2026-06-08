"""Tests for external parser support (external.py, _registry.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from knox.core.models import ElementType, ParsedDocument
from knox.preprocessing.parsers._registry import (
    clear_registry,
    get_parser_names,
    get_registered_parsers,
    register_parser,
)
from knox.preprocessing.parsers.external import (
    ExternalParser,
    HTTPParser,
    parse_result_from_json,
    register_external_parsers,
)

# ── Registry tests ──────────────────────────────────────────────────────


class TestRegistry:
    def setup_method(self) -> None:
        clear_registry()

    def test_register_and_get(self) -> None:
        register_parser("TestParser", "built-in", [".pdf", ".docx"])
        parsers = get_registered_parsers()
        assert len(parsers) == 1
        assert parsers[0]["name"] == "TestParser"
        assert parsers[0]["type"] == "built-in"
        assert ".pdf" in parsers[0]["extensions"]

    def test_idempotent_register(self) -> None:
        register_parser("SameName", "built-in", [".pdf"])
        register_parser("SameName", "built-in", [".pdf"])  # duplicate
        assert len(get_registered_parsers()) == 1

    def test_get_names(self) -> None:
        register_parser("A", "built-in", [])
        register_parser("B", "external", [])
        names = get_parser_names()
        assert "A" in names
        assert "B" in names

    def test_clear(self) -> None:
        register_parser("Temp", "built-in", [])
        clear_registry()
        assert get_registered_parsers() == []


# ── ExternalParser tests ────────────────────────────────────────────────


class FakeExternalParser:
    """Simulates an external CLI tool that outputs valid JSON."""

    @staticmethod
    def write_valid_json(path: Path) -> dict:
        return {
            "title": "Test Doc",
            "author": "Test Author",
            "language": "en",
            "page_count": 5,
            "metadata": {},
            "elements": [
                {"type": "TEXT", "content": "Hello world", "page": 1, "reading_order": 0},
                {"type": "HEADING", "content": "Intro", "page": 1, "reading_order": 1, "heading_level": 1},
            ],
        }


class TestExternalParser:
    def test_init(self) -> None:
        parser = ExternalParser(command="myparser", extensions=[".xyz"], name="myparser")
        assert parser._command == "myparser"
        assert parser._exts == (".xyz",)
        assert parser._name == "myparser"

    def test_supported_extensions(self) -> None:
        parser = ExternalParser(command="myparser", extensions=[".xyz", ".abc"])
        assert parser.supported_extensions == (".xyz", ".abc")

    def test_repr(self) -> None:
        parser = ExternalParser(command="mycli", extensions=[".xyz"])
        r = repr(parser)
        assert "ExternalParser" in r
        assert "mycli" in r

    def test_parse_command_not_found(self) -> None:
        parser = ExternalParser(command="nonexistent-command-12345", extensions=[".xyz"])
        with pytest.raises(RuntimeError, match="not found"):
            parser.parse(Path("test.xyz"))


class TestHTTPParser:
    def test_init(self) -> None:
        parser = HTTPParser(endpoint="http://localhost:8080/convert", extensions=[".abc"])
        assert parser._endpoint == "http://localhost:8080/convert"
        assert parser._exts == (".abc",)

    def test_supported_extensions(self) -> None:
        parser = HTTPParser(endpoint="http://localhost:8080/convert", extensions=[".abc", ".def"])
        assert parser.supported_extensions == (".abc", ".def")

    def test_repr(self) -> None:
        parser = HTTPParser(endpoint="http://localhost:8080/convert", extensions=[".abc"])
        r = repr(parser)
        assert "HTTPParser" in r
        assert "localhost" in r


# ── parse_result_from_json ──────────────────────────────────────────────


class TestParseResultFromJson:
    def test_minimal_document(self) -> None:
        data = {
            "title": "Minimal Doc",
            "elements": [],
        }
        doc = parse_result_from_json(data, Path("test.xyz"))
        assert isinstance(doc, ParsedDocument)
        assert doc.title == "Minimal Doc"
        assert doc.elements == []

    def test_with_elements(self) -> None:
        data = {
            "title": "Full Doc",
            "elements": [
                {"type": "TEXT", "content": "Hello"},
                {"type": "HEADING", "content": "Title", "heading_level": 1},
            ],
        }
        doc = parse_result_from_json(data, Path("test.xyz"))
        assert len(doc.elements) == 2
        assert doc.elements[0].type == ElementType.TEXT
        assert doc.elements[1].type == ElementType.HEADING

    def test_invalid_element_type(self) -> None:
        data = {
            "title": "Test",
            "elements": [{"type": "UNKNOWN_TYPE", "content": "x"}],
        }
        doc = parse_result_from_json(data, Path("test.xyz"))
        # Invalid type should fall back to TEXT
        assert doc.elements[0].type == ElementType.TEXT

    def test_metadata_roundtrip(self, tmp_path: Path) -> None:
        data = FakeExternalParser.write_valid_json(tmp_path / "test.xyz")
        doc = parse_result_from_json(data, Path("test.xyz"))
        assert doc.page_count == 5
        assert doc.author == "Test Author"


# ── register_external_parsers ───────────────────────────────────────────


class TestRegisterExternalParsers:
    def test_external_config(self) -> None:
        clear_registry()
        parsers = register_external_parsers(
            external_configs=[{"name": "myparser", "command": "mycli", "extensions": [".xyz"]}],
        )
        assert len(parsers) == 1
        assert isinstance(parsers[0], ExternalParser)
        # Check it's also in the global registry
        registered = get_registered_parsers()
        assert len(registered) == 1

    def test_http_config(self) -> None:
        clear_registry()
        parsers = register_external_parsers(
            http_config={"endpoint": "http://localhost:8080/convert", "extensions": [".abc"]},
        )
        assert len(parsers) == 1
        assert isinstance(parsers[0], HTTPParser)

    def test_both_configs(self) -> None:
        clear_registry()
        parsers = register_external_parsers(
            external_configs=[{"name": "ext", "command": "extcli", "extensions": [".xyz"]}],
            http_config={"endpoint": "http://localhost:8080/convert", "extensions": [".abc"]},
        )
        assert len(parsers) == 2

    def test_no_config(self) -> None:
        parsers = register_external_parsers()
        assert parsers == []
