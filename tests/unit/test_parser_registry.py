"""Tests for the pluggable ingestion backend registry.

Backends self-register at import time via ``register_backend``; importing the
``knox.preprocessing.parsers`` package (done by the imports below) triggers
registration before any assertion runs. Exactly one backend is registered per
router.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from knox.core.models import ParsedDocument
from knox.core.models.elements import DocumentElement
from knox.core.models.enums import ElementType
from knox.preprocessing.parsers._registry import (
    KNOWN_BACKENDS,
    ParserBuildOptions,
    build_default_router,
    get_all_extensions,
    get_backend,
    get_registered_parsers,
    register_backend,
)
from knox.preprocessing.parsers.base import IMAGE_SCALE_DEFAULT, BaseParser, ParserRouter, scrub_document
from knox.preprocessing.parsers.parsecraft_parser import ParsecraftParser


def test_known_backends_are_the_surviving_set() -> None:
    """The registry ships no duplicate of a parsecraft capability.

    docling, liteparse, pandoc and glm-ocr were removed once parsecraft
    provided them; web followed when parsecraft's trafilatura backend landed
    (pc-axb). marker and mineru stay until knox declares their parsecraft
    extras (pc-u4g / pc-gv8).
    """
    assert set(KNOWN_BACKENDS) == {"marker", "mineru", "parsecraft"}


@pytest.mark.parametrize(
    ("name", "expected_cls_name"),
    [
        ("parsecraft", "ParsecraftParser"),
    ],
)
def test_get_backend_factory_builds_expected_class(name: str, expected_cls_name: str) -> None:
    parser = get_backend(name)(ParserBuildOptions())
    assert type(parser).__name__ == expected_cls_name


def test_get_backend_unknown_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown parser backend"):
        get_backend("does-not-exist")


def test_parser_build_options_defaults() -> None:
    opts = ParserBuildOptions()
    assert opts.images_dir is None
    assert opts.image_mode == "file"
    assert opts.images_scale == IMAGE_SCALE_DEFAULT


@pytest.mark.parametrize(
    ("name", "expected_cls"),
    [
        ("parsecraft", ParsecraftParser),
    ],
)
def test_build_default_router_registers_exactly_the_named_backend(name: str, expected_cls: type) -> None:
    """One backend, no chain: knox keeps no second level of hand-off."""
    router = build_default_router(name=name)
    assert len(router) == 1
    assert isinstance(router._parsers[0], expected_cls)  # noqa: SLF001 — parser identity is the contract


def test_build_default_router_rejects_a_chain() -> None:
    """The comma-separated backend chain is gone; a chain name is just unknown."""
    with pytest.raises(ValueError, match="Unknown parser backend"):
        build_default_router(name="liteparse,docling")


def test_build_default_router_missing_extra_is_loud() -> None:
    """A configured backend that is not installed fails with the install hint."""
    with pytest.raises(ValueError, match="is not installed"):
        build_default_router(name="mineru")


def test_register_backend_populates_metadata() -> None:
    cls = _dummy_cls()
    register_backend("_test_dummy", lambda _opts: cls(), parser_cls=cls)
    assert get_backend("_test_dummy") is not None
    names = {entry["name"] for entry in get_registered_parsers()}
    assert "_test_dummy" in names


def _dummy_cls() -> type[BaseParser]:
    class _Dummy(BaseParser):
        supported_extensions = set()
        figure_enrichment = True

        def parse(self, source_file: Path) -> ParsedDocument:
            raise NotImplementedError

    return _Dummy


def test_figure_enrichment_metadata_matches_class_attr() -> None:
    by_name = {entry["name"]: entry for entry in get_registered_parsers()}
    assert by_name["parsecraft"]["figure_enrichment"] is ParsecraftParser.figure_enrichment


def test_get_all_extensions_follows_the_named_backend() -> None:
    """Discovery is the named backend's suffix set, not the union of every
    backend the registry knows — a vault should only ingest what its parser can
    actually read.
    """
    parsecraft = get_all_extensions(["parsecraft"])
    assert ".pdf" in parsecraft
    assert ".docx" in parsecraft
    # parsecraft reads source files as plain text through its native-text
    # backend, so a vault configured for it can ingest code.
    assert ".c" in parsecraft
    # A backend with no text-family support contributes none of them.
    assert ".c" not in get_all_extensions(["marker"])


def test_scrub_document_replaces_lone_surrogates() -> None:
    doc = scrub_document(_make_doc_with_surrogates())
    assert doc.elements[0].content == "ok \ufffd mid \ufffd end"
    assert doc.elements[0].metadata == {"caption": ["fig \ufffd"]}
    assert doc.elements[1].content == [["a\ufffd", "b"]]
    assert doc.title == "t\ufffd"
    assert doc.author == "a\ufffd"
    assert doc.metadata == {"nested": {"v": "x\ufffd"}}
    # The whole document must now be strictly UTF-8 encodable.
    json.dumps(asdict(doc), ensure_ascii=False, default=str).encode("utf-8")


def test_router_parse_scrubs_parser_output(tmp_path: Path) -> None:
    class FakeParser(BaseParser):
        def can_parse(self, source_file: Path) -> bool:
            return True

        def parse(self, source_file: Path) -> ParsedDocument:
            return _make_doc_with_surrogates()

    router = ParserRouter()
    router.register(FakeParser())
    doc = router.parse(tmp_path / "broken.pdf")
    assert doc.elements[0].content == "ok \ufffd mid \ufffd end"


# ── Surrogate scrubbing (ParserRouter.parse boundary) ─────────────────────


def _make_doc_with_surrogates() -> ParsedDocument:
    return ParsedDocument(
        source_file=Path("broken.pdf"),
        elements=[
            DocumentElement(
                type=ElementType.TEXT,
                content="ok \ud800 mid \udfff end",
                metadata={"caption": ["fig \udc80"]},
            ),
            DocumentElement(type=ElementType.TABLE, content=[["a\ud900", "b"]]),
        ],
        title="t\ud83d",
        author="a\udead",
        metadata={"nested": {"v": "x\udabc"}},
    )
