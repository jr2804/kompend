"""Tests for the shared chunk-embedding text composition."""

from __future__ import annotations

from knox.core.text.embedding_text import CONTEXT_SEPARATOR, embed_text


class TestEmbedText:
    def test_no_context_is_a_noop(self) -> None:
        assert embed_text("chunk body") == "chunk body"
        assert embed_text("chunk body", "") == "chunk body"

    def test_context_is_prepended(self) -> None:
        assert embed_text("chunk body", "situating context") == f"situating context{CONTEXT_SEPARATOR}chunk body"
