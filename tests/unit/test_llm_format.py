"""Tests for the lenient JSON salvage used by LLM JSON parsing."""

from __future__ import annotations

from knox.llm.format import salvage_json


class TestSalvageValid:
    def test_clean_json(self) -> None:
        assert salvage_json('{"a": 1}') == {"a": 1}

    def test_empty_returns_none(self) -> None:
        assert salvage_json("") is None
        assert salvage_json("   ") is None

    def test_fenced_json(self) -> None:
        assert salvage_json("```json\n{\"a\": 1}\n```") == {"a": 1}

    def test_leading_prose_then_json(self) -> None:
        assert salvage_json("Here you go:\n{\"a\": 1}") == {"a": 1}


class TestSalvageRawControlChars:
    """Small models emit raw newlines inside JSON string values (invalid JSON)."""

    def test_raw_newline_in_complete_string(self) -> None:
        # raw newline inside the value — must be recovered as \n
        out = salvage_json('{"pages": [{"content": "line one\nline two"}]}')
        assert out == {"pages": [{"content": "line one\nline two"}]}

    def test_raw_newline_with_truncation(self) -> None:
        out = salvage_json('{"pages": [{"title": "X", "content": "line one\nline two')
        assert out is not None
        assert out["pages"][0]["content"].startswith("line one")

    def test_structural_whitespace_preserved(self) -> None:
        # newline BETWEEN tokens is valid structural whitespace, not touched
        out = salvage_json('{\n"pages": [\n]\n}')
        assert out == {"pages": []}

    def test_tab_in_string(self) -> None:
        out = salvage_json('{"a": "col1\tcol2"}')
        assert out == {"a": "col1\tcol2"}


class TestSalvageTruncation:
    def test_clean_truncation_mid_word(self) -> None:
        out = salvage_json('{"pages": [{"title": "X", "content": "plain text mid wor')
        assert out is not None
        assert out["pages"][0]["content"] == "plain text mid wor"

    def test_trailing_backslash_truncation(self) -> None:
        # truncated mid-escape: trailing backslash must not break the close
        out = salvage_json('{"pages": [{"title": "X", "content": "text \\')
        assert out is not None

    def test_trailing_comma_closed(self) -> None:
        out = salvage_json('{"a": 1, "b": 2,')
        assert out == {"a": 1, "b": 2}

    def test_nested_brackets_closed(self) -> None:
        out = salvage_json('{"pages": [{"title": "X"')
        assert out == {"pages": [{"title": "X"}]}

    def test_unparseable_returns_none(self) -> None:
        assert salvage_json("just prose, no json at all here") is None
