"""Tests for the lenient JSON salvage used by LLM JSON parsing."""

from __future__ import annotations

from pydantic import BaseModel

from kompend.llm.format import salvage_json
from tests.conftest import as_obj


class TestSalvageValid:
    def test_clean_json(self) -> None:
        assert salvage_json('{"a": 1}') == {"a": 1}

    def test_empty_returns_none(self) -> None:
        assert salvage_json("") is None
        assert salvage_json("   ") is None

    def test_fenced_json(self) -> None:
        assert salvage_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_leading_prose_then_json(self) -> None:
        assert salvage_json('Here you go:\n{"a": 1}') == {"a": 1}


class TestSalvageRawControlChars:
    """Small models emit raw newlines inside JSON string values (invalid JSON)."""

    def test_raw_newline_in_complete_string(self) -> None:
        # raw newline inside the value — must be recovered as \n
        out = salvage_json('{"pages": [{"content": "line one\nline two"}]}')
        assert out == {"pages": [{"content": "line one\nline two"}]}

    def test_raw_newline_with_truncation(self) -> None:
        out = salvage_json('{"pages": [{"title": "X", "content": "line one\nline two')
        assert out is not None
        assert as_obj(out)["pages"][0]["content"].startswith("line one")

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
        assert as_obj(out)["pages"][0]["content"] == "plain text mid wor"

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


class TestSalvageRecursionGuard:
    def test_deeply_nested_does_not_leak_recursionerror(self) -> None:
        """Pathological nesting must not crash the pipeline via RecursionError.

        Models occasionally emit runaway nested brackets/braces; stdlib json's
        recursive-descent parser raises RecursionError on sufficiently deep
        input. salvage_json must absorb it so the call degrades gracefully
        (None -> PermanentError -> skip) instead of propagating up through
        complete() and crashing the whole compile.
        """
        deep = "[" * 5000 + "]" * 5000  # trips json's parser on CPython 3.13
        # Contract: never leak RecursionError. None if json tripped (caught),
        # the structure if json handled it — either is fine, the point is no crash.
        result = salvage_json(deep)
        assert result is None or isinstance(result, list)


class TestSalvageSchemaGuided:
    """Schema-guided repair completes truncated/missing-field JSON toward a model."""

    def test_truncated_completes_toward_schema(self) -> None:
        Out = self._schema()
        # truncated mid-value — schema salvage completes the structure
        out = salvage_json('{"summary": "s", "pages": [{"title": "A", "content": "x', schema=Out)
        assert out == {"summary": "s", "pages": [{"title": "A", "content": "x"}]}

    def test_missing_field_filled_with_default(self) -> None:
        Out = self._schema()
        out = salvage_json('{"summary": "s"}', schema=Out)
        assert out is not None
        assert as_obj(out)["summary"] == "s"
        assert as_obj(out)["pages"] == []  # salvage fills the missing list field

    def test_wrong_field_name_falls_back_to_schemaless(self) -> None:
        Out = self._schema()
        # schema mode raises on wrong field names — must fall back, not crash
        out = salvage_json('{"sumry": "s", "pages": []}', schema=Out)
        assert out == {"sumry": "s", "pages": []}

    def test_correct_json_unchanged_by_schema_mode(self) -> None:
        Out = self._schema()
        out = salvage_json('{"summary": "ok", "pages": [{"title": "A", "content": "c"}]}', schema=Out)
        assert out == {"summary": "ok", "pages": [{"title": "A", "content": "c"}]}

    def test_no_schema_backward_compatible(self) -> None:
        # callers that pass no schema still work exactly as before
        assert salvage_json('{"a": 1}') == {"a": 1}

    @staticmethod
    def _schema() -> type[BaseModel]:
        class Page(BaseModel):
            title: str
            content: str

        class Out(BaseModel):
            summary: str
            pages: list[Page]

        return Out
