"""Tests for knox.core.exception_format."""

from __future__ import annotations

import pytest

from knox.core.exception_format import MAX_CHARS_EXCEPTION, shorten_exception


class TestShortenException:
    def test_short_message_unchanged(self) -> None:
        result = shorten_exception(ValueError("oops"))
        assert result == "oops"

    def test_default_limit_matches_constant(self) -> None:
        assert shorten_exception(ValueError("x")) == "x"
        assert MAX_CHARS_EXCEPTION == 200

    def test_truncates_long_message(self) -> None:
        long_msg = "x" * 500
        result = shorten_exception(ValueError(long_msg))
        assert len(result) <= MAX_CHARS_EXCEPTION
        assert result.endswith("…")

    def test_custom_limit(self) -> None:
        result = shorten_exception(ValueError("a" * 50), limit=10)
        assert len(result) == 10
        assert result.endswith("…")

    def test_multiline_collapsed_to_single_line(self) -> None:
        exc = ValueError("first line\nsecond line\nthird")
        result = shorten_exception(exc)
        assert "\n" not in result
        assert "first line second line third" == result

    def test_strips_whitespace(self) -> None:
        result = shorten_exception(ValueError("  padded  "))
        assert result == "padded"

    def test_preserves_exactly_limit_chars_without_ellipsis(self) -> None:
        msg = "a" * 200
        result = shorten_exception(ValueError(msg))
        assert result == msg
        assert "…" not in result

    def test_ellipsis_added_when_one_over_limit(self) -> None:
        msg = "a" * 201
        result = shorten_exception(ValueError(msg))
        assert len(result) == 200
        assert result.endswith("…")

    def test_works_with_non_valueerror(self) -> None:
        result = shorten_exception(RuntimeError("boom"))
        assert result == "boom"

    def test_empty_message(self) -> None:
        result = shorten_exception(ValueError(""))
        assert result == ""

    def test_works_with_chained_exception(self) -> None:
        try:
            try:
                raise ValueError("inner")
            except ValueError as inner:
                raise RuntimeError("outer") from inner
        except RuntimeError as exc:
            result = shorten_exception(exc)
            assert result == "outer"
