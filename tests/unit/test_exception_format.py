"""Tests for knox.core.utils.exceptions."""

from __future__ import annotations

from knox.core.utils.exceptions import MAX_CHARS_EXCEPTION, shorten_exception, shorten_string


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
        assert result == "first line second line third"

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


class TestShortenString:
    def test_short_string_unchanged(self) -> None:
        assert shorten_string("hello") == "hello"

    def test_truncates_long_string(self) -> None:
        result = shorten_string("x" * 500)
        assert len(result) <= MAX_CHARS_EXCEPTION
        assert result.endswith("…")

    def test_multiline_collapsed(self) -> None:
        assert shorten_string("a\nb\nc") == "a b c"

    def test_strips_whitespace(self) -> None:
        assert shorten_string("  padded  ") == "padded"

    def test_custom_limit(self) -> None:
        result = shorten_string("a" * 50, limit=10)
        assert len(result) == 10
        assert result.endswith("…")

    def test_empty_string(self) -> None:
        assert shorten_string("") == ""
