"""Tests for the prompt-cache rejection fallback."""

from __future__ import annotations

from pydantic_ai.exceptions import ModelHTTPError

from kompend.agents.base import _is_cache_rejection


class TestCacheRejection:
    def test_a_400_naming_the_field_is_a_rejection(self) -> None:
        exc = ModelHTTPError(400, "model", body={"message": "cache_control: extra inputs are not permitted"})
        assert _is_cache_rejection(exc) is True

    def test_an_unknown_field_rejection_matches(self) -> None:
        exc = ModelHTTPError(422, "model", body={"message": "unknown field 'cache'"})
        assert _is_cache_rejection(exc) is True

    def test_a_400_about_something_else_is_not(self) -> None:
        assert _is_cache_rejection(ModelHTTPError(400, "model", body={"message": "invalid tool_choice"})) is False

    def test_a_timeout_is_not(self) -> None:
        assert _is_cache_rejection(ModelHTTPError(504, "model", body={"message": "cache"})) is False
