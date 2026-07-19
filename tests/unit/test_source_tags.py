"""Tests for SourceTags CLI argument type.

Verifies parsing of comma-separated AND groups and multiple --tags OR logic.
Note: SourceTags is a Typer Annotated type alias (list[str] | None), not a
class. Tests verify the _parse_tag_groups helper in query.py instead.
"""

from __future__ import annotations

from knox.cli.cmds.knowledge.query import _parse_tag_groups


class TestParseTagGroups:
    """_parse_tag_groups helper — mimics CLI --tags parsing."""

    def test_empty_list_returns_empty(self) -> None:
        assert _parse_tag_groups([]) == []

    def test_none_returns_empty(self) -> None:
        assert _parse_tag_groups(None) == []

    def test_single_tag(self) -> None:
        result = _parse_tag_groups(["audio"])
        assert result == [["audio"]]

    def test_comma_separated_tags_are_one_group(self) -> None:
        """Comma-separated tags within one value form an AND group."""
        result = _parse_tag_groups(["audio,paper"])
        assert result == [["audio", "paper"]]

    def test_whitespace_is_stripped(self) -> None:
        result = _parse_tag_groups(["  audio  ,  paper  "])
        assert result == [["audio", "paper"]]

    def test_empty_value_skipped(self) -> None:
        """Empty string produces no group."""
        result = _parse_tag_groups([""])
        assert result == []

    def test_empty_value_after_comma_skipped(self) -> None:
        """Trailing comma produces no empty group."""
        result = _parse_tag_groups(["audio,"])
        assert result == [["audio"]]

    def test_multiple_values_are_separate_groups(self) -> None:
        """Multiple --tags values are separate OR groups."""
        result = _parse_tag_groups(["audio", "paper"])
        assert result == [["audio"], ["paper"]]

    def test_complex_combination(self) -> None:
        """Mix of AND-groups and singles."""
        result = _parse_tag_groups(["audio,paper", "video"])
        assert result == [["audio", "paper"], ["video"]]

    def test_tags_containing_spaces_preserved(self) -> None:
        result = _parse_tag_groups(["my tag,other tag"])
        assert result == [["my tag", "other tag"]]

    def test_only_commas_produced_no_groups(self) -> None:
        """A value that is only commas produces no group."""
        result = _parse_tag_groups([",,,"])
        assert result == []
