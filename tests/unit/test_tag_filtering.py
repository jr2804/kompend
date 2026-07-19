"""Tests for tag filtering logic in the query command.

Verifies AND/OR boolean logic for tag-based filtering of search results.
"""

from __future__ import annotations

from knox.cli.cmds.knowledge.query import _matches_tag_filter


class TestMatchesTagFilter:
    """Boolean AND/OR tag matching logic."""

    def test_no_filter_returns_all(self) -> None:
        """Empty tag_groups matches all nodes."""
        assert _matches_tag_filter({"audio"}, []) is True
        assert _matches_tag_filter(set(), []) is True

    def test_single_tag_exact_match(self) -> None:
        """A single tag group matches nodes with that tag."""
        assert _matches_tag_filter({"audio"}, [["audio"]]) is True
        assert _matches_tag_filter({"paper"}, [["audio"]]) is False

    def test_and_group_requires_all_tags(self) -> None:
        """Comma-separated tags within one group require ALL tags (AND)."""
        assert _matches_tag_filter({"audio", "paper"}, [["audio", "paper"]]) is True
        assert _matches_tag_filter({"audio"}, [["audio", "paper"]]) is False
        assert _matches_tag_filter({"paper"}, [["audio", "paper"]]) is False

    def test_or_between_groups(self) -> None:
        """Multiple tag groups are ORed — match any group."""
        assert _matches_tag_filter({"audio"}, [["audio"], ["paper"]]) is True
        assert _matches_tag_filter({"paper"}, [["audio"], ["paper"]]) is True
        assert _matches_tag_filter({"video"}, [["audio"], ["paper"]]) is False

    def test_complex_and_or_combination(self) -> None:
        """(a AND b) OR c — one AND-group plus one single."""
        assert _matches_tag_filter({"audio", "paper"}, [["audio", "paper"], ["video"]]) is True
        assert _matches_tag_filter({"video"}, [["audio", "paper"], ["video"]]) is True
        assert _matches_tag_filter({"audio"}, [["audio", "paper"], ["video"]]) is False

    def test_empty_tag_group_matches_nothing(self) -> None:
        """An empty tag group (from empty string input) matches nothing."""
        assert _matches_tag_filter({"audio"}, [[]]) is False

    def test_node_without_tags_does_not_match(self) -> None:
        """Empty node_tags does not match any non-empty group."""
        assert _matches_tag_filter(set(), [["audio"]]) is False

    def test_case_sensitive_matching(self) -> None:
        """Tag matching is case-sensitive."""
        assert _matches_tag_filter({"Audio"}, [["audio"]]) is False
        assert _matches_tag_filter({"audio"}, [["Audio"]]) is False

    def test_multiple_tags_on_node(self) -> None:
        """A node with many tags matches if any group is satisfied."""
        node_tags = {"audio", "paper", "video", "image"}
        assert _matches_tag_filter(node_tags, [["video", "image"]]) is True
        assert _matches_tag_filter(node_tags, [["audio", "video"]]) is True
        assert _matches_tag_filter(node_tags, [["audio", "something"]]) is False

    def test_empty_group_in_or_chain(self) -> None:
        """An empty group in an OR chain is simply not matched."""
        assert _matches_tag_filter({"audio"}, [["audio"], []]) is True
        assert _matches_tag_filter({"video"}, [["audio"], []]) is False
