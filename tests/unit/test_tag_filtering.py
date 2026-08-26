"""Tests for tag filtering logic in the query command.

Verifies AND/OR boolean logic for tag-based filtering of search results.
"""

from __future__ import annotations

from knox.cli.cmds.knowledge.query import _parse_attr_filters, matches_attr_filter, matches_tag_filter


class TestMatchesTagFilter:
    """Boolean AND/OR tag matching logic."""

    def test_no_filter_returns_all(self) -> None:
        """Empty tag_groups matches all nodes."""
        assert matches_tag_filter({"audio"}, []) is True
        assert matches_tag_filter(set(), []) is True

    def test_single_tag_exact_match(self) -> None:
        """A single tag group matches nodes with that tag."""
        assert matches_tag_filter({"audio"}, [["audio"]]) is True
        assert matches_tag_filter({"paper"}, [["audio"]]) is False

    def test_and_group_requires_all_tags(self) -> None:
        """Comma-separated tags within one group require ALL tags (AND)."""
        assert matches_tag_filter({"audio", "paper"}, [["audio", "paper"]]) is True
        assert matches_tag_filter({"audio"}, [["audio", "paper"]]) is False
        assert matches_tag_filter({"paper"}, [["audio", "paper"]]) is False

    def test_or_between_groups(self) -> None:
        """Multiple tag groups are ORed — match any group."""
        assert matches_tag_filter({"audio"}, [["audio"], ["paper"]]) is True
        assert matches_tag_filter({"paper"}, [["audio"], ["paper"]]) is True
        assert matches_tag_filter({"video"}, [["audio"], ["paper"]]) is False

    def test_complex_and_or_combination(self) -> None:
        """(a AND b) OR c — one AND-group plus one single."""
        assert matches_tag_filter({"audio", "paper"}, [["audio", "paper"], ["video"]]) is True
        assert matches_tag_filter({"video"}, [["audio", "paper"], ["video"]]) is True
        assert matches_tag_filter({"audio"}, [["audio", "paper"], ["video"]]) is False

    def test_empty_tag_group_matches_nothing(self) -> None:
        """An empty tag group (from empty string input) matches nothing."""
        assert matches_tag_filter({"audio"}, [[]]) is False

    def test_node_without_tags_does_not_match(self) -> None:
        """Empty node_tags does not match any non-empty group."""
        assert matches_tag_filter(set(), [["audio"]]) is False

    def test_case_sensitive_matching(self) -> None:
        """Tag matching is case-sensitive."""
        assert matches_tag_filter({"Audio"}, [["audio"]]) is False
        assert matches_tag_filter({"audio"}, [["Audio"]]) is False

    def test_multiple_tags_on_node(self) -> None:
        """A node with many tags matches if any group is satisfied."""
        node_tags = {"audio", "paper", "video", "image"}
        assert matches_tag_filter(node_tags, [["video", "image"]]) is True
        assert matches_tag_filter(node_tags, [["audio", "video"]]) is True
        assert matches_tag_filter(node_tags, [["audio", "something"]]) is False

    def test_empty_group_in_or_chain(self) -> None:
        """An empty group in an OR chain is simply not matched."""
        assert matches_tag_filter({"audio"}, [["audio"], []]) is True
        assert matches_tag_filter({"video"}, [["audio"], []]) is False


class TestMatchesAttrFilter:
    """Key/value attribute matching logic (AND of key checks)."""

    def test_no_filter_returns_all(self) -> None:
        """Empty attr_filters matches all nodes."""
        assert matches_attr_filter({"version": ["1"]}, []) is True
        assert matches_attr_filter({}, []) is True

    def test_key_presence_only(self) -> None:
        """Bare key filter (value=None) requires only that the key exists."""
        assert matches_attr_filter({"version": ["1"]}, [("version", None)]) is True
        assert matches_attr_filter({"status": ["draft"]}, [("version", None)]) is False

    def test_exact_value_match(self) -> None:
        """key=value requires the exact value in the node's value list."""
        assert matches_attr_filter({"version": ["1"]}, [("version", "1")]) is True
        assert matches_attr_filter({"version": ["2"]}, [("version", "1")]) is False
        assert matches_attr_filter({"version": ["1", "2"]}, [("version", "2")]) is True

    def test_multiple_filters_are_anded(self) -> None:
        """Multiple --attr flags are AND'd — all must match."""
        attrs = {"version": ["1"], "status": ["draft"]}
        assert matches_attr_filter(attrs, [("version", "1"), ("status", "draft")]) is True
        assert matches_attr_filter(attrs, [("version", "1"), ("status", "final")]) is False

    def test_missing_key_fails(self) -> None:
        """A missing key fails the filter even with a value check."""
        assert matches_attr_filter({}, [("missing", "x")]) is False


class TestParseAttrFilters:
    """Parsing of --attr KEY=VALUE / --attr KEY CLI values."""

    def test_none_and_empty(self) -> None:
        assert _parse_attr_filters(None) == []
        assert _parse_attr_filters([]) == []

    def test_key_value(self) -> None:
        assert _parse_attr_filters(["version=1"]) == [("version", "1")]

    def test_bare_key(self) -> None:
        assert _parse_attr_filters(["status"]) == [("status", None)]

    def test_mixed(self) -> None:
        assert _parse_attr_filters(["a=1", "b"]) == [("a", "1"), ("b", None)]

    def test_value_with_equals(self) -> None:
        """Only the first '=' splits — value may contain '='."""
        assert _parse_attr_filters(["k=a=b"]) == [("k", "a=b")]
