"""Tag and attribute filter predicates for query results.

Canonical home for the filter semantics used across the query paths:

- Tags are an **OR of ANDs** (DNF): a node matches when its tag set contains
  *all* tags of *at least one* group.
- Attributes are an **AND of checks**: ``(key, value)`` requires the value to
  be present; ``(key, None)`` requires only that the key exists.

Declared here (rather than in ``cli._query_helpers``) so the portable
:class:`kompend.search.retriever.KeywordRetriever` can apply the same predicates
without importing the CLI (which imports the search package — a cycle).
``cli._query_helpers`` re-exports both for its callers.
"""

from __future__ import annotations


def matches_tag_filter(node_tags: set[str], tag_groups: list[list[str]]) -> bool:
    """Check if node satisfies at least one AND-group (OR of ANDs)."""
    if not tag_groups:
        return True
    return any(group and all(tag in node_tags for tag in group) for group in tag_groups)


def matches_attr_filter(
    node_attrs: dict[str, list[str]],
    attr_filters: list[tuple[str, str | None]],
) -> bool:
    """Check if node attributes satisfy all filters (AND of key/value checks).

    Each filter is ``(key, value)``: *value* is ``None`` when only key presence
    is required; otherwise the node's value list must contain the exact value.
    """
    if not attr_filters:
        return True
    for key, value in attr_filters:
        values = node_attrs.get(key)
        if values is None:
            return False
        if value is not None and value not in values:
            return False
    return True
