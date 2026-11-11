"""The lint ``details`` registry stays one vocabulary.

``details`` crosses the lint history, the MCP envelope and the web API, so its
keys are a wire contract. :class:`~kompend.lint.details.LintField` is the single
owner of that vocabulary and :class:`~kompend.lint.details.LintDetails` is its
typed shape; if the two drift, a rule writes a key no reader knows about.
"""

from __future__ import annotations

from kompend.lint.details import LintDetails, LintField


def test_the_enum_and_the_typed_shape_declare_the_same_keys() -> None:
    """A key in one and not the other is a contract nobody can honour."""
    assert set(LintField) == set(LintDetails.__annotations__)


def test_every_member_value_matches_its_field_name() -> None:
    """The enum member name and the wire key must be the same word."""
    for member in LintField:
        assert member.name.lower() == member.value


def test_no_field_carries_a_keyword_name() -> None:
    """The shape is a class-form TypedDict, so every key must be an identifier."""
    for key in LintDetails.__annotations__:
        assert key.isidentifier(), key
