"""The lint ``details`` registry stays one vocabulary.

``details`` crosses the lint history, the MCP envelope and the web API, so its
keys are a wire contract. :class:`~kompend.lint.details.LintField` is the single
owner of that vocabulary and :class:`~kompend.lint.details.LintDetails` is its
typed shape; if the two drift, a rule writes a key no reader knows about.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_type_hints

import pytest

from kompend.lint.details import (
    _PATH_KEYS,
    LintDetails,
    LintField,
    details_for_storage,
    details_from_storage,
)

#: Bare nouns the naming reference forbids for a filesystem path. A detail key
#: that names a file must say ``_file`` and hold a ``Path``.
_FORBIDDEN_PATH_NAMES = frozenset({"path", "file", "folder", "dir", "directory", "dest"})


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


def test_no_path_key_is_a_bare_noun() -> None:
    """``attachment_file``, never ``path``; ``markdown_file``, never ``file``.

    Scoped to the path keys on purpose: a bare noun is only forbidden where it
    names a filesystem path — ``target`` is a wikilink target, a page name.
    """
    for key in _PATH_KEYS:
        assert str(key) not in _FORBIDDEN_PATH_NAMES, f"{key!r} is a forbidden name for a filesystem path"


def test_every_path_key_holds_a_pathlib_path() -> None:
    """A path-valued key is a ``Path`` in memory, not a string a caller must parse."""
    assert _PATH_KEYS, "the registry declares no path key — the guard would be vacuous"
    hints = get_type_hints(LintDetails)
    for key in _PATH_KEYS:
        assert hints[str(key)] is Path, key
        assert str(key).endswith(("_file", "_dir")), key


def test_a_path_key_is_a_string_in_the_store() -> None:
    """The store keeps a relative POSIX string, so a moved vault keeps its history."""
    stored = details_for_storage({LintField.ATTACHMENT_FILE: Path("attachments/a/b.png")})
    assert stored == {LintField.ATTACHMENT_FILE: "attachments/a/b.png"}


def test_an_absolute_path_is_refused_at_the_boundary() -> None:
    """Persistence is where a vault stops being relocatable, so it must fail there."""
    with pytest.raises(ValueError, match="vault-relative"):
        details_for_storage({LintField.ATTACHMENT_FILE: Path("C:/elsewhere/a.png")})


def test_a_string_under_a_path_key_is_refused() -> None:
    """A caller that kept the old string shape is a bug, not a legacy row."""
    with pytest.raises(TypeError, match="pathlib.Path"):
        details_for_storage({LintField.ATTACHMENT_FILE: "attachments/a/b.png"})


def test_the_round_trip_restores_a_path() -> None:
    """What the runner stores, the history reader hands back as a ``Path``."""
    original: LintDetails = {LintField.MARKDOWN_FILE: Path("wiki/page.md"), LintField.LINE: 3}
    assert details_from_storage(details_for_storage(original)) == original


def test_a_row_written_by_an_older_build_still_loads() -> None:
    """A stored string under a path key is data, not a bug to reject."""
    restored = details_from_storage({LintField.MARKDOWN_FILE: "wiki/old.md", "legacy_note": "kept"})
    assert restored == {LintField.MARKDOWN_FILE: Path("wiki/old.md"), "legacy_note": "kept"}
