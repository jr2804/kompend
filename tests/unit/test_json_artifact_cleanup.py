"""Tests for JSON artifact cleanup in litellm_model.py.

Tests both Layer 1 (automatic cleanup of non-dict items from dict-containing
lists) and Layer 2 (targeted retry feedback with removed artifacts).
"""

from __future__ import annotations

from knox.agents.litellm_model import _clean_json_artifacts


class TestCleanJsonArtifacts:
    """Layer 1: automatic cleanup of mixed-type list artifacts."""

    def test_removes_bare_strings_from_dict_list(self) -> None:
        """The exact pattern from the gemma-4 debug file."""
        data = {
            "pages": [
                {"title": "MTD", "slug": "mtd", "content": "..."},
                "provenances_cited_filenames_from_content_block",
                "related_concepts_linked",
            ]
        }
        cleaned, removed = _clean_json_artifacts(data)
        assert len(cleaned["pages"]) == 1
        assert cleaned["pages"][0]["title"] == "MTD"
        assert removed == [
            "provenances_cited_filenames_from_content_block",
            "related_concepts_linked",
        ]

    def test_preserves_clean_dict_list(self) -> None:
        """No artifacts → no changes."""
        data = {"pages": [{"title": "A"}, {"title": "B"}]}
        cleaned, removed = _clean_json_artifacts(data)
        assert removed == []
        assert len(cleaned["pages"]) == 2

    def test_preserves_string_list(self) -> None:
        """Lists with only strings (no dicts) are left unchanged."""
        data = {"tags": ["a", "b", "c"]}
        cleaned, removed = _clean_json_artifacts(data)
        assert removed == []
        assert cleaned["tags"] == ["a", "b", "c"]

    def test_preserves_int_list(self) -> None:
        """Lists with only ints are left unchanged."""
        data = {"nums": [1, 2, 3]}
        cleaned, removed = _clean_json_artifacts(data)
        assert removed == []
        assert cleaned["nums"] == [1, 2, 3]

    def test_nested_cleanup(self) -> None:
        """Artifacts in nested structures are cleaned recursively.

        When a list contains dicts, ALL non-dict items (ints, strings, etc.)
        are removed because they're artifacts the model misplaced.
        """
        data = {
            "outer": [
                {
                    "inner": [1, {"a": 1}, "str_artifact", {"b": 2}],
                }
            ]
        }
        cleaned, removed = _clean_json_artifacts(data)
        assert removed == ["1", "str_artifact"]
        inner = cleaned["outer"][0]["inner"]
        assert len(inner) == 2
        assert inner[0] == {"a": 1}
        assert inner[1] == {"b": 2}

    def test_nested_cleanup_only_dicts_kept(self) -> None:
        """When a list has dicts, ALL non-dict items are removed (ints too)."""
        data = {"items": [{"a": 1}, 42, {"b": 2}, "str"]}
        cleaned, removed = _clean_json_artifacts(data)
        assert len(cleaned["items"]) == 2
        assert removed == ["42", "str"]

    def test_empty_list_unchanged(self) -> None:
        """Empty lists are left unchanged."""
        data = {"pages": []}
        cleaned, removed = _clean_json_artifacts(data)
        assert removed == []
        assert cleaned["pages"] == []

    def test_non_dict_non_list_unchanged(self) -> None:
        """Scalars are returned as-is."""
        cleaned, removed = _clean_json_artifacts("hello")
        assert cleaned == "hello"
        assert removed == []
        cleaned, removed = _clean_json_artifacts(42)
        assert cleaned == 42
        assert removed == []

    def test_deeply_nested(self) -> None:
        """Multiple levels of nesting."""
        data = {
            "level1": [
                {
                    "level2": [
                        {"name": "valid"},
                        "artifact_str",
                        {"name": "also_valid"},
                    ]
                }
            ]
        }
        cleaned, removed = _clean_json_artifacts(data)
        assert removed == ["artifact_str"]
        assert len(cleaned["level1"][0]["level2"]) == 2
