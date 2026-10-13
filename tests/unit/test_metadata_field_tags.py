"""Tests for the MetadataField.TAGS enum value.

Ensures the canonical metadata key is consistent across the pipeline.
"""

from __future__ import annotations

from kompend.core.models.enums import MetadataField


class TestMetadataFieldTags:
    """MetadataField.TAGS enum value."""

    def test_tags_member_exists(self) -> None:
        """TAGS is a valid member of MetadataField."""
        assert hasattr(MetadataField, "TAGS")
        assert MetadataField.TAGS.value == "tags"

    def test_tags_is_string(self) -> None:
        """TAGS value is a plain string suitable for dict keys."""
        assert isinstance(MetadataField.TAGS.value, str)

    def test_tags_can_be_used_as_dict_key(self) -> None:
        """TAGS value works as a dictionary key."""
        d: dict[str, list[str]] = {}
        d[MetadataField.TAGS.value] = ["audio", "paper"]
        assert d["tags"] == ["audio", "paper"]
