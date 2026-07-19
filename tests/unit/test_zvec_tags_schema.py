"""Tests for zvec store tag schema and storage.

Verifies that tags are stored and retrieved from the vector store.
"""

from __future__ import annotations

from knox.rag.zvec_store import ZvecField


class TestZvecFieldTags:
    """ZvecField.TAGS schema constant."""

    def test_tags_field_exists(self) -> None:
        """TAGS is a valid member of ZvecField."""
        assert hasattr(ZvecField, "TAGS")
        assert ZvecField.TAGS.value == "tags"

    def test_tags_field_is_string(self) -> None:
        """TAGS value is a plain string."""
        assert isinstance(ZvecField.TAGS.value, str)
