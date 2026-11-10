"""Who owns what after a split: attachment and figure attribution for children.

A child is a virtual source — no file on disk, no entry in the attachment
manifest. The parent owns the attachments (they were registered against the
parent's stem) and the parent's element list holds nothing the child already
has: children are cut from *copies*, so the parent document stays whole.

Two questions are answered here, both from the metadata a split stamps:

* which stem's attachments belong to a child (:func:`attachment_owner_stem`);
* which document a child was cut from (:func:`parent_source_of`).

Reading the keys through :class:`SplitMetadata` keeps the vocabulary in one
place; a caller never writes ``"_split_parent_stem"`` itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._elements import SplitMetadata

__all__ = ["attachment_owner_stem", "parent_source_of"]


def attachment_owner_stem(source_file: Path | str | None, metadata: Mapping[str, Any] | None) -> str | None:
    """Stem whose attachment directory holds *source_file*'s attachments.

    A split child answers with its parent's stem — attachments are registered
    against the file that was ingested, and the child was never ingested on its
    own. Any other source answers with its own stem. ``None`` means there is no
    stem to look attachments up by (no source file at all).
    """
    parent = _metadata_text(metadata, SplitMetadata.PARENT_STEM)
    if parent:
        return parent
    if source_file is None:
        return None
    stem = Path(str(source_file)).stem
    return stem or None


def parent_source_of(metadata: Mapping[str, Any] | None) -> Path | None:
    """The parent document a split child was cut from, if it is one.

    The path is the parent's vault-relative path as stamped at partition time;
    it names a file that exists, unlike the child's own synthetic path.
    """
    parent = _metadata_text(metadata, SplitMetadata.PARENT_SOURCE)
    return Path(parent) if parent else None


def _metadata_text(metadata: Mapping[str, Any] | None, key: SplitMetadata) -> str:
    """The non-empty string *metadata* carries under *key*, else ``""``."""
    if not metadata:
        return ""
    value = metadata.get(str(key.value)) or metadata.get(key.value)
    return value.strip() if isinstance(value, str) else ""
