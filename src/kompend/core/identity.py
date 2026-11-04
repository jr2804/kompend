"""How kompend identifies a source document.

A source is identified by its **file name plus a prefix of its content hash** —
never by its path. This is the single definition of that rule, shared by the
parse cache (:mod:`kompend.core.pipeline.parse_cache`), the tier state
(:mod:`kompend.core.pipeline.state`) and the store migration that rekeyed
``tier_state`` (``kompend.core.store.schema``).

Two consequences follow from it, and both are deliberate:

- Grouping sources into folders, moving them around under ``sources/``, or
  duplicating a file never changes a source's identity, so caches and state
  survive a reorganisation.
- Two files that share a name but differ in content stay two distinct sources,
  because the hash prefix is part of the key.
"""

from __future__ import annotations

from typing import Final

#: Length of the content-hash prefix that discriminates two same-named sources.
IDENTITY_HASH_LEN: Final[int] = 8


def source_key(stem: str, content_hash: str) -> str:
    """Identity of a source: ``<stem>-<hash8>``.

    Parameters
    ----------
    stem:
        The source's file name without its suffix (``"beerends-2013-perceptualI"``).
    content_hash:
        Content hash of the source bytes; any hex digest works, only its first
        :data:`IDENTITY_HASH_LEN` characters are used.

    Returns:
    -------
    str
        The identity string, e.g. ``"beerends-2013-perceptualI-7c4d4f17"``.
    """
    return f"{stem}-{content_hash[:IDENTITY_HASH_LEN]}"


__all__ = ["IDENTITY_HASH_LEN", "source_key"]
