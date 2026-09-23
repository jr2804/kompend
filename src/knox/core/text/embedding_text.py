"""Compose the text to embed for a chunk.

A chunk's situating context (``ChunkNode.contextual_context``) is prepended to
its content before embedding, so the vector carries the document narrative
rather than only the fragment (see ``ContextualEnricher``). This module is the
single composition point shared by every chunk-embedding consumer — dedup and
gap analysis must agree on the same chunk's vector, so the rule lives here and
nowhere else.
"""

from __future__ import annotations

#: Separator between the situating context and the chunk content.
CONTEXT_SEPARATOR = "\n"


def embed_text(text: str, contextual_context: str = "") -> str:
    """Return the text to embed: *contextual_context* prepended when present.

    When *contextual_context* is empty the input is returned unchanged, so a
    pipeline without the enricher behaves exactly as before.
    """
    if not contextual_context:
        return text
    return f"{contextual_context}{CONTEXT_SEPARATOR}{text}"
