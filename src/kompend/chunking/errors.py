"""Failures that belong to the chunking tier.

Chunking has one failure mode worth a name of its own: the text splitter could
not split a section that is larger than the token budget. That is not a
semantic LLM failure — no model was asked anything — so it is a
:class:`RuntimeError` and callers may treat it as a backend failure; it must
never be reported as a chunk that merely happens to be oversized, because the
unsplit section would then be embedded and searched as if it had been chunked
correctly.
"""

from __future__ import annotations


class ChunkingError(RuntimeError):
    """Raised when a section cannot be split into pieces that fit the budget.

    The message names the splitter, its tokenizer and chunk size, and the
    original failure is kept as ``__cause__`` so the traceback still shows the
    splitter's own error.
    """
