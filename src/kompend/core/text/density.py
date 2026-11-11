"""The chars-per-token heuristic, defined once.

A request is sized in characters and paid for in tokens, so every stage that
plans content needs one number: how many characters one token occupies. Two
numbers were in play — a nominal 3.85 (the corpus median) and a pessimistic
2.5 (the density floor) — and they were used for different jobs. The planner
sized content with the nominal one, the request-fit guard counted with the
pessimistic one, so a segment planned at 3.85 arrived roughly 54 % over the
window and the provider answered "prompt too long" after minutes of upstream
work.

This module owns the number every job uses. The value is pessimistic on
purpose: a request that fits by this count fits at every density the corpus
shows, and the cost of the margin is a smaller batch — never a cropped one.

Measured on the p863 corpus (733 request tails: the context growth between
two responses that each reported usage, minus the later response's output),
reproducible with a throwaway probe in the OS temp dir:
p01 2.65, p05 2.95, p10 3.18, median 3.86 chars per token. Tool returns —
JSON, tables, standard clauses — carry the dense end. At the harness's 4 the
estimate under-counts 560 of those 733 tails (76 %), worst by 2.02x; at this
value it covers 726 of 733 (99.0 %) and the seven it leaves under-count by at
most 1.27x, which the trigger's headroom absorbs.

Erring high is the cheap failure — a trim slightly early costs prompt-cache
writes; a trim late is a provider 400 after minutes of upstream work. The
cost of the margin: a ``keep_tokens`` tail is counted in these units, so a
named 8 000-token tail retains roughly 5 200 real tokens of the same text (it
held ~8 300 at 4 chars/token).
"""

from __future__ import annotations

from math import ceil
from typing import Final

#: Characters per token kompend assumes where the provider's own count is not
#: available. The harness defaults to 4, which is honest for English prose
#: and optimistic for everything else a run puts in a request.
CONSERVATIVE_CHARS_PER_TOKEN: Final[float] = 2.5


def conservative_tokens(text: str) -> int:
    """Count *text* at kompend's measured density floor, never the harness's 4.

    The harness calls this only for the part of a request no provider count
    covers — the messages after the newest usage-carrying response, and a
    history with no usage at all. That is exactly where dense content lives,
    and where under-counting lets a history blow the window.
    """
    return ceil(len(text) / CONSERVATIVE_CHARS_PER_TOKEN)
