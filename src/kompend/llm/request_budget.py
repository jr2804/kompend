"""One place that decides whether an LLM request fits the context window.

Every stage that talks to a model must answer the same question: *how many
tokens are left for my content once the fixed per-request cost, the answer
reserve and the safety margin are paid?* Answering it per stage means each
stage carries its own arithmetic, and the arithmetic drifts — one stage
forgets the safety margin, another forgets the answer reserve, and the
symptom is always the same provider-side 400 (``request (33145 tokens)
exceeds the available context size (32768 tokens)``) discovered only after
minutes of upstream work.

This module owns that arithmetic:

- :class:`RequestBudget` — a window plus the named payloads reserved on top
  of it, with the content budget derived once.
- :func:`context_safety_tokens` — the shared headroom every stage holds
  back (a server's usable context is smaller than its nominal window, and
  the chars-per-token heuristic undercounts dense content).
- :func:`chars_to_tokens` / :func:`tokens_to_chars` — the single
  chars-per-token conversion, driven by the configured value.

The rule the whole pipeline obeys: **content is never cropped to fit.**
A budget that is too small makes the work split across more batches or
segments; it never drops text. When a request cannot fit at all, the
planner raises :class:`LLMOutputError` naming the components instead of
letting the provider reject it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from kompend.core.utils.exceptions import LLMOutputError
from kompend.llm.context_window import ContextSource, ContextWindow

#: Fraction of a model's context window held back as headroom beyond the
#: measured per-request overhead.  Two reasons: a server's *effective* usable
#: context is smaller than the nominal window (KV cache slotting, prompt-cache
#: rollover), and the chars-per-token heuristic undercounts token-dense content
#: (formulas, tables, identifiers).  Without it a nominally exact-fit request
#: still overflows — observed as ``request (65762 tokens) exceeds the available
#: context size (65536 tokens)`` on a formula-heavy standard.
#:
#: Content is never cropped because of this margin: the content budget shrinks,
#: so the work splits across more batches/segments instead.
CONTEXT_SAFETY_FRACTION: float = 0.08

#: Floor for :func:`context_safety_tokens` so a narrow window cannot round the
#: headroom away to nothing.
CONTEXT_SAFETY_TOKENS: int = 512

#: Default chars-per-token heuristic. Configurable via ``compile.llm.default_chars_per_token``
#: (env override ``KOMP_LLM_DEFAULT_CHARS_PER_TOKEN``).
DEFAULT_CHARS_PER_TOKEN: float = 3.85


@dataclass(frozen=True)
class RequestBudget:
    """What a request may carry, given the window and the fixed overhead.

    *reserved* names the fixed payloads that land in the request besides the
    content — ``{"system": 1200, "known_titles": 4000}``. The names exist so
    a log line and an error message can say *which* payload ate the window;
    only the sum is subtracted.

    *answer_tokens* defaults to the window's answer reserve and *safety_tokens*
    to :func:`context_safety_tokens` of the window, so a stage cannot forget
    either by accident. Pass an explicit value only when the stage has a
    measured reason to differ.

    *utilization* scales the remaining content budget: a stage that needs
    room for a second leg of the same request (a probe and its evaluation)
    takes a fraction instead of the whole remainder.
    """

    window: ContextWindow
    reserved: Mapping[str, int] = field(default_factory=dict)
    answer_tokens: int | None = None
    safety_tokens: int | None = None
    utilization: float = 1.0

    def __post_init__(self) -> None:
        if self.answer_tokens is None:
            object.__setattr__(self, "answer_tokens", self.window.answer_reserve)
        if self.safety_tokens is None:
            object.__setattr__(self, "safety_tokens", context_safety_tokens(self.window.max_input))

    @property
    def answer_reserve(self) -> int:
        """Tokens held back for the answer."""
        return int(self.answer_tokens or 0)

    @property
    def safety(self) -> int:
        """Tokens held back as headroom."""
        return int(self.safety_tokens or 0)

    @property
    def reserved_total(self) -> int:
        """Total tokens the named reserved payloads occupy."""
        return sum(int(value) for value in self.reserved.values())

    @property
    def available_tokens(self) -> int:
        """Tokens left for content after window, answer reserve, safety and reserved.

        May be negative: that is the signal that the fixed overhead does not
        fit the window at all. Use :attr:`content_tokens` for a usable budget
        and :meth:`require` to fail loudly.
        """
        return self.window.max_input - self.answer_reserve - self.safety - self.reserved_total

    @property
    def content_tokens(self) -> int:
        """Tokens available for content, scaled by :attr:`utilization` (at least 1)."""
        return max(1, int(self.available_tokens * self.utilization))

    def content_chars(self, chars_per_token: float) -> int:
        """Character budget for content, using the configured chars-per-token."""
        return tokens_to_chars(self.content_tokens, chars_per_token=chars_per_token)

    def fits(self, content_tokens: int) -> bool:
        """Whether *content_tokens* of content fit this budget."""
        return content_tokens <= self.content_tokens

    def require(self, content_tokens: int, *, stage: str) -> None:
        """Raise :class:`LLMOutputError` when *content_tokens* do not fit.

        The message names the components (window, answer reserve, safety
        margin, each reserved payload) so the fix is visible without reading
        the code.
        """
        if self.fits(content_tokens):
            return
        reserved = ", ".join(f"{name}={value}" for name, value in sorted(self.reserved.items()))
        raise LLMOutputError(
            f"{stage}: content of {content_tokens} tokens does not fit the "
            f"{self.window.max_input}-token context window "
            f"(answer reserve {self.answer_reserve}, safety margin {self.safety}"
            + (f", reserved {reserved}" if reserved else "")
            + f", usable {self.available_tokens} tokens"
            + (f" x utilization {self.utilization}" if self.utilization != 1.0 else "")
            + "). Content is never cropped: shrink the reserved payloads "
            "(known_titles is windowed) or raise compile.llm.context_window."
        )

    def report(self) -> str:
        """One-line description of the budget, for log lines."""
        reserved = ", ".join(f"{name}={value}" for name, value in sorted(self.reserved.items()))
        return (
            f"cw={self.window.max_input} answer={self.answer_reserve} safety={self.safety} "
            f"reserved={self.reserved_total} ({reserved}) usable={self.available_tokens} "
            f"content={self.content_tokens}"
        )


@dataclass
class EffectiveWindow:
    """The window a run actually plans against.

    Discovery reports what a model *can* do; a server may enforce less. The
    gap shows up only when a request is rejected, and the provider's own
    number is then the only authoritative one. This holder owns the window
    the stages plan against and lets a provider overflow lower it once, for
    the rest of the run.

    Lowering is one-way: a window that was already clamped is never raised
    back, because a provider that rejected a request of size *N* has
    rejected the evidence for a larger window.
    """

    window: ContextWindow
    #: Why the window was lowered, for the log line.
    reason: str = ""
    #: The window before the first clamp (the discovered one).
    discovered: ContextWindow | None = None

    def __post_init__(self) -> None:
        if self.discovered is None:
            self.discovered = self.window

    @property
    def current(self) -> ContextWindow:
        """The window stages must size their requests against."""
        return self.window

    @property
    def clamped(self) -> bool:
        """Whether the window was lowered below the discovered one."""
        return self.window.max_input < (self.discovered.max_input if self.discovered else self.window.max_input)

    def clamp(self, max_input: int, *, reason: str) -> bool:
        """Lower the window to *max_input* tokens.

        Returns whether the window changed. A *max_input* at or above the
        current window is ignored — the provider either repeated the limit
        kompend already planned against, or named a number larger than the
        limit (in which case trusting it would grow the request that just
        failed).
        """
        if max_input <= 0 or max_input >= self.window.max_input:
            return False
        self.window = replace(self.window, max_input=max_input, source=ContextSource.FALLBACK)
        self.reason = reason
        return True

    def report(self) -> str:
        """One-line description of the effective window, for log lines."""
        if not self.clamped:
            return f"cw={self.window.max_input} (source={self.window.source})"
        discovered = self.discovered.max_input if self.discovered else self.window.max_input
        return f"cw={self.window.max_input} (discovered {discovered}, {self.reason})"


def context_safety_tokens(max_input: int) -> int:
    """Headroom in tokens to hold back from a *max_input*-token context window.

    The single definition of the safety margin shared by every stage that
    sizes a request against the context window (extraction batching, agentic
    generation segments, reference extraction, the agentic section digest) —
    so the stages cannot drift apart on how much of the window is actually
    available for content.
    """
    return max(CONTEXT_SAFETY_TOKENS, int(round(CONTEXT_SAFETY_FRACTION * max_input)))


def tokens_to_chars(tokens: int, *, chars_per_token: float | None = None) -> int:
    """Convert token count to approximate character count.

    Args:
        tokens: Token count to convert.
        chars_per_token: Optional override (e.g. from ``compile.llm.default_chars_per_token``).
    """
    return int(tokens * (chars_per_token if chars_per_token is not None else DEFAULT_CHARS_PER_TOKEN))


def chars_to_tokens(chars: int, *, chars_per_token: float | None = None) -> int:
    """Convert character count to approximate token count.

    Args:
        chars: Character count to convert.
        chars_per_token: Optional override (e.g. from ``compile.llm.default_chars_per_token``).
    """
    cpt = chars_per_token if chars_per_token is not None else DEFAULT_CHARS_PER_TOKEN
    return max(1, int(chars / cpt))
