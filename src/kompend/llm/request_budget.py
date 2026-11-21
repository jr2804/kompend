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
- :func:`usable_window_tokens` — the window minus the answer reserve and the
  safety margin: the share a trigger or a target is a fraction of.
- :func:`chars_to_tokens` / :func:`tokens_to_chars` — the single
  chars-per-token conversion, driven by the configured value.
- :func:`calibrated_density` — the sizing density tightened by the run's
  own measured chars-per-token, never loosened past the floor.
- :func:`calibrated_density_for` — the same blend resolved for a stage
  from its config, tracker and model: the single entry point stages call.

The rule the whole pipeline obeys: **content is never cropped to fit.**
A budget that is too small makes the work split across more batches or
segments; it never drops text. When a request cannot fit at all, the
planner raises :class:`LLMOutputError` naming the components instead of
letting the provider reject it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from kompend.core.config.models.llm import LLMConfig
from kompend.core.text.density import CONSERVATIVE_CHARS_PER_TOKEN
from kompend.core.utils.exceptions import LLMOutputError
from kompend.llm.context_window import ContextSource, ContextWindow, answer_reserve_tokens
from kompend.llm.throughput import SAFETY_FACTOR, Throughput
from kompend.llm.usage import UsageTracker

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
#: (env override ``KOMP_LLM_DEFAULT_CHARS_PER_TOKEN``). The value is the pessimistic
#: density bound every stage shares, so the planner and the request-fit guard count
#: a request in the same unit — see :mod:`kompend.core.text.density`.
DEFAULT_CHARS_PER_TOKEN: float = CONSERVATIVE_CHARS_PER_TOKEN


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

    The window's ``content_window_tokens`` (``compile.llm.content_window_tokens``)
    sizes the result: a model that answers nothing to a very long prompt gets a
    smaller payload per request, which means more requests — never less content.
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
        """Tokens available for content, scaled by :attr:`utilization` (at least 1).

        Bounded by :attr:`ContextWindow.content_window_tokens` when the run
        configured one, so a stage never packs more content into one request
        than the model is known to handle even though the window could hold it.
        The stage splits its payload; nothing is dropped.
        """
        scaled = int(self.available_tokens * self.utilization)
        if self.window.content_window_tokens is not None:
            scaled = min(scaled, self.window.content_window_tokens)
        return max(1, scaled)

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
            + (f", content window {self.window.content_window_tokens} tokens" if self.window.content_window_tokens is not None else "")
            + "). Content is never cropped: shrink the reserved payloads "
            "(known_titles is windowed) or raise compile.llm.context_window."
        )

    def report(self) -> str:
        """One-line description of the budget, for log lines."""
        reserved = ", ".join(f"{name}={value}" for name, value in sorted(self.reserved.items()))
        cap = f" window={self.window.content_window_tokens}" if self.window.content_window_tokens is not None else ""
        return (
            f"cw={self.window.max_input} answer={self.answer_reserve} safety={self.safety} "
            f"reserved={self.reserved_total} ({reserved}) usable={self.available_tokens} "
            f"content={self.content_tokens}{cap}"
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


def usable_window_tokens(max_input: int) -> int:
    """Tokens of a *max_input*-token window a request may spend on content.

    The answer reserve and the safety margin are paid before the content:
    a history that fills the *whole* window already asks for more than the
    window holds. At a 64 000-token window the usable share is
    ``64 000 - 8 192 (answer) - 5 120 (safety) = 50 688`` tokens; at a
    1 000 000-token window it is ``1 000 000 - 8 192 - 80 000 = 911 808``.

    Every bound that scales with the window derives from this number, so a
    trigger or a target expressed as a fraction of it holds at any window
    size — the formula, not the 64 000 case, is the contract.
    """
    return max(1, max_input - answer_reserve_tokens(max_input) - context_safety_tokens(max_input))


def context_safety_tokens(max_input: int) -> int:
    """Headroom in tokens to hold back from a *max_input*-token context window.

    The single definition of the safety margin shared by every stage that
    sizes a request against the context window (extraction batching, agentic
    generation segments, reference extraction, the agentic section digest) —
    so the stages cannot drift apart on how much of the window is actually
    available for content.
    """
    return max(CONTEXT_SAFETY_TOKENS, int(round(CONTEXT_SAFETY_FRACTION * max_input)))


def chars_to_tokens(chars: int, *, chars_per_token: float | None = None) -> int:
    """Convert character count to the tokens it will occupy.

    Args:
        chars: Character count to convert.
        chars_per_token: Optional override (e.g. from ``compile.llm.default_chars_per_token``),
            capped at :func:`sizing_density`.
    """
    return max(1, int(chars / sizing_density(chars_per_token)))


def time_bounded_chars(
    throughput: Throughput | None,
    *,
    ceiling_s: float,
    completion_tokens: int,
    chars_per_token: float | None = None,
) -> int | None:
    """Characters that also fit the wall clock, or ``None`` when unbounded.

    The window says how much a request *may* carry; it says nothing about how
    much this model can *answer* before the ceiling. Both bounds apply, and this
    is the time half: the prompt may only be as large as
    ``ceiling / SAFETY_FACTOR`` allows once the answer's decode time is paid.

    ``None`` means "no time bound available yet" (no fit - the first calls are
    how the fit is learned) or "even the answer alone does not fit", in which
    case shrinking the prompt cannot help and the caller must not pretend it
    has a budget: it keeps its window bound, the request runs, and the honest
    timeout/shrink path decides.
    """
    if throughput is None or ceiling_s <= 0.0:
        return None
    tokens = throughput.prompt_tokens_within(ceiling_s / SAFETY_FACTOR, completion_tokens)
    if tokens <= 0:
        return None
    return tokens_to_chars(tokens, chars_per_token=chars_per_token)


def tokens_to_chars(tokens: int, *, chars_per_token: float | None = None) -> int:
    """Convert token count to the character count that fits *tokens*.

    Args:
        tokens: Token count to convert.
        chars_per_token: Optional override (e.g. from ``compile.llm.default_chars_per_token``),
            capped at :func:`sizing_density`.
    """
    return int(tokens * sizing_density(chars_per_token))


def calibrated_density(configured: float | None, measured: float | None) -> float:
    """Characters per token to *size* a request with, informed by measurement.

    The floor (:func:`sizing_density`) is the worst case — the density that
    keeps a planned request inside the window for every density the corpus
    shows. The measured fit (chars per provider prompt token across the
    run's own calls) is the better-informed signal: a deployed tokenizer
    denser than the floor is caught here, before it overflows a window the
    floor said was safe.

    The blend is the minimum of the two, so the measurement can only ever
    *tighten* sizing, never loosen it: replacing the floor with an
    optimistic measurement would invert the safety direction (at 4
    chars/token the estimate undercounts 76% of tails). ``None`` measured
    keeps the floor until enough requests exist.
    """
    floor = sizing_density(configured)
    if measured is None:
        return floor
    return min(floor, measured)


def calibrated_density_for(llm: LLMConfig, tracker: UsageTracker | None, model: str | None) -> float:
    """Chars-per-token a stage sizes a new payload with, given the run so far.

    The single entry point for self-calibrated sizing: the configured floor
    tightened by the run's own measured density for *model* (see
    :func:`calibrated_density`). No tracker, no model, or too few measured
    calls keeps the floor — the first calls are how the fit is learned, so
    they must not be starved by it.
    """
    measured = tracker.measured_chars_per_token(model or "", min_samples=llm.density_min_samples) if tracker is not None else None
    return calibrated_density(llm.default_chars_per_token, measured)


def sizing_density(chars_per_token: float | None = None) -> float:
    """Characters per token to *size* a request with, never an optimistic guess.

    A configured density above kompend's measured floor
    (:data:`~kompend.core.text.density.CONSERVATIVE_CHARS_PER_TOKEN`) is a
    planning guess: the corpus median, which half the text exceeds. Sizing with
    it is how a payload planned for a 64 k window arrives as 78 k tokens,
    overflows, and drags the compaction loop behind it. The floor is the only
    value that keeps a planned request inside the window for every density the
    corpus shows, so a configured value can only make sizing *more*
    pessimistic, never less.

    A model whose text really is sparser than the floor gets smaller batches
    than it could take — the cheap failure, because it costs calls, not
    correctness.
    """
    if chars_per_token is None:
        return DEFAULT_CHARS_PER_TOKEN
    return min(chars_per_token, CONSERVATIVE_CHARS_PER_TOKEN)
