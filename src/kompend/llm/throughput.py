"""Measured request cost — the one place a payload is sized against time.

A request's wall-clock cost is two things with different slopes: the prompt is
*prefilled* before the first token, and the answer is *decoded* one token at a
time. One "tokens per second" number cannot size a request, because a 49,000-token
prompt and a 400-token answer dominate opposite ends of the curve. Measured on
pyvas (2026-11-18): requests of 49,212 prompt tokens were sent with a 120 s
ceiling to a model decoding at 1.4 tokens/s and failed 18 times out of 35 - the
window allowed them, the clock did not.

So the cost model is a two-term linear fit over real calls:

``elapsed_s ≈ prompt_tokens / prefill_rate + completion_tokens / decode_rate``

Two unknowns, solved by least squares over the retained samples. Nothing here is
configured: the rates are whatever the provider actually delivered, which is the
only honest answer to "how big may this request be".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

#: Samples a fit needs before it may size anything. Two calls define a line,
#: but not a trustworthy one; below this the caller keeps the ceiling.
MIN_SAMPLES: Final[int] = 3

#: A rate below this is treated as unusable, so an estimate never divides by
#: zero and never claims a request fits when the model is effectively stalled.
MIN_RATE: Final[float] = 0.05

#: Multiplier applied to an estimate before it becomes a timeout. A fit is a
#: mean, and a mean is not a bound: the slack is what keeps a normal call from
#: timing out because it was slower than average.
SAFETY_FACTOR: Final[float] = 1.5

#: A request may not be given less than this, however small the estimate: a
#: tiny payload still pays connection setup and provider overhead.
MIN_TIMEOUT_S: Final[float] = 15.0


@dataclass(frozen=True, slots=True)
class CallSample:
    """One completed request, as the provider actually served it."""

    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class Throughput:
    """Fitted prefill and decode rates for one model."""

    prefill_rate: float
    decode_rate: float
    samples: int

    def seconds_for(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Expected wall clock for a request of this shape."""
        prompt_s = max(prompt_tokens, 0) / self.prefill_rate
        decode_s = max(completion_tokens, 0) / self.decode_rate
        return prompt_s + decode_s

    def prompt_tokens_within(self, seconds: float, completion_tokens: int) -> int:
        """Prompt tokens that fit in *seconds* once the answer is paid for.

        Returns ``0`` when the answer alone does not fit: the request cannot be
        made smaller by shrinking its input, so a caller must either raise the
        ceiling or accept a shorter answer - never crop the content silently.
        """
        decode_s = max(completion_tokens, 0) / self.decode_rate
        remaining = seconds - decode_s
        if remaining <= 0.0:
            return 0
        return int(remaining * self.prefill_rate)


def timeout_for(throughput: Throughput | None, prompt_tokens: int, completion_tokens: int, *, ceiling_s: float) -> float:
    """Per-request timeout: the fitted estimate plus slack, never above *ceiling_s*.

    A model with no fit yet gets the full ceiling - the first calls are how the
    fit is learned, so they must not be starved by it.
    """
    if throughput is None:
        return ceiling_s
    estimate = throughput.seconds_for(prompt_tokens, completion_tokens) * SAFETY_FACTOR
    return max(MIN_TIMEOUT_S, min(ceiling_s, estimate))


def fits_within(throughput: Throughput | None, prompt_tokens: int, completion_tokens: int, *, ceiling_s: float) -> bool:
    """Would a request of this shape finish inside the ceiling, with slack?"""
    if throughput is None:
        return True
    return throughput.seconds_for(prompt_tokens, completion_tokens) * SAFETY_FACTOR <= ceiling_s


def fit(samples: Sequence[CallSample]) -> Throughput | None:
    """Least-squares fit of the two-term cost model, or ``None`` without data.

    Solved in closed form. A sample set that does not spread across both terms
    (every call the same shape) leaves the system singular: the fit then says
    nothing, and the caller keeps its ceiling rather than inventing a rate.
    """
    usable = [s for s in samples if s.elapsed_s > 0.0 and (s.prompt_tokens > 0 or s.completion_tokens > 0)]
    if len(usable) < MIN_SAMPLES:
        return None

    s11 = sum(s.prompt_tokens * s.prompt_tokens for s in usable)
    s12 = sum(s.prompt_tokens * s.completion_tokens for s in usable)
    s22 = sum(s.completion_tokens * s.completion_tokens for s in usable)
    t1 = sum(s.prompt_tokens * s.elapsed_s for s in usable)
    t2 = sum(s.completion_tokens * s.elapsed_s for s in usable)

    determinant = s11 * s22 - s12 * s12
    scale = max(s11 * s22, 1.0)
    if determinant <= scale * 1e-9:
        return None

    seconds_per_prompt_token = (t1 * s22 - t2 * s12) / determinant
    seconds_per_completion_token = (t2 * s11 - t1 * s12) / determinant
    if seconds_per_prompt_token <= 0.0 or seconds_per_completion_token <= 0.0:
        # A negative slope is not a model: it means the samples do not describe
        # a cost that grows with work (a cached prefix, a warm second call).
        return None

    prefill_rate = _rate(1.0 / seconds_per_prompt_token)
    decode_rate = _rate(1.0 / seconds_per_completion_token)
    if prefill_rate is None or decode_rate is None:
        return None
    return Throughput(prefill_rate=prefill_rate, decode_rate=decode_rate, samples=len(usable))


def _rate(value: float) -> float | None:
    """Clamp a fitted rate into the usable range, or reject it."""
    if value < MIN_RATE:
        return None
    return value
