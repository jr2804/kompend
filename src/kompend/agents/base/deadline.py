"""Wall-clock bounds for a run: one ceiling per request, one deadline per session.

A request count alone does not bound a session. ``session_limits`` caps how many
turns a session may take, but a slow model can spend a long time inside each
one: a 3.7B model at ~1.4 tokens/s answered nothing, timed out 18 of 35 times
and still consumed 15 hours. The clock is the other half of the same plan —
both are derived from the units the session has to cover.

The request ceiling stays the *upper* bound. What the agent fits from measured
throughput is tighter, and it is applied per request through
:class:`RequestDeadline`, so a turn that would not finish inside the session's
remaining time is cut off with the rest of the budget rather than the full
ceiling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import monotonic

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext

#: Never hand a turn less than this. A timeout shorter than the cost of
#: establishing the connection fails for a reason unrelated to the model.
MIN_TURN_TIMEOUT_S: float = 1.0


@dataclass
class DeadlineState:
    """The wall clock a session may spend, read by :class:`RequestDeadline`.

    ``budget_s = None`` means the session is unbounded (a single-turn agent has
    nothing to distribute), and the request hook then applies only the ceiling
    the agent fitted.
    """

    started: float = field(default_factory=monotonic)
    budget_s: float | None = None

    def arm(self, budget_s: float | None) -> None:
        """Start the clock for a new session, or clear it for an unbounded one."""
        self.started = monotonic()
        self.budget_s = budget_s

    def remaining(self) -> float | None:
        """Seconds left in the session, or ``None`` when it is unbounded."""
        if self.budget_s is None:
            return None
        return self.budget_s - (monotonic() - self.started)


@dataclass
class RequestDeadline(AbstractCapability[AgentDepsT]):
    """Tighten each request's timeout to what the turn can actually finish.

    The agent supplies the callable; the capability only carries it to the
    per-request seam. Returning ``None`` leaves the constructed settings alone.
    """

    timeout_for: Callable[[list[ModelMessage]], float | None]

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT] | None,
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Replace this request's timeout with the fitted, deadline-bounded one."""
        del ctx  # the estimate depends on the messages, not on the deps
        timeout = self.timeout_for(list(request_context.messages))
        if timeout is None:
            return request_context
        settings = dict(request_context.model_settings or {})
        settings["timeout"] = timeout
        return replace(request_context, model_settings=settings)
