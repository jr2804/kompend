"""Standard pydantic-ai-harness capability bundle for agentic agents.

Agentic agents in kompend share these harness capabilities:

- **Memory** — ``InMemoryStore``, fresh per run (no cross-session persistence).
  Lets the agent write/read a notebook across turns for long-running tasks.
  **Opt-in** (``memory=True``): the notebook is only useful when the agent's
  prompt tells the model to keep notes.  Offering it unconditionally invites
  small models to call ``read_memory`` for a file that does not exist — the
  tool answers with ``ModelRetry`` and, once the tool-retry budget is spent,
  the whole run dies with the node left unsummarized.
- **SlidingWindowCompaction** — ``max_fraction=0.6`` of context window.
  Drops oldest turns when context fills up, avoiding token-limit failures
  on multi-turn tool agents.
- **ClampOversizedMessages** — ``max_part_tokens=context_window // 4``.
  Clamps runaway tool outputs that would blow a single turn.

``AgenticHarness`` centralizes the construction of these capabilities
from a known context window, with an ``extra_capabilities`` hook for
agent-specific additions (e.g. ``ContextManagerCapability`` for agents
that need full-context management instead of sliding window).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai_harness.compaction import ClampOversizedMessages, SlidingWindowCompaction
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.memory import InMemoryStore, Memory


@dataclass
class AgenticHarness:
    """Standard capability bundle for agentic (multi-turn tool) agents.

    All defaults are tuned for the typical kompend agentic agent. Call
    ``build_capabilities()`` to get a list ready for
    ``KompendAgentBase.extra_capabilities``.

    Example::

        harness = AgenticHarness(context_window=ctx)
        agent = MyAgent(..., extra_capabilities=harness.build_capabilities())
    """

    context_window: int | None = None
    memory_agent_name: str = "agentic"
    memory: bool = False
    compaction: bool = True
    clamp: bool = True
    extra_capabilities: Sequence[AbstractCapability] = field(default_factory=list)

    def build_capabilities(self) -> Sequence[AbstractCapability]:
        """Build the capability list from the configured flags.

        Returned as a :class:`~collections.abc.Sequence` (not ``list``) so it is
        assignable to ``KompendAgentBase.extra_capabilities`` regardless of the
        agent's ``DepsT``: ``AbstractCapability`` is contravariant in its deps
        parameter, so the built capabilities target ``object`` and fit any agent.
        """
        _ctx = self.context_window or DEFAULT_CONTEXT_WINDOW
        caps: list[AbstractCapability] = []
        if self.memory:
            caps.append(Memory(store=InMemoryStore(), agent_name=self.memory_agent_name))
        if self.compaction:
            caps.append(SlidingWindowCompaction(max_fraction=0.6, context_window=self.context_window))
        if self.clamp:
            caps.append(ClampOversizedMessages(max_part_tokens=_ctx // 4))
        caps.extend(self.extra_capabilities)
        return caps
