"""Standard pydantic-ai-harness capability bundle for agentic agents.

Agentic agents in kompend share these harness capabilities:

- **Memory** - ``InMemoryStore`` by default: fresh per run (no cross-session
  persistence).  Lets the agent write/read a notebook across turns for
  long-running tasks.  **Opt-in per agent** (``memory=True``): the notebook is
  only useful when the agent's prompt tells the model to keep notes.  Offering
  it unconditionally invites small models to call ``read_memory`` for a file
  that does not exist - the tool answers with ``ModelRetry`` and, once the
  tool-retry budget is spent, the whole run dies with the node left
  unsummarized.  ``compile.llm.agent_memory`` (default on) turns it on for the
  stages where one agent instance handles many outputs.  A persistent notebook
  (``compile.llm.memory_store = "vault"``) is wired by passing
  ``memory_store=DirectoryMemoryStore(...)`` from the caller; the injected
  ``MEMORY.md`` index is capped by the capability, and topic files stay behind
  the memory tools.  ``memory_agent_name`` is the scope segment of those
  paths - the harness namespaces them, so stages never share a file.
- **SlidingWindowCompaction** — ``max_fraction=0.6`` of context window.
  Drops oldest turns when context fills up, avoiding token-limit failures
  on multi-turn tool agents.
- **ClampOversizedMessages** — ``max_part_tokens=context_window // 4``.
  Clamps runaway tool outputs that would blow a single turn.
- **StepPersistence** (``step_store``) — records every agent turn as a
  continuable snapshot in a :class:`~pydantic_ai_harness.step_persistence.SqliteStepStore`.
  The generation flow reads those snapshots to skip finished work and to
  resume an interrupted segment mid-conversation (replacing the old
  hand-rolled generation checkpoint).
- **ConversationSearch** (``search_history=True``, needs ``step_store``) —
  a BM25 ``search_conversation_history`` tool over the same store, scoped
  to the calling run's conversation, so a segment agent can recall what
  compaction dropped.
- **SystemReminders** (``reminder_content`` + ``reminder_interval``) —
  re-injects a short anchor instruction every N model requests. The text
  rides at the request tail behind a cache point, so the cached prompt
  prefix stays byte-identical.

``AgenticHarness`` centralizes the construction of these capabilities
from a known context window, with an ``extra_capabilities`` hook for
agent-specific additions (e.g. ``SummarizingCompaction`` for agents
that need full-context management instead of sliding window).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai_harness.compaction import ClampOversizedMessages, SlidingWindowCompaction
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.conversation_search import ConversationSearch, SnapshotHistorySource, SnapshotStore
from pydantic_ai_harness.memory import InMemoryStore, Memory, MemoryStore
from pydantic_ai_harness.step_persistence import StepPersistence, StepStore
from pydantic_ai_harness.system_reminders import Reminder, SystemReminders


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
    #: Scope segment for the notebook - the harness namespaces every memory
    #: path by it, so each stage owns ``<memory_dir>/<memory_agent_name>/``.
    #: Only meaningful when ``memory`` is on.
    memory_agent_name: str = "agentic"
    memory: bool = False
    #: Backend for the Memory capability. ``None`` (default) uses
    #: :class:`InMemoryStore` - fresh per stage run. A persistent notebook
    #: passes a :class:`~kompend.agents.vault_memory.DirectoryMemoryStore`
    #: rooted at the vault's agent-memory directory.
    memory_store: MemoryStore | None = None
    compaction: bool = True
    clamp: bool = True
    #: Persistent step store for the StepPersistence capability. ``None``
    #: (default) disables persistence. When set, every ``Agent.run`` call
    #: records continuable snapshots keyed by the ``run_id`` the caller
    #: passes to :meth:`KompendAgentBase.run`.
    step_store: StepStore | None = None
    #: Add a ConversationSearch tool over ``step_store`` (BM25 recall of the
    #: run's own conversation). Requires ``step_store``; runs must carry a
    #: ``conversation_id`` or the tool searches nothing.
    search_history: bool = False
    #: Anchor text re-injected every ``reminder_interval`` model requests
    #: (SystemReminders). Both fields must be set together; ``None`` disables.
    reminder_content: str | None = None
    reminder_interval: int | None = None
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
            caps.append(Memory(store=self.memory_store if self.memory_store is not None else InMemoryStore(), agent_name=self.memory_agent_name))
        if self.compaction:
            caps.append(SlidingWindowCompaction(max_fraction=0.6, context_window=self.context_window))
        if self.clamp:
            caps.append(ClampOversizedMessages(max_part_tokens=_ctx // 4))
        if self.step_store is not None:
            # agent_name stays None: the capability then derives its store
            # run_id verbatim from the caller-provided ``run_id`` (no
            # base64 namespacing), which is what the recovery scan matches.
            caps.append(StepPersistence(store=self.step_store, agent_name=None))
            if self.search_history:
                # The concrete stores (SqliteStepStore, FileStepStore) implement
                # list_snapshots; the harness StepStore protocol just omits it.
                caps.append(ConversationSearch(source=SnapshotHistorySource(cast("SnapshotStore", self.step_store)), scope="conversation"))
        if self.reminder_content is not None or self.reminder_interval is not None:
            if self.reminder_content is None or self.reminder_interval is None:
                raise ValueError("reminder_content and reminder_interval must be set together")
            caps.append(SystemReminders(reminders=(Reminder(content=self.reminder_content, interval=self.reminder_interval),)))
        caps.extend(self.extra_capabilities)
        return caps
