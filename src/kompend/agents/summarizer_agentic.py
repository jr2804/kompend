"""AgenticSummarizerAgent — multi-turn summarizer with tools.

One session covers all internal nodes of a chunk tree: the agent pulls
nodes via ``next_node()`` and records each summary via
``summarize_node(node_id, summary)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic_ai import UsageLimits

from kompend.agents.agentic_harness import AgenticHarness
from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT, KompendAgentBase, resolve_retry_budget
from kompend.agents.types import SummarizerDeps
from kompend.agents.vault_memory import DirectoryMemoryStore
from kompend.compilation.summarizer_tools import build_summarizer_tools
from kompend.core.config._configs import LLMConfig, LlmMemoryStore, RlmConfig


@dataclass
class AgenticSummarizerAgent(KompendAgentBase[str, SummarizerDeps]):
    """PydanticAI agent that summarizes chunk-tree nodes via tools."""

    template_name: ClassVar[str] = "summarizer_agentic_prompt.md.j2"
    system_template_name: ClassVar[str] = "summarizer_agentic_system.md.j2"
    template_prefix: ClassVar[str] = "chunking/"

    #: LLM requests budgeted per payload node. The pipeline sets this from
    #: ``budget.summary_requests_per_node`` — small local models routinely
    #: need more than the historical 2 (prose turns, id retries) and starve
    #: otherwise.
    requests_per_node: int = 2

    def _usage_limits_for(
        self,
        deps: SummarizerDeps,
        *,
        requests_per_node: int | None = None,
        headroom_frac: float = 1.1,
        session_overhead: int = 2,
    ) -> UsageLimits | None:
        """Honest request budget for one tree session.

        The tool loop costs ``requests_per_node`` per node (one
        ``next_node()`` turn plus one ``summarize_node()`` turn), but a
        session also pays a fixed price: one ``next_node()`` probe that finds
        no work left, and the closing turn.  Without that constant a
        single-node session is budgeted at exactly its own cost and dies on
        the closing turn — which is how a one-node payload ends up reported
        as ``request_limit of 3``.

        The per-node factor defaults to the instance value (the pipeline
        passes ``budget.summary_requests_per_node``); an explicit kwarg
        overrides it for probing. ``session_overhead`` carries the constant;
        ``headroom_frac`` then covers tool/output retries. The pydantic-ai
        implicit default of 50 requests aborts trees with more than ~24
        eligible nodes.
        """
        if self._rlm_usage_limits is not None:
            return self._rlm_usage_limits
        per_node = self.requests_per_node if requests_per_node is None else requests_per_node
        planned = len(deps.nodes) * per_node + session_overhead
        return UsageLimits(request_limit=math.ceil(planned * headroom_frac))


def make_agentic_summarizer_agent(
    model_string: str,
    *,
    timeout: float = UNSET_TIMEOUT,
    skills: dict[str, str] | None = None,
    retries: int = UNSET_RETRIES,
    output_retries: int | None = None,
    context_window: int | None = None,
    requests_per_node: int = 2,
    rlm: RlmConfig | None = None,
    llm: LLMConfig | None = None,
    memory_dir: Path | None = None,
) -> AgenticSummarizerAgent:
    """Build an :class:`AgenticSummarizerAgent` with tools + harness capabilities.

    Registers the ``next_node()`` / ``summarize_node()`` tools the system prompt
    instructs the model to call, and injects the standard agentic capabilities
    via :class:`AgenticHarness`.

    ``memory_dir`` is the vault's agent-memory directory; it only takes effect
    when ``llm.memory_store`` is ``vault`` — the notebook then persists as a
    ``MEMORY.md`` index plus topic files instead of dying with the process.
    """
    agent_retries = resolve_retry_budget(llm, retries=retries, output_retries=output_retries)
    memory_store: DirectoryMemoryStore | None = None
    if memory_dir is not None and llm is not None and llm.memory_store is LlmMemoryStore.VAULT:
        memory_store = DirectoryMemoryStore(memory_dir)
    harness = AgenticHarness(
        context_window=context_window,
        memory_agent_name="summarizer",
        memory=bool(llm and llm.agent_memory),
        memory_store=memory_store,
    )
    return AgenticSummarizerAgent(
        model_string=model_string,
        timeout=timeout,
        skills=skills or {},
        retries=agent_retries,
        context_window=context_window,
        requests_per_node=requests_per_node,
        rlm=rlm,
        llm=llm,
        tools=build_summarizer_tools(),
        extra_capabilities=harness.build_capabilities(),
    )
