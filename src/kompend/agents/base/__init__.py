"""Public surface of the agent base package.

The package splits the former 1115-line ``agents/base.py`` by concern —
the agent class, model wiring, context/RLM capabilities, the deps
protocol, the retry/failure path and the delta-retry salvage hook (see
each module's docstring). ``kompend.agents.base`` keeps importing exactly
as before the split: every name the module ever exported is re-exported
here, so the 39 importing files (31 source, 7 tests, 1 docs page) need
no edits.
"""

from kompend.agents.base.agent import RLM_CONTEXT_PLACEHOLDER, KompendAgentBase
from kompend.agents.base.context import _CONTEXT_PRESETS
from kompend.agents.base.deps import (
    UNSET_RETRIES,
    UNSET_TIMEOUT,
    AgentDeps,
    DepsT,
    ResultT,
    resolve_retry_budget,
)
from kompend.agents.base.retries import _is_cache_rejection
from kompend.agents.base.spec import AgentSpec, agent_class, build_agent
from kompend.agents.context_manager import _SLIDING_COMPRESS_THRESHOLD, _SLIDING_KEEP_MESSAGES, _SLIDING_MAX_MESSAGES

__all__ = [
    "RLM_CONTEXT_PLACEHOLDER",
    "UNSET_RETRIES",
    "UNSET_TIMEOUT",
    "_CONTEXT_PRESETS",
    "_SLIDING_COMPRESS_THRESHOLD",
    "_SLIDING_KEEP_MESSAGES",
    "_SLIDING_MAX_MESSAGES",
    "AgentDeps",
    "AgentSpec",
    "DepsT",
    "KompendAgentBase",
    "ResultT",
    "_is_cache_rejection",
    "agent_class",
    "build_agent",
    "resolve_retry_budget",
]
