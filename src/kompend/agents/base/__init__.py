"""Public surface of the agent base package.

The package splits the former 1115-line ``agents/base.py`` by concern —
the agent class, model wiring, context/RLM capabilities, the deps
protocol, the retry/failure path and the delta-retry salvage hook (see
each module's docstring). ``kompend.agents.base`` keeps importing exactly
as before the split: every name the module ever exported is re-exported
here, so the 39 importing files (31 source, 7 tests, 1 docs page) need
no edits.
"""

from kompend.agents.base.agent import (
    _CONTEXT_PRESETS,
    _SLIDING_KEEP_MESSAGES,
    _SLIDING_MAX_MESSAGES,
    RLM_CONTEXT_PLACEHOLDER,
    UNSET_RETRIES,
    UNSET_TIMEOUT,
    AgentDeps,
    DepsT,
    KompendAgentBase,
    ResultT,
    _is_cache_rejection,
    resolve_retry_budget,
)

__all__ = [
    "RLM_CONTEXT_PLACEHOLDER",
    "UNSET_RETRIES",
    "UNSET_TIMEOUT",
    "_CONTEXT_PRESETS",
    "_SLIDING_KEEP_MESSAGES",
    "_SLIDING_MAX_MESSAGES",
    "AgentDeps",
    "DepsT",
    "KompendAgentBase",
    "ResultT",
    "_is_cache_rejection",
    "resolve_retry_budget",
]
