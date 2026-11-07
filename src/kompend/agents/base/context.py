"""Context-management and RLM capability assembly for the agent base.

Turns the ``compile.context_manager`` / ``compile.llm.rlm`` config groups
into pydantic-ai capabilities: compaction preset →
:class:`SlidingWindowCompaction` or the fallback summarizer, RLM →
:class:`RLMCapability` with its sub-model. Both read only construction
state off the agent (preset, window, rlm group, usage tracker), so they
live here as free functions; the agent class keeps thin delegating
methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model
from pydantic_ai_harness.compaction import ContextUsage, SlidingWindowCompaction

from kompend.agents.base.deps import DepsT, ResultT
from kompend.agents.context_manager import _SLIDING_KEEP_MESSAGES, _SLIDING_MAX_MESSAGES, build_fallback_compaction
from kompend.agents.kompend_model import KompendModel
from kompend.agents.rlm import _DEFAULT_NODE_BUDGET, RLMCapability
from kompend.core.config._configs import ContextManagerPreset, RlmConfig
from kompend.llm.models import build_model

if TYPE_CHECKING:
    from kompend.agents.base.agent import KompendAgentBase

#: Compaction presets: preset -> (compress threshold, messages kept verbatim).
#: Keys are the validated :class:`~kompend.core.config.ContextManagerPreset`
#: members — a preset absent here fails the ``[]`` lookup loudly instead of
#: degrading to a different compaction shape.
_CONTEXT_PRESETS: Final[dict[ContextManagerPreset, tuple[float, int]]] = {
    ContextManagerPreset.DEFAULT: (0.9, 0),
    ContextManagerPreset.AGGRESSIVE: (0.7, 5),
}


def resolve_context_capability(
    agent: KompendAgentBase[ResultT, DepsT],
    model: Model,
) -> AbstractCapability[DepsT] | None:
    """Build the context-management capability from the configured preset.

    Returns ``None`` when context management is disabled, in which case the
    pydantic-ai Agent is constructed without ``capabilities``.

    Args:
        agent: The agent whose ``context_manager`` / ``context_window``
            construction state decides the shape.
        model: The top-level (failover) model — reused as the
            summarization model so compressions run on the same chain.
    """
    setting = agent.context_manager
    if setting is False or setting is None:
        return None
    preset = ContextManagerPreset.DEFAULT if setting is True else setting

    if preset is ContextManagerPreset.SLIDING:
        return SlidingWindowCompaction(
            max_messages=_SLIDING_MAX_MESSAGES,
            keep_messages=_SLIDING_KEEP_MESSAGES,
        )
    threshold, keep = _CONTEXT_PRESETS[preset]
    return build_fallback_compaction(
        context_window=agent.context_window,
        compress_threshold=threshold,
        keep_messages=keep,
        model=model,
    )


def resolve_rlm_capability(
    agent: KompendAgentBase[ResultT, DepsT],
    model: Model,
) -> RLMCapability | None:
    """Build the RLM capability from the ``compile.llm.rlm`` group.

    Returns ``None`` when RLM is disabled (``rlm`` unset or
    ``enable = false``). The sub-model defaults to the agent's top-level
    model (same failover chain); ``sub_model`` resolves via
    ``build_model`` and is wrapped in :class:`KompendModel` for usage
    tracking.

    Args:
        agent: The agent whose ``rlm`` group and ``rlm_context_field``
            construction state feed the capability.
        model: The top-level (failover) model — reused as the
            ``llm_query`` sub-model when ``sub_model`` is unset.
    """
    rlm = agent.rlm
    if not isinstance(rlm, RlmConfig) or not rlm.enable:
        return None
    max_tool_calls = rlm.max_tool_calls
    code_timeout = rlm.code_timeout
    sub_model_str = rlm.sub_model
    depth = rlm.max_depth
    node_budget = rlm.node_budget
    if sub_model_str:
        sub_model: Model = KompendModel(
            build_model(sub_model_str),
            usage_tracker=agent.usage_tracker,
        )
    else:
        sub_model = model
    return RLMCapability(
        context_field=agent.rlm_context_field,
        code_timeout=code_timeout,
        sub_model=sub_model,
        max_tool_calls=max_tool_calls,
        main_model=model,
        depth=depth,
        node_budget=node_budget or _DEFAULT_NODE_BUDGET,
    )


def on_usage_update(agent: KompendAgentBase[ResultT, DepsT], usage: ContextUsage) -> None:
    """Token-usage snapshot from the compaction capability → UsageTracker."""
    if agent.usage_tracker is not None:
        agent.usage_tracker.record_context_usage(usage.used_tokens, usage.window_tokens)
