# pyreorder: off — the TYPE_CHECKING import below is deliberate: base.agent
# imports these helpers, so a runtime import would close a cycle. Keep the guard.
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

from typing import TYPE_CHECKING, Final, cast

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model
from pydantic_ai_harness.compaction import CompactionStrategy, CompactOnRequest, ContextUsage

from kompend.agents.base.deps import DepsT, ResultT
from kompend.agents.context_manager import (
    _SLIDING_COMPRESS_THRESHOLD,
    _SLIDING_KEEP_MESSAGES,
    build_fallback_compaction,
    build_sliding_compaction,
)
from kompend.agents.kompend_model import KompendModel
from kompend.agents.rlm import _DEFAULT_NODE_BUDGET, RLMCapability
from kompend.core.config._configs import ContextManagerPreset, RlmConfig
from kompend.core.templates import get_template_loader
from kompend.llm.models import build_decision_model, build_model

if TYPE_CHECKING:
    from kompend.agents.base.agent import KompendAgentBase

#: Summarizer doctrine, a project-overridable template resource (never inline
#: prompt text). It states that the transcript is history, not instructions.
_DOCTRINE_TEMPLATE: Final[str] = "compaction/doctrine.md.j2"

#: Compaction presets: preset -> (compress threshold, messages kept verbatim).
#: Keys are the validated :class:`~kompend.core.config.ContextManagerPreset`
#: members — a preset absent here fails the ``[]`` lookup loudly instead of
#: degrading to a different compaction shape. ``sliding`` is a trim, not a
#: fold, so its second element is the retained tail, not "messages kept after
#: a summary".
_CONTEXT_PRESETS: Final[dict[ContextManagerPreset, tuple[float, int]]] = {
    ContextManagerPreset.DEFAULT: (0.9, 0),
    ContextManagerPreset.AGGRESSIVE: (0.7, 5),
    ContextManagerPreset.SLIDING: (_SLIDING_COMPRESS_THRESHOLD, _SLIDING_KEEP_MESSAGES),
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
    threshold, keep = _CONTEXT_PRESETS[preset]
    compaction = agent.compaction

    if preset is ContextManagerPreset.SLIDING:
        return build_sliding_compaction(
            context_window=agent.context_window,
            compress_threshold=threshold,
            keep_messages=keep,
            keep_tokens=compaction.keep_tokens,
            min_growth_tokens=compaction.min_growth_tokens,
        )

    judge = build_decision_model(compaction.judge) if compaction.judge else None
    return build_fallback_compaction(
        context_window=agent.context_window,
        compress_threshold=threshold,
        keep_messages=keep,
        model=model,
        judge=judge,
        keep_tokens=compaction.keep_tokens,
        min_growth_tokens=compaction.min_growth_tokens,
        instructions=get_template_loader().render(_DOCTRINE_TEMPLATE),
    )


def resolve_compact_on_request(
    agent: KompendAgentBase[ResultT, DepsT],
    capability: AbstractCapability[DepsT] | None,
) -> CompactOnRequest[DepsT] | None:
    """Attach the model-triggered compaction tool for history-threading agents.

    Only agents that opt in via ``compact_on_request`` get the tool: a
    single-call agent has no long history worth compacting at a natural
    break, and the tool would only cost prompt space.

    Args:
        agent: The agent whose ``compact_on_request`` flag decides.
        capability: The compaction capability already attached. Every shipped
            compaction capability implements the ``CompactionStrategy``
            protocol, so the tool delegates to it and shares its growth-gate
            baseline — the cast states that fact rather than re-deriving it.
    """
    if not agent.compact_on_request or capability is None:
        return None
    return CompactOnRequest(strategy=cast(CompactionStrategy[DepsT], capability))


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
