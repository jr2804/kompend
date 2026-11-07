"""Model wiring for the agent base: top-level model and output ceilings.

``build_top_level_model`` builds the failover chain wrapped in
:class:`KompendModel` for usage tracking; the ceiling helpers derive the
protocol-mandated ``max_tokens`` per request (window minus prompt minus
margin, clamped by the catalog's output limit). All of them read only
construction state off the agent, so they live here as free functions;
the agent class keeps thin delegating methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel

from kompend.agents.kompend_model import KompendModel
from kompend.llm.models import build_model, resolve_model_limits

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.agent import AgentModelSettings
    from pydantic_ai.settings import ModelSettings

    from kompend.agents.base.agent import KompendAgentBase

#: Share of the window kept free on top of the estimated prompt when a
#: protocol demands an explicit output ceiling.
_CEILING_MARGIN_FRACTION: float = 0.05

#: Floor for that margin, in tokens.
_CEILING_MARGIN_MIN: int = 256

#: Chars-per-token used when the agent has no ``compile.llm`` config group.
_FALLBACK_CHARS_PER_TOKEN: float = 4.0


def build_top_level_model(agent: KompendAgentBase) -> Model:
    """Build the model (with failover + kompend wrapper) for the Agent."""
    primary = KompendModel(
        build_model(agent.model_string),
        usage_tracker=agent.usage_tracker,
        agent=agent,
        reasoning_effort=agent.reasoning_effort,
    )
    if not agent.fallback_models:
        return primary

    return FallbackModel(
        primary,
        *(
            KompendModel(
                build_model(m),
                usage_tracker=agent.usage_tracker,
                agent=agent,
                reasoning_effort=agent.reasoning_effort,
            )
            for m in agent.fallback_models
        ),
    )


def ceiling_model_settings(
    agent: KompendAgentBase,
    base: ModelSettings,
) -> AgentModelSettings:
    """Per-request settings that add the protocol-mandated ``max_tokens``.

    Only protocols that reject an unset value use this (see
    :func:`kompend.llm.models.requires_output_ceiling`); everything else
    keeps the plain dict and sends no ceiling at all.
    """

    def _settings(ctx: RunContext) -> ModelSettings:
        ceiling = agent._output_ceiling(ctx.messages)
        if ceiling is None:
            return base
        return {**base, "max_tokens": ceiling}

    return _settings


def output_ceiling(agent: KompendAgentBase, messages: Sequence[ModelMessage]) -> int | None:
    """Tokens this request may generate, or ``None`` when nothing is sent.

    The value is the physical maximum, not a configured cap: the model's
    window minus the prompt about to be sent, minus a margin, clamped by
    the model's own output limit when the catalog knows it.
    """
    window = agent.context_window
    if not window:
        return None
    prompt_tokens = agent._estimate_prompt_tokens(messages)
    if prompt_tokens is None:
        return None
    margin = max(_CEILING_MARGIN_MIN, int(window * _CEILING_MARGIN_FRACTION))
    available = window - prompt_tokens - margin
    limit = agent._model_max_output()
    if limit is not None:
        available = min(available, limit)
    return max(1, available)


def estimate_prompt_tokens(agent: KompendAgentBase, messages: Sequence[ModelMessage]) -> int | None:
    """Rough token count of the request payload, ``None`` when unknown."""
    try:
        payload = ModelMessagesTypeAdapter.dump_json(list(messages))
    except Exception:  # noqa: BLE001 - an unmeasurable prompt must not break the call
        return None
    chars_per_token = agent.llm.default_chars_per_token if agent.llm is not None else _FALLBACK_CHARS_PER_TOKEN
    return max(1, int(len(payload) / chars_per_token))


def model_max_output(agent: KompendAgentBase) -> int | None:
    """The model's own output limit from the catalog, or ``None`` if unknown."""
    try:
        _, max_output = resolve_model_limits(agent.model_string)
    except (KeyError, LookupError, ValueError):
        return None
    return max_output or None
