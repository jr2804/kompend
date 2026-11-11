"""The construction-state contract the agent-base helpers read off an agent.

``agent.py`` sits at the top of this package: it imports the helper modules
(``models``, ``context``, ``retries``, ``delta_retry``), so none of them may
import it back. Each helper therefore types its argument as the narrow
protocol below instead of the class it cannot name at runtime.

The protocols are per concern rather than one shared surface, so a helper
that starts reading new state has to declare it in the protocol it owns
instead of silently widening every other helper's contract. The agent class
satisfies all of them structurally, so the delegating methods on
:class:`KompendAgentBase` pass ``self`` with no cast, and a helper is
testable with a stub that carries only the members listed here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic_ai import UsageLimits
from pydantic_ai.agent.abstract import AgentRetries
from pydantic_ai.messages import ModelMessage

from kompend.core.config._configs import (
    CompactionConfig,
    ContextManagerPreset,
    LLMConfig,
    ReasoningEffort,
    RlmConfig,
)
from kompend.llm.debug import _DebugSource
from kompend.llm.usage import UsageTracker
from kompend.types import JSONValue


class ContextHost(Protocol):
    """What :mod:`kompend.agents.base.context` reads to build capabilities."""

    #: ``compile.context_manager`` (or the explicit constructor value).
    context_manager: ContextManagerPreset | bool | None
    #: ``compile.llm.compaction`` — thresholds, judge and keep sizes.
    compaction: CompactionConfig
    #: Resolved context window, ``None`` when the catalog is to decide.
    context_window: int | None
    #: ``compile.llm.rlm`` group, ``None`` when RLM is off.
    rlm: RlmConfig | None
    #: Deps field the RLM tool exposes as its working context.
    rlm_context_field: str
    #: Per-run token accounting; ``None`` in tests and dry runs.
    usage_tracker: UsageTracker | None

    @property
    def compact_on_request(self) -> bool:
        """Class flag: may the model compact its own history mid-run?"""


class ModelHost(_DebugSource, Protocol):
    """What :mod:`kompend.agents.base.models` reads to build the model.

    Extends the debug contract :class:`kompend.agents.kompend_model.KompendModel`
    already asks of the agent it wraps (``stage``, ``current_source_id``,
    ``_last_debug_file``), so the wrapper's debug hook is satisfied by the
    same object the wiring passes in.
    """

    model_string: str
    fallback_models: list[str] | None
    reasoning_effort: ReasoningEffort | None
    llm: LLMConfig | None
    context_window: int | None
    usage_tracker: UsageTracker | None

    def _estimate_prompt_tokens(self, messages: Sequence[ModelMessage]) -> int | None:
        """Rough token count of a request payload, ``None`` when unknown."""

    def _model_max_output(self) -> int | None:
        """The model's own output limit, ``None`` when the catalog is silent."""

    def _output_ceiling(self, messages: Sequence[ModelMessage]) -> int | None:
        """Tokens one request may generate, ``None`` when nothing is sent."""


class RetryHost(Protocol):
    """What :mod:`kompend.agents.base.retries` reads to run a request."""

    model_string: str
    llm: LLMConfig | None
    retries: int | AgentRetries
    usage_tracker: UsageTracker | None
    #: The built pydantic-ai agent; ``None`` until the first run.
    _agent: Any
    #: Whether this agent still asks the provider for prompt caching.
    _prompt_cache: bool
    _last_debug_file: Path | None
    #: RLM preset limits, when the agent configured them.
    _rlm_usage_limits: UsageLimits | None

    def _build_agent(self) -> Any:
        """Build (or rebuild) the pydantic-ai agent."""

    def _transient_retries(self) -> int:
        """Configured whole-request retries after a transient failure."""

    def _transient_backoff(self) -> float:
        """Configured first backoff (seconds) for a transient retry."""

    def _usage_limits_for(self, deps: Any) -> UsageLimits | None:
        """Per-run request budget for ``deps``."""


class DeltaRetryHost(Protocol):
    """What :mod:`kompend.agents.base.delta_retry` reads to salvage an answer.

    The hook accumulates the valid parts of a partially valid structured
    output per ``run_step`` and merges them on the next attempt, so it needs
    the agent's state dictionary and the schema it is validating against.
    """

    #: Structured-output type the agent was built with.
    result_type: type
    #: Per-``run_step`` merge state of the current run.
    _delta_retry_state: dict[int, JSONValue]
