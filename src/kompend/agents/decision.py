"""DecisionAgent - asks a decision model what to do with a failed step.

The model is a TypeSafe decision model reached through pydantic-ai's
``typesafe`` provider, so it is configured like any other model
(``provider/model`` plus the provider key) instead of through a bespoke client.

Usage::

    from kompend.agents.decision import decide_error_action

    outcome = decide_error_action(config.decision, error=..., stage=..., attempt=1, max_attempts=3)
    # ErrorActionOutcome: ANSWERED carries .action, NOT_CONFIGURED means the
    # sidecar is off, UNAVAILABLE means no decision model answered.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Final

from kompend.agents.base import AgentSpec, KompendAgentBase
from kompend.compilation.adaptation import AdaptationSignal
from kompend.core.config._configs import DecisionConfig
from kompend.core.utils.exceptions import LLMTransientError
from kompend.llm.breaker import BreakerOpenError
from kompend.llm.decision import (
    DecisionState,
    ErrorActionOutcome,
    ErrorDecision,
    FigureRelevanceDecision,
    QueryIntentDecision,
    QueryIntentVerdict,
    UnitAdaptationOutcome,
    decision_model_string,
)
from kompend.llm.errors import is_quota_refusal
from kompend.llm.models import build_decision_model
from kompend.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DecisionDeps:
    """Deps for the error-triage decision.

    Attributes:
        stage: Pipeline stage that failed.
        error: The error text the pipeline would log.
        attempt: Which attempt just failed (1-based).
        max_attempts: How many attempts the caller would allow.
        context: Optional run context (model string, token usage).
        timeout: Per-call timeout in seconds.
        source_id: Log/provenance label.
    """

    stage: str
    error: str
    attempt: int
    max_attempts: int
    context: str = ""
    timeout: float = 30.0
    source_id: str = "decision"


SPEC_DECISION: Final[AgentSpec] = AgentSpec(
    name="decision",
    result_type=ErrorDecision,
    deps_type=DecisionDeps,
    template_name="error_triage.md.j2",
    system_template_name="error_triage_system.md.j2",
    template_prefix="decision/",
    temperature=0.0,
    model_builder=build_decision_model,
)


class DecisionAgent(KompendAgentBase[ErrorDecision, DecisionDeps]):
    """PydanticAI agent backed by a decision model (TypeSafe Jev)."""

    spec: ClassVar[AgentSpec | None] = SPEC_DECISION
    temperature: float | None = 0.0


SPEC_ADAPTATION: Final[AgentSpec] = AgentSpec(
    name="adaptation",
    result_type=ErrorDecision,
    deps_type=DecisionDeps,
    template_name="adaptation_triage.md.j2",
    system_template_name="error_triage_system.md.j2",
    template_prefix="decision/",
    temperature=0.0,
    model_builder=build_decision_model,
)


@dataclass
class QueryIntentDeps:
    """Deps for the query-intent decision.

    Attributes:
        query: The user's message, verbatim.
        timeout: Per-call timeout in seconds.
        source_id: Log/provenance label (satisfies the agent deps protocol).
    """

    query: str
    timeout: float = 10.0
    source_id: str = "query_intent"


class AdaptationAgent(DecisionAgent):
    """Decision model for ambiguous generation failures (Loop 2).

    Same verdict schema as error triage plus the shrink remedy: the model
    sees the compact failure record and decides whether one more halving
    is worth trying.
    """

    spec: ClassVar[AgentSpec | None] = SPEC_ADAPTATION


SPEC_QUERY_INTENT: Final[AgentSpec] = AgentSpec(
    name="query_intent",
    result_type=QueryIntentDecision,
    deps_type=QueryIntentDeps,
    template_name="query_intent.md.j2",
    system_template_name="query_intent_system.md.j2",
    template_prefix="decision/",
    temperature=0.0,
    model_builder=build_decision_model,
)


@dataclass
class FigureRelevanceDeps:
    """Deps for the figure-relevance decision.

    Attributes:
        context: What surrounds the figure — the same rendering the caption
            prompt gets (section, neighbouring text, contextual summary).
            The figure image itself is never sent; the judgement is made
            from the surrounding text alone.
        source_id: Log/provenance label.
        timeout: Per-call timeout in seconds.
    """

    context: str = ""
    source_id: str = "vlm-relevance"
    timeout: float = 10.0


class QueryIntentAgent(KompendAgentBase[QueryIntentDecision, QueryIntentDeps]):
    """Decision model that classifies what a user query is asking for."""

    spec: ClassVar[AgentSpec | None] = SPEC_QUERY_INTENT
    temperature: float | None = 0.0


def decide_query_intent(
    config: DecisionConfig | None,
    *,
    query: str,
) -> QueryIntentVerdict | None:
    """Ask the decision model what kind of message *query* is.

    Returns the verdict, or ``None`` when no decision model is configured or
    the call failed - callers then keep their deterministic classifier.
    """
    model_string = decision_model_string(config)
    if model_string is None or config is None:
        return None

    agent = QueryIntentAgent(model_string=model_string, timeout=config.timeout, retries=0)
    try:
        decision = agent.run(QueryIntentDeps(query=query, timeout=config.timeout))
    except Exception as exc:  # ponytail: a sidecar failure must never break a query
        logger.warning("Decision model unavailable (%s) - falling back to the deterministic classifier", exc)
        return None
    logger.info("Decision model: query intent %s", decision.intent)
    return decision.intent


def decide_error_action(
    config: DecisionConfig | None,
    *,
    error: str,
    stage: str,
    attempt: int,
    max_attempts: int,
    context: str = "",
    fallback_models: Sequence[str] | None = None,
) -> ErrorActionOutcome:
    """Ask the decision model what to do with a failed LLM call.

    Returns an :class:`ErrorActionOutcome`: ``ANSWERED`` carries the decided
    action, ``NOT_CONFIGURED`` means the sidecar is off (the caller keeps
    its deterministic fallback, exactly as before), and ``UNAVAILABLE``
    means a decision model is configured but none answered — the caller
    must degrade the run and say so, not silently fall back.

    A quota-dead primary fails over to the configured fallback chain
    (``compile.llm.fallback_chain`` — typically a local model) before the
    call is declared unavailable.
    """
    model_string = decision_model_string(config)
    if model_string is None or config is None:
        return ErrorActionOutcome(state=DecisionState.NOT_CONFIGURED)

    models = [model_string, *_failover_candidates(model_string, fallback_models)]
    last_exc: Exception | None = None
    for position, candidate in enumerate(models):
        agent = DecisionAgent(model_string=candidate, timeout=config.timeout, retries=0)
        try:
            decision = agent.run(
                DecisionDeps(
                    stage=stage,
                    error=error,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    context=context,
                    timeout=config.timeout,
                )
            )
        except Exception as exc:  # ponytail: a sidecar failure must never break the pipeline
            last_exc = exc
            if _is_dead_model_error(exc) and position + 1 < len(models):
                logger.warning(
                    "Decision model '%s' is dead (%s) - failing over to '%s'",
                    candidate,
                    exc,
                    models[position + 1],
                )
                continue
            logger.warning("Decision model unavailable (%s) - falling back to the deterministic classification", exc)
            return ErrorActionOutcome(state=DecisionState.UNAVAILABLE)
        if position > 0:
            logger.info("Decision model failed over to '%s'", candidate)
        logger.info("Decision model: %s for %s after attempt %d/%d", decision.action, stage, attempt, max_attempts)
        return ErrorActionOutcome(state=DecisionState.ANSWERED, action=decision.action)
    logger.warning("Decision model unavailable (%s) - falling back to the deterministic classification", last_exc)
    return ErrorActionOutcome(state=DecisionState.UNAVAILABLE)


def decide_unit_adaptation(
    config: DecisionConfig | None,
    *,
    signal: AdaptationSignal,
    stage: str,
    max_adaptations: int,
    fallback_models: Sequence[str] | None = None,
) -> UnitAdaptationOutcome:
    """Ask the decision model whether a declined unit deserves one more split.

    Returns a :class:`UnitAdaptationOutcome`: ``ANSWERED`` carries the
    model's verdict, ``NOT_CONFIGURED`` means the sidecar is off, and
    ``UNAVAILABLE`` means a decision model is configured but none answered.
    Consumes only the compact failure record, never request content. A
    quota-dead primary fails over to the configured fallback chain before
    the call is declared unavailable.
    """
    model_string = decision_model_string(config)
    if model_string is None or config is None:
        return UnitAdaptationOutcome(state=DecisionState.NOT_CONFIGURED)

    models = [model_string, *_failover_candidates(model_string, fallback_models)]
    last_exc: Exception | None = None
    for position, candidate in enumerate(models):
        agent = AdaptationAgent(model_string=candidate, timeout=config.timeout, retries=0)
        try:
            decision = agent.run(
                DecisionDeps(
                    stage=stage,
                    error=signal.last_error,
                    attempt=signal.adaptations_used + 1,
                    max_attempts=max_adaptations,
                    context=signal.describe(),
                    timeout=config.timeout,
                )
            )
        except Exception as exc:  # ponytail: a sidecar failure must never break the pipeline
            last_exc = exc
            if _is_dead_model_error(exc) and position + 1 < len(models):
                logger.warning(
                    "Decision model '%s' is dead (%s) - failing over to '%s'",
                    candidate,
                    exc,
                    models[position + 1],
                )
                continue
            logger.warning("Decision model unavailable (%s) - falling back to the deterministic path", exc)
            return UnitAdaptationOutcome(state=DecisionState.UNAVAILABLE)
        if position > 0:
            logger.info("Decision model failed over to '%s'", candidate)
        return UnitAdaptationOutcome(state=DecisionState.ANSWERED, decision=decision)
    logger.warning("Decision model unavailable (%s) - falling back to the deterministic path", last_exc)
    return UnitAdaptationOutcome(state=DecisionState.UNAVAILABLE)


def _failover_candidates(primary: str, fallback_models: Sequence[str] | None) -> list[str]:
    """Fallback-chain entries worth asking after the primary decision model died.

    The primary is skipped (asking it again would refuse instantly) and
    empties are dropped; order is the operator's chain order.
    """
    if not fallback_models:
        return []
    return [model for model in fallback_models if model and model != primary]


def _is_dead_model_error(exc: BaseException) -> bool:
    """Whether *exc* means the model is dead rather than the question bad.

    Quota/breaker refusals and an exhausted transient budget say the model
    cannot answer right now, so a fallback chain entry may still answer. A
    config error (bad model string, unknown provider) or a model error says
    the question or the configuration is wrong — failing over would mask it.
    """
    if isinstance(exc, BreakerOpenError):
        return True
    if isinstance(exc, LLMTransientError):
        return True
    return is_quota_refusal(exc)


SPEC_FIGURE_RELEVANCE: Final[AgentSpec] = AgentSpec(
    name="figure_relevance",
    result_type=FigureRelevanceDecision,
    deps_type=FigureRelevanceDeps,
    template_name="figure_relevance.md.j2",
    system_template_name="figure_relevance_system.md.j2",
    template_prefix="decision/",
    temperature=0.0,
    model_builder=build_decision_model,
)


class FigureRelevanceAgent(KompendAgentBase[FigureRelevanceDecision, FigureRelevanceDeps]):
    """Decision-model agent that judges whether a figure deserves a caption.

    Judges from the surrounding text only — the figure image itself is
    never sent, so the decision sidecar's text-only transport is not a
    limitation. The prompt is conservative: only a context that clearly
    indicates decoration answers ``decorative``; anything ambiguous
    answers ``informative`` so no caption is lost.
    """

    spec: ClassVar[AgentSpec | None] = SPEC_FIGURE_RELEVANCE
