"""DecisionAgent - asks a decision model what to do with a failed step.

The model is a TypeSafe decision model reached through pydantic-ai's
``typesafe`` provider, so it is configured like any other model
(``provider/model`` plus the provider key) instead of through a bespoke client.

Usage::

    from kompend.agents.decision import decide_error_action

    action = decide_error_action(config.decision, error=..., stage=..., attempt=1, max_attempts=3)
    # "retry" | "skip" | "abort" | None (no decision model configured)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from kompend.agents.base import KompendAgentBase
from kompend.core.config._configs import DecisionConfig
from kompend.llm.decision import ErrorAction, ErrorDecision, decision_model_string
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


class DecisionAgent(KompendAgentBase[ErrorDecision, DecisionDeps]):
    """PydanticAI agent backed by a decision model (TypeSafe Jev)."""

    template_name: ClassVar[str] = "error_triage.md.j2"
    system_template_name: ClassVar[str] = "error_triage_system.md.j2"
    template_prefix: ClassVar[str] = "decision/"
    temperature: float | None = 0.0


def decide_error_action(
    config: DecisionConfig | None,
    *,
    error: str,
    stage: str,
    attempt: int,
    max_attempts: int,
    context: str = "",
) -> ErrorAction | None:
    """Ask the decision model what to do with a failed LLM call.

    Returns the decided action, or ``None`` when no decision model is
    configured or the call failed - callers then keep their deterministic
    fallback (see :func:`kompend.llm.errors.is_transient_error`).
    """
    model_string = decision_model_string(config)
    if model_string is None or config is None:
        return None

    agent = DecisionAgent(model_string=model_string, timeout=config.timeout, retries=0)
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
        logger.warning("Decision model unavailable (%s) - falling back to the deterministic classification", exc)
        return None
    logger.info("Decision model: %s for %s after attempt %d/%d", decision.action, stage, attempt, max_attempts)
    return decision.action
