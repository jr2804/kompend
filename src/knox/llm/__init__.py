"""Centralised LLM client for KNOX.

All LLM calls in the pipeline go through :class:`LLMClient` from
:mod:`knox.llm.client`. The client wraps litellm with:

- A typed error hierarchy (:mod:`knox.llm.errors`)
- Configurable fallback chains (:mod:`knox.llm.failover`)
- Per-call observability (model_used, attempts, latency)

This module is the single source of truth for LLM semantics in KNOX.
Business modules must not import litellm directly.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _install_litellm_agentic_loop_compat() -> None:
    try:
        from litellm.types.integrations.custom_logger import AgenticLoopPlan  # noqa: F401
    except ImportError:
        try:
            from pydantic import BaseModel

            class AgenticLoopPlan(BaseModel):  # type: ignore[no-redef]
                pass

            from litellm.types.integrations import custom_logger as _custom_logger_module
            setattr(_custom_logger_module, "AgenticLoopPlan", AgenticLoopPlan)
        except Exception as exc:  # pragma: no cover - best-effort compat
            logger.debug("Failed to install litellm AgenticLoopPlan compat: %s", exc)


_install_litellm_agentic_loop_compat()

from knox.llm.client import LLMClient, LLMResult
from knox.llm.errors import (
    ConfigurationError,
    LLMError,
    PermanentError,
    TransientError,
    classify_litellm_error,
)
from knox.llm.failover import FailoverChain

__all__ = [
    "ConfigurationError",
    "FailoverChain",
    "LLMClient",
    "LLMError",
    "LLMResult",
    "PermanentError",
    "TransientError",
    "classify_litellm_error",
]
