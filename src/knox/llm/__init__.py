"""Centralised LLM client for KNOX.

All LLM calls in the pipeline go through :class:`LLMClient` from
:mod:`knox.llm.client`. The client wraps litellm with:

- A typed error hierarchy (:mod:`knox.llm.errors`)
- Configurable fallback chains (:mod:`knox.llm.failover`)
- An exception classifier (:mod:`knox.llm.classifier`)
- Per-call observability (model_used, attempts, latency)

This module is the single source of truth for LLM semantics in KNOX.
Business modules must not import litellm directly.
"""

from __future__ import annotations

import logging

from knox.llm.classifier import classify_litellm_error
from knox.llm.client import LLMClient, LLMResult
from knox.llm.errors import (
    AmbiguousError,
    ConfigurationError,
    LLMError,
    PermanentError,
    TransientError,
)
from knox.llm.failover import FailoverChain

logger = logging.getLogger(__name__)

try:
    from litellm.types.integrations.custom_logger import AgenticLoopPlan  # noqa: F401
except ImportError:
    try:
        from pydantic import BaseModel

        class AgenticLoopPlan(BaseModel):  # type: ignore[no-redef]
            pass

        from litellm.types.integrations import custom_logger as _custom_logger_module

        _custom_logger_module.AgenticLoopPlan = AgenticLoopPlan
    except Exception as _exc:  # pragma: no cover - best-effort compat
        logger.debug("Failed to install litellm AgenticLoopPlan compat: %s", _exc, exc_info=True)

__all__ = [
    "AmbiguousError",
    "ConfigurationError",
    "FailoverChain",
    "LLMClient",
    "LLMError",
    "LLMResult",
    "PermanentError",
    "TransientError",
    "classify_litellm_error",
]
