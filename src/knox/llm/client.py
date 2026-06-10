"""Shared LiteLLM-backed client used by every business module.

:class:`LLMClient` is the single call site for LLM completions in
KNOX. Modules pass a ``role`` (e.g. ``"resolve"``, ``"wicer.judge"``)
and the client picks the right model + fallback chain for that role.

This module wraps litellm — business modules must not import litellm
directly. Doing so reintroduces the silent fallback antipattern this
client exists to eliminate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import litellm

from knox.llm.errors import LLMError, PermanentError
from knox.llm.failover import CallDiagnostics, FailoverChain

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResult:
    """Successful completion returned by :meth:`LLMClient.complete`.

    ``content`` is the raw text content of the assistant message.
    ``json_data`` is the parsed JSON object when ``response_format``
    requested one (``None`` otherwise). ``diagnostics`` carries the
    failover-chain observability data.
    """

    content: str
    json_data: dict[str, Any] | None
    diagnostics: CallDiagnostics


@dataclass
class LLMClient:
    """Shared completion client — requires an explicit model choice.

    The client is configured with a **required** default model and an
    optional fallback chain. Per-role overrides are looked up in
    ``role_models``; a role with no entry uses the default chain.

    There is intentionally no default model — callers must choose one
    appropriate for their environment, budget, and task.

    Parameters
    ----------
    default_model:
        **Required.** Model name in litellm format
        (``"provider/model"``), used as the primary when no
        role-specific override exists.
    fallback_chain:
        Optional ordered list of model names tried when the primary
        fails. Empty means "single-model, single-retry" — backwards
        compatible with the pre-Q12 behaviour.
    role_models:
        Optional per-role overrides. Each entry is the *full chain*
        for that role: the first item is the primary, the rest are
        fallbacks. When set, the global ``fallback_chain`` is ignored
        for that role.
    max_retries_per_model:
        Forwarded to :class:`FailoverChain`.
    """

    default_model: str
    """Model name in litellm format (``"provider/model"``).

    This is the primary model used when no role-specific override
    exists. **There is no default** — callers must explicitly choose
    a model. This is intentional: model availability and pricing
    change over time, and the right choice depends on the user's
    environment and budget.
    """
    fallback_chain: list[str] = field(default_factory=list)
    role_models: dict[str, list[str]] = field(default_factory=dict)
    max_retries_per_model: int = 2
    timeout: float | None = None

    def chain_for(self, role: str) -> FailoverChain:
        """Return the :class:`FailoverChain` to use for ``role``.

        If the role has explicit ``role_models``, those are used in
        order. Otherwise the chain is ``[default_model, *fallback_chain]``.
        """
        models = list(self.role_models[role]) if role in self.role_models and self.role_models[role] else [self.default_model, *self.fallback_chain]
        return FailoverChain(
            models=models,
            max_retries_per_model=self.max_retries_per_model,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str = "default",
        response_format: dict[str, str] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Run a chat completion through the failover chain for ``role``.

        Returns an :class:`LLMResult` with the response content, parsed
        JSON (when ``response_format`` requests JSON) and diagnostics.
        Raises a typed :class:`knox.llm.errors.LLMError` subclass on
        failure — never a raw provider exception and never silently.
        """
        chain = self.chain_for(role)

        def _call(model: str) -> Any:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if self.timeout is not None:
                kwargs["timeout"] = self.timeout
            return litellm.completion(**kwargs)

        result = chain.run(_call)
        response = result.value
        diagnostics = result.diagnostics

        # Extract content (raise PermanentError on missing/empty content).
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:  # malformed response
            msg = "LLM response missing choices[0].message.content"
            raise PermanentError(
                msg,
                model=diagnostics.model_used,
                attempts=diagnostics.attempts,
                cause=exc,
            ) from exc

        if not content:
            msg = "LLM returned empty content"
            raise PermanentError(
                msg,
                model=diagnostics.model_used,
                attempts=diagnostics.attempts,
            )

        # Parse JSON when requested.
        json_data: dict[str, Any] | None = None
        if response_format is not None and response_format.get("type") == "json_object":
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                msg = f"LLM returned invalid JSON: {exc}"
                raise PermanentError(
                    msg,
                    model=diagnostics.model_used,
                    attempts=diagnostics.attempts,
                    cause=exc,
                ) from exc
            if not isinstance(parsed, dict):
                msg = f"LLM returned JSON of type {type(parsed).__name__}, expected object"
                raise PermanentError(
                    msg,
                    model=diagnostics.model_used,
                    attempts=diagnostics.attempts,
                )
            json_data = parsed

        logger.info(
            "LLM ok role=%s model=%s attempts=%d latency_s=%.2f",
            role,
            diagnostics.model_used,
            diagnostics.attempts,
            diagnostics.total_latency_s,
        )

        return LLMResult(
            content=content,
            json_data=json_data,
            diagnostics=diagnostics,
        )

    def try_complete(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str = "default",
        response_format: dict[str, str] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult | None:
        """Like :meth:`complete` but returns ``None`` instead of raising.

        Use this in pipeline tiers where the existing behaviour was a
        passthrough on failure. The error is logged with full diagnostics
        so failures are still visible.
        """
        try:
            return self.complete(
                messages,
                role=role,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LLMError as err:
            logger.warning(
                "LLM call failed role=%s model=%s attempts=%d: %s",
                role,
                err.model,
                err.attempts,
                err,
            )
            return None
