"""Shared LiteLLM-backed client used by every business module.

:class:`LLMClient` is the single call site for LLM completions in
KNOX. Modules pass a ``role`` (e.g. ``"resolve"``, ``"wicer.judge"``)
and the client picks the right model + fallback chain for that role.

This module wraps litellm — business modules must not import litellm
directly. Doing so reintroduces the silent fallback antipattern this
client exists to eliminate.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

import litellm
from pydantic import BaseModel

import knox.llm.providers  # noqa: F401 — registers custom providers on import
from knox.core.config.keys import (
    COMPILE_LLM_CONTEXT_WINDOW_CAP,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
    COMPILE_LLM_ROLE_MODELS,
)
from knox.core.utils.exceptions import shorten_exception
from knox.llm.context_window import ContextWindow, discover_context_window
from knox.llm.errors import ConfigurationError, LLMError, PermanentError
from knox.llm.failover import CallDiagnostics, FailoverChain
from knox.llm.format import salvage_json, strip_reasoning  # noqa: F401 — re-exported; callers import from here
from knox.llm.roles import LLMRole

# Re-export with old private name for backward compat
_strip_reasoning = strip_reasoning
from knox.logging import get_logger

litellm.suppress_debug_info = True  # suppress stdout noise on exceptions

logger = get_logger(__name__)

JSON_RESPONSE_FORMAT: Final[dict[str, str]] = {"type": "json_object"}

# Max times to bump max_tokens when a reasoning model exhausts the budget on
# reasoning_content (finish_reason="length", empty visible content). Each bump
# multiplies the budget; a genuinely empty response still raises PermanentError.
_REASONING_TRUNCATION_BUMPS: Final[int] = 3

# Reasoning models that do not use a separate ``reasoning_content`` field
# sometimes emit their chain-of-thought inline in the *content* field,
# wrapped in an XML-like tag. Tag names vary by model — this list covers the
# common ones. To support a new model's tag, add its name here (lowercase,
# without angle brackets). Stripping happens centrally in :meth:`LLMClient.complete`
# so every consumer (compile, query, organize, lint) gets clean content.

@dataclass(frozen=True)
class ToolCall:
    """A single tool/function call returned by the LLM.

    Used by :meth:`LLMClient.complete_with_tools` for the conversational
    agent loop. ``arguments`` is the parsed JSON dict.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResult:
    """Successful completion returned by :meth:`LLMClient.complete`.

    ``content`` is the raw text content of the assistant message.
    ``json_data`` is the parsed JSON object when ``response_format``
    requested one (``None`` otherwise). ``diagnostics`` carries the
    failover-chain observability data. ``tool_calls`` is populated by
    :meth:`~LLMClient.complete_with_tools` (``None`` otherwise).
    """

    content: str
    json_data: dict[str, Any] | None
    diagnostics: CallDiagnostics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: list[ToolCall] | None = None


@dataclass(frozen=True)
class _StreamResult:
    """Internal: assembled content + timing from a streamed response."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float  # time from call start to first token


from knox.llm.usage import UsageTracker


def _resolve_env_value(value: str) -> str | None:
    """Resolve a config value that may be an env-var reference.

    If *value* starts with ``$``, the remainder is treated as an
    environment variable name. Returns the env-var value, or ``None``
    if the variable is not set — preventing a literal ``$XXX`` string
    from leaking into downstream code as a real value.
    """
    if value.startswith("$"):
        return os.environ.get(value[1:])
    return value


class _ConfigLike(Protocol):
    """Minimal config interface (ResolvedConfig or plain dict)."""

    def get(self, key: str, default: Any = None) -> Any: ...


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
    context_window:
        Discovered context window limits. If ``None``, discovered
        automatically from litellm / Ollama at init time.
    """

    default_model: str
    """Model name in litellm format (``"provider/model"``).

    This is the primary model used when no role-specific override
    exists. **There is no default** — callers must explicitly choose
    a model. This is intentional: model availability and pricing
    change over time, and the right choice depends on the user's
    environment and budget.
    """
    data_format: str = "json"
    """LLM communication format: ``"json"`` (default), ``"toon"``, or ``"ison"``.

    Controls both how structured context data is serialized for prompt
    input (via :func:`encode_for_llm`) and how the LLM response is
    parsed. For non-JSON formats, the API-level ``response_format:
    json_object`` constraint is replaced with a system-message hint.
    Requires optional packages: ``toon_format`` (TOON) or ``ison-py`` (ISON).
    """
    fallback_chain: list[str] = field(default_factory=list)
    role_models: dict[str, list[str]] = field(default_factory=dict)
    max_retries_per_model: int = 2
    timeout: float | None = None
    context_window: ContextWindow | None = None
    usage: UsageTracker = field(default_factory=UsageTracker, repr=False)
    failure_count: int = field(default=0, repr=False)
    degraded_calls: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _fallback_max_input: int = field(default=8192, repr=False)
    _fallback_max_output: int = field(default=4096, repr=False)
    _num_ctx_min: int = field(default=2048, repr=False)
    _num_ctx_max: int = field(default=131072, repr=False)
    _context_window_cap: int | None = field(default=None, repr=False)
    _api_base: str | None = field(default=None, repr=False)
    _api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.default_model:
            raise ValueError("An explicit 'default_model' is required but was not provided.")
        if self.context_window is None:
            self.context_window = discover_context_window(
                self.default_model,
                fallback_max_input=self._fallback_max_input,
                fallback_max_output=self._fallback_max_output,
                num_ctx_min=self._num_ctx_min,
                num_ctx_max=self._num_ctx_max,
                context_window_cap=self._context_window_cap,
            )
            logger.info(
                "Context window: model=%s max_input=%d max_output=%d source=%s",
                self.default_model,
                self.context_window.max_input,
                self.context_window.max_output,
                self.context_window.source,
            )

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

    def complete(  # noqa: PLR0915
        self,
        messages: list[dict[str, Any]],
        *,
        role: str = LLMRole.DEFAULT,
        response_format: dict[str, str] | type[BaseModel] | None = None,
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
        cw = self.context_window
        if cw is None:
            raise ValueError("context_window must be set before calling complete()")
        call_start = time.monotonic()

        def _call(model: str) -> Any:  # noqa: PLR0915
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
            if self._api_base is not None:
                kwargs["api_base"] = self._api_base
            if self._api_key is not None:
                kwargs["api_key"] = _resolve_env_value(self._api_key)
            # Streaming enables TTFT measurement for split TPS metrics.
            # Reasoning models (e.g. deepseek-v4-flash) can spend a tight
            # max_tokens budget entirely on ``reasoning_content`` and return
            # finish_reason="length" with empty *visible* content. When that
            # happens, bump the budget and retry the same call so the model can
            # finish reasoning and emit its answer. Bounded attempts avoid
            # runaway cost; a genuinely empty response still raises below.
            budget = max_tokens
            chunks: list[str] = []
            pt = 0
            ct = 0
            ttft: float | None = None
            for _ in range(_REASONING_TRUNCATION_BUMPS + 1):
                if budget is not None:
                    kwargs["max_tokens"] = budget
                kwargs["stream"] = True
                kwargs["stream_options"] = {"include_usage": True}
                try:
                    stream = litellm.completion(**kwargs)
                except (TypeError, ValueError):
                    # Some providers don't support streaming params.
                    kwargs.pop("stream", None)
                    kwargs.pop("stream_options", None)
                    resp = litellm.completion(**kwargs)
                    content = resp.choices[0].message.content or ""
                    usage = getattr(resp, "usage", None)
                    pt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
                    ct = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
                    elapsed = time.monotonic() - call_start
                    return _StreamResult(content=content, prompt_tokens=pt, completion_tokens=ct, ttft_s=elapsed)

                chunks = []
                pt = 0
                ct = 0
                ttft = None
                saw_reasoning = False
                for event in stream:
                    if ttft is None:
                        ttft = time.monotonic() - call_start
                    if event.choices:
                        choice = event.choices[0]
                        delta = choice.delta
                        if delta:
                            if delta.content:
                                chunks.append(delta.content)
                            if getattr(delta, "reasoning_content", None):
                                saw_reasoning = True
                    u = getattr(event, "usage", None)
                    if u is not None:
                        pt = int(getattr(u, "prompt_tokens", 0) or 0)
                        ct = int(getattr(u, "completion_tokens", 0) or 0)
                content = "".join(chunks)
                # Reasoning model spent the budget on thinking but emitted no
                # visible answer -> retry with more room. (finish_reason is
                # often "stop" here, not "length", so key on empty content +
                # reasoning seen rather than the finish reason.)
                if not content and saw_reasoning and budget is not None:
                    budget = max(budget * 4, 4096)
                    continue
                return _StreamResult(
                    content=content,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    ttft_s=ttft if ttft is not None else time.monotonic() - call_start,
                )
            return _StreamResult(
                content="".join(chunks),
                prompt_tokens=pt,
                completion_tokens=ct,
                ttft_s=ttft if ttft is not None else time.monotonic() - call_start,
            )

        result = chain.run(_call)
        sr = result.value
        diagnostics = result.diagnostics

        total_latency = diagnostics.total_latency_s
        content = _strip_reasoning(sr.content)

        # Extract content (raise PermanentError on missing/empty content).
        if not content:
            msg = (
                f"LLM returned empty content "
                f"(model={diagnostics.model_used!r}, role={role!r}). "
                f"This often indicates a wrong model name, missing API key, "
                f"or provider returning empty — check your --llm-default-model "
                f"and config."
            )
            raise PermanentError(
                msg,
                model=diagnostics.model_used,
                attempts=diagnostics.attempts,
            )

        json_data: dict[str, Any] | None = None
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            parsed = salvage_json(content)
            if parsed is None:
                msg = f"LLM returned invalid JSON (model={diagnostics.model_used!r}, role={role!r})"
                raise PermanentError(
                    msg,
                    model=diagnostics.model_used,
                    attempts=diagnostics.attempts,
                )
            if not isinstance(parsed, dict):
                msg = f"LLM returned {type(parsed).__name__}, expected object (model={diagnostics.model_used!r}, role={role!r})"
                raise PermanentError(
                    msg,
                    model=diagnostics.model_used,
                    attempts=diagnostics.attempts,
                )
            json_data = parsed

        pt = sr.prompt_tokens
        ct = sr.completion_tokens
        ttft_s = sr.ttft_s
        if pt or ct:
            gen_time = max(total_latency - ttft_s, 0.001)
            p_tps = f"{pt / ttft_s:.1f}" if ttft_s > 0 else "?"
            c_tps = f"{ct / gen_time:.1f}"
            t_tps = f"{(pt + ct) / total_latency:.1f}" if total_latency > 0 else "?"
            logger.info(
                "LLM ok role=%s model=%s attempts=%d ttft=%.1fs in=%d out=%d max_in=%d p_tps=%s c_tps=%s t_tps=%s",
                role,
                diagnostics.model_used,
                diagnostics.attempts,
                ttft_s,
                pt,
                ct,
                cw.max_input,
                p_tps,
                c_tps,
                t_tps,
            )
            self.usage.record(
                prompt_tokens=pt,
                completion_tokens=ct,
                model=diagnostics.model_used,
                ttft_s=ttft_s,
            )
        else:
            logger.info(
                "LLM ok role=%s model=%s attempts=%d in=%.1fs (no token info)",
                role,
                diagnostics.model_used,
                diagnostics.attempts,
                total_latency,
            )

        return LLMResult(
            content=content,
            json_data=json_data,
            diagnostics=diagnostics,
            prompt_tokens=sr.prompt_tokens,
            completion_tokens=sr.completion_tokens,
        )

    def try_complete(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str = LLMRole.DEFAULT,
        response_format: dict[str, str] | type[BaseModel] | None = None,
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
        except ConfigurationError:
            raise  # fatal — auth / bad config, let it burn
        except LLMError as err:
            self.failure_count += 1
            self.degraded_calls.append(
                {
                    "role": role,
                    "model": err.model,
                    "error": type(err).__name__,
                    "message": shorten_exception(err),
                }
            )
            logger.warning(
                "LLM call failed role=%s model=%s attempts=%d: %s",
                role,
                err.model,
                err.attempts,
                err,
            )
            return None

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        role: str = LLMRole.AGENT,
        tool_choice: str = "auto",
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Run a chat completion with tool-calling enabled.

        Unlike :meth:`complete`, this uses non-streaming mode and passes
        ``tools`` + ``tool_choice`` to the LLM. The response may contain
        ``tool_calls`` (the LLM decided to call a tool) or plain ``content``
        (the LLM produced a final answer).

        Used by the conversational agent loop. Does NOT support
        ``response_format`` (tools and JSON schema are mutually exclusive
        in most providers).

        Args:
            messages: Conversation messages including system prompt.
            tools: Tool schemas in litellm/OpenAI format.
            role: LLM role for failover chain resolution.
            tool_choice: "auto", "required", "none", or specific tool name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            :class:`LLMResult` with ``tool_calls`` populated when the LLM
            requested tool execution, or ``content`` when it produced a
            final answer.
        """
        chain = self.chain_for(role)
        call_start = time.monotonic()
        tool_calls_holder: list[list[ToolCall]] = [[]]

        def _call(model: str) -> _StreamResult:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "tools": tools,
                "tool_choice": tool_choice,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if self.timeout is not None:
                kwargs["timeout"] = self.timeout
            if self._api_base is not None:
                kwargs["api_base"] = self._api_base
            if self._api_key is not None:
                kwargs["api_key"] = _resolve_env_value(self._api_key)

            resp = litellm.completion(**kwargs)
            message = resp.choices[0].message
            content = message.content or ""

            # Stash tool calls for the outer scope.
            raw_tool_calls = getattr(message, "tool_calls", None)
            parsed_tool_calls: list[ToolCall] = []
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    fn = getattr(tc, "function", None)
                    if fn is None:
                        continue
                    try:
                        args = json.loads(fn.arguments) if fn.arguments else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    parsed_tool_calls.append(ToolCall(id=tc.id, name=fn.name, arguments=args))
            tool_calls_holder[0] = parsed_tool_calls

            usage = getattr(resp, "usage", None)
            pt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            ct = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            elapsed = time.monotonic() - call_start
            return _StreamResult(content=content, prompt_tokens=pt, completion_tokens=ct, ttft_s=elapsed)

        result = chain.run(_call)
        sr = result.value
        tool_calls = tool_calls_holder[0]
        content = _strip_reasoning(sr.content)

        return LLMResult(
            content=content,
            json_data=None,
            diagnostics=result.diagnostics,
            prompt_tokens=sr.prompt_tokens,
            completion_tokens=sr.completion_tokens,
            tool_calls=tool_calls if tool_calls else None,
        )

    def degradation_report(self) -> dict[str, Any]:
        """Return a summary of LLM call successes vs failures."""
        total = self.usage.call_count + self.failure_count
        return {
            "successful_calls": self.usage.call_count,
            "failed_calls": self.failure_count,
            "total_calls": total,
            "degradation_rate": round(self.failure_count / total, 3) if total > 0 else 0.0,
            "recent_failures": self.degraded_calls[-20:],
        }

    @classmethod
    def from_config(
        cls,
        cfg: _ConfigLike,
        *,
        default_model: str | None = None,
        timeout: float | None = None,
    ) -> LLMClient | None:
        """Build from resolved config, or ``None`` when no model is set.

        ``default_model`` (e.g. a ``--llm-default-model`` CLI flag) overrides
        ``compile.llm.default_model`` from config.  Fallback chain and
        per-role models are read from config when present.
        """
        model = default_model or cfg.get(COMPILE_LLM_DEFAULT_MODEL)
        if not model:
            return None
        # Coerce context_window_cap to int — env var expansion produces strings.
        raw_cap = cfg.get(COMPILE_LLM_CONTEXT_WINDOW_CAP)
        cap: int | None = None
        if raw_cap is not None:
            with contextlib.suppress(ValueError, TypeError):
                cap = int(raw_cap)
        return cls(
            default_model=model,
            fallback_chain=list(cfg.get(COMPILE_LLM_FALLBACK_CHAIN, [])),
            role_models=dict(cfg.get(COMPILE_LLM_ROLE_MODELS, {})),
            timeout=timeout,
            _context_window_cap=cap,
        )
