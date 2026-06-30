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
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import litellm

import knox.llm.providers  # noqa: F401 — registers custom providers on import
from knox.llm.errors import ConfigurationError, LLMError, PermanentError
from knox.llm.failover import CallDiagnostics, FailoverChain
from knox.llm.format import (
    salvage_json,  # noqa: F401 — re-exported; callers import from here
)
from knox.logging import get_logger

litellm.suppress_debug_info = True  # suppress stdout noise on exceptions

logger = get_logger(__name__)

JSON_RESPONSE_FORMAT: Final[dict[str, str]] = {"type": "json_object"}

# Max times to bump max_tokens when a reasoning model exhausts the budget on
# reasoning_content (finish_reason="length", empty visible content). Each bump
# multiplies the budget; a genuinely empty response still raises PermanentError.
_REASONING_TRUNCATION_BUMPS: Final[int] = 3


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


@dataclass(frozen=True)
class _StreamResult:
    """Internal: assembled content + timing from a streamed response."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float  # time from call start to first token


class ContextSource(StrEnum):
    LITELLM = "litellm"
    OLLAMA = "ollama"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class ContextWindow:
    """Discovered context window limits for a model.

    ``max_input`` and ``max_output`` are in **tokens**.
    ``source`` records where the values came from for diagnostics.
    """

    max_input: int
    max_output: int
    source: ContextSource


@dataclass
class UsageTracker:
    """Accumulates LLM usage statistics across complete() calls.

    Each :class:`LLMClient` holds one instance. After pipeline runs,
    the caller reads ``client.usage`` to populate the wiki log.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0
    total_ttft_s: float = 0.0
    per_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, *, prompt_tokens: int, completion_tokens: int, model: str, ttft_s: float = 0.0) -> None:
        """Record one successful completion."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.call_count += 1
        self.total_ttft_s += ttft_s
        entry = self.per_model.setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
        entry["prompt"] += prompt_tokens
        entry["completion"] += completion_tokens
        entry["calls"] += 1

    @property
    def avg_ttft(self) -> float:
        """Average time-to-first-token across all recorded calls."""
        return self.total_ttft_s / self.call_count if self.call_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for wiki.metadata storage."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "call_count": self.call_count,
            "avg_ttft": round(self.avg_ttft, 2),
            "per_model": dict(self.per_model),
        }


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


def _discover_context_window(
    model: str,
    *,
    fallback_max_input: int = 8192,
    fallback_max_output: int = 4096,
    num_ctx_min: int = 2048,
    num_ctx_max: int = 131072,
    context_window_cap: int | None = None,
) -> ContextWindow:
    """Discover the context window for *model*.

    Strategy:
    1. Try ``litellm.get_model_info()`` for static max_input/max_output.
    2. For Ollama models, query ``GET /api/show`` for the runtime
       ``num_ctx`` parameter (also checks ``OLLAMA_NUM_CTX`` env var).
    3. Fall back to *fallback_max_input* / *fallback_max_output*,
       clamped to ``[num_ctx_min, num_ctx_max]``, and emit a warning
       when the model couldn't be looked up (custom provider models
       always fall through here — their context windows aren't in
       litellm's DB nor discoverable via API).

    Discovered values from litellm or Ollama are used **as-is** — they
    are never clamped or capped.  The ``num_ctx_min`` / ``num_ctx_max``
    rails apply only to the **fallback** path, never to discovered values.

    If *context_window_cap* is set (non-``None``), it overrides discovered
    values by capping ``max_input`` and ``max_output`` to the cap.  This is
    a user-configured limit, not a hardcoded assumption.
    """
    max_input: int | None = None
    max_output: int | None = None
    source = "fallback"

    # 1. litellm model info — skip custom providers (never in litellm's DB,
    # so get_model_info always raises; calling it only produces a misleading
    # traceback-style warning). Build the registered custom prefixes once.
    custom_prefixes = {f"{e['provider']}/" for e in (litellm.custom_provider_map or []) if e.get("provider")}
    if not any(model.startswith(p) for p in custom_prefixes):
        try:
            info = litellm.get_model_info(model)
            if info:
                max_input = info.get("max_input_tokens")
                max_output = info.get("max_output_tokens")
                if max_input is not None:
                    source = ContextSource.LITELLM
        except Exception:  # litellm raises bare Exception for unmapped models (best-effort probe)
            # Not fatal: discovery falls through to the Ollama/fallback paths.
            # No exc_info — litellm's message is self-explanatory and a
            # traceback box looks like a crash.
            logger.warning(
                "litellm has no model info for %s — using fallback context limits",
                model,
            )
    else:
        # Custom providers are never in litellm's model DB, and the cloud
        # APIs they delegate to (e.g. Ollama Cloud) typically don't expose
        # context window metadata either.  The model's actual context is
        # lost — the user must configure it via config key.
        logger.warning(
            "%s is a custom provider model — context window unknown, "
            "using fallback_max_input=%d.  "
            "Set fallback_max_input or context_window_cap in config to match "
            "your model's actual context window",
            model,
            fallback_max_input,
        )

    # 2. Ollama runtime num_ctx (overrides litellm if available).
    if model.startswith("ollama/"):
        ollama_ctx = _query_ollama_num_ctx(model.removeprefix("ollama/"))
        if ollama_ctx is not None:
            # Ollama's num_ctx is the total context (input + output).
            if max_input is None or ollama_ctx < max_input:
                max_input = ollama_ctx
                source = ContextSource.OLLAMA
            if max_output is None or ollama_ctx < max_output:
                max_output = ollama_ctx
                source = ContextSource.OLLAMA

    # 3. Fallback only — clamp the fallback, never the discovered value.
    if max_input is None:
        max_input = max(num_ctx_min, min(fallback_max_input, num_ctx_max))
        source = ContextSource.FALLBACK
    if max_output is None:
        max_output = max(num_ctx_min, min(fallback_max_output, num_ctx_max))
        if source == ContextSource.FALLBACK:
            pass  # already fallback
        elif source != ContextSource.FALLBACK:
            source = ContextSource.FALLBACK

    # 4. User-configured cap overrides discovered values.
    # When the fallback was used and a cap is set, the cap replaces
    # the fallback entirely (it can raise as well as lower).
    if context_window_cap is not None:
        if source == ContextSource.FALLBACK:
            max_input = context_window_cap
            max_output = context_window_cap
        else:
            max_input = min(max_input, context_window_cap)
            max_output = min(max_output, context_window_cap)

    return ContextWindow(max_input=max_input, max_output=max_output, source=source)


def _query_ollama_num_ctx(model_name: str) -> int | None:
    """Query Ollama for the model's runtime ``num_ctx``.

    Checks in priority order:
    1. ``OLLAMA_NUM_CTX`` environment variable (global Ollama limit).
    2. ``parameters.num_ctx`` from ``GET /api/show`` (per-model Modelfile setting).

    When both are set, the **more restrictive** (smaller) value wins.
    Returns ``None`` if neither is set or Ollama is unreachable.
    """
    # 1. Global env var.
    env_ctx: int | None = None
    env_val = os.environ.get("OLLAMA_NUM_CTX")
    if env_val:
        try:
            env_ctx = int(env_val)
            if env_ctx <= 0:
                env_ctx = None
        except ValueError:
            logger.warning("OLLAMA_NUM_CTX=%r is not a valid integer", env_val, exc_info=True)

    # 2. Per-model from /api/show.
    per_model: int | None = None
    try:
        url = "http://localhost:11434/api/show"
        payload = json.dumps({"name": model_name}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read())
        params = data.get("parameters", {})
        # /api/show may return parameters as a Modelfile string (not a dict)
        # — handle both forms gracefully.
        num_ctx_str = params.get("num_ctx") if isinstance(params, dict) else None
        if num_ctx_str is not None:
            per_model = int(num_ctx_str)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError):
        # Ollama unreachable or returned malformed data — optional probe, degrade to None.
        logger.debug("Ollama /api/show query failed for %s", model_name, exc_info=True)

    # Most restrictive wins when both are set.
    candidates = [c for c in (env_ctx, per_model) if c is not None]
    return min(candidates) if candidates else None


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
            self.context_window = _discover_context_window(
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
        content = sr.content

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
        if response_format is not None and response_format.get("type") == "json_object":
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
        except ConfigurationError:
            raise  # fatal — auth / bad config, let it burn
        except LLMError as err:
            self.failure_count += 1
            self.degraded_calls.append(
                {
                    "role": role,
                    "model": err.model,
                    "error": type(err).__name__,
                    "message": str(err)[:200],
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
