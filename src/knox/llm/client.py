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
import os
import time
import urllib.request
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


@dataclass(frozen=True)
class _StreamResult:
    """Internal: assembled content + timing from a streamed response."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float  # time from call start to first token


@dataclass(frozen=True)
class ContextWindow:
    """Discovered context window limits for a model.

    ``max_input`` and ``max_output`` are in **tokens**.
    ``source`` records where the values came from for diagnostics.
    """

    max_input: int
    max_output: int
    source: str  # "litellm" | "ollama" | "fallback"


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
       clamped to ``[num_ctx_min, num_ctx_max]``.

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

    # 1. litellm model info
    try:
        info = litellm.get_model_info(model)
        if info:
            max_input = info.get("max_input_tokens")
            max_output = info.get("max_output_tokens")
            if max_input is not None:
                source = "litellm"
    except Exception:  # noqa: BLE001
        logger.debug("litellm.get_model_info failed for %s", model, exc_info=True)

    # 2. Ollama runtime num_ctx (overrides litellm if available).
    if model.startswith("ollama/"):
        ollama_ctx = _query_ollama_num_ctx(model.removeprefix("ollama/"))
        if ollama_ctx is not None:
            # Ollama's num_ctx is the total context (input + output).
            if max_input is None or ollama_ctx < max_input:
                max_input = ollama_ctx
                source = "ollama"
            if max_output is None or ollama_ctx < max_output:
                max_output = ollama_ctx
                source = "ollama"

    # 3. Fallback only — clamp the fallback, never the discovered value.
    if max_input is None:
        max_input = max(num_ctx_min, min(fallback_max_input, num_ctx_max))
        source = "fallback"
    if max_output is None:
        max_output = max(num_ctx_min, min(fallback_max_output, num_ctx_max))
        if source == "fallback":
            pass  # already fallback
        elif source != "fallback":
            source = "fallback"

    # 4. User-configured cap overrides discovered values.
    if context_window_cap is not None:
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
            logger.warning("OLLAMA_NUM_CTX=%r is not a valid integer", env_val)

    # 2. Per-model from /api/show.
    per_model: int | None = None
    try:
        url = "http://localhost:11434/api/show"
        payload = json.dumps({"name": model_name}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read())
        params = data.get("parameters", {})
        num_ctx_str = params.get("num_ctx")
        if num_ctx_str is not None:
            per_model = int(num_ctx_str)
    except Exception:  # noqa: BLE001
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
    fallback_chain: list[str] = field(default_factory=list)
    role_models: dict[str, list[str]] = field(default_factory=dict)
    max_retries_per_model: int = 2
    timeout: float | None = None
    context_window: ContextWindow | None = None
    usage: UsageTracker = field(default_factory=UsageTracker, repr=False)
    _fallback_max_input: int = field(default=8192, repr=False)
    _fallback_max_output: int = field(default=4096, repr=False)
    _num_ctx_min: int = field(default=2048, repr=False)
    _num_ctx_max: int = field(default=131072, repr=False)
    _context_window_cap: int | None = field(default=None, repr=False)

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
        call_start = time.monotonic()

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
            # Streaming enables TTFT measurement for split TPS metrics.
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

            ttft: float | None = None
            chunks: list[str] = []
            pt = 0
            ct = 0
            for event in stream:
                if ttft is None:
                    ttft = time.monotonic() - call_start
                if event.choices:
                    delta = event.choices[0].delta
                    if delta and delta.content:
                        chunks.append(delta.content)
                u = getattr(event, "usage", None)
                if u is not None:
                    pt = int(getattr(u, "prompt_tokens", 0) or 0)
                    ct = int(getattr(u, "completion_tokens", 0) or 0)
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

        # Log split TPS metrics.
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
        except LLMError as err:
            logger.warning(
                "LLM call failed role=%s model=%s attempts=%d: %s",
                role,
                err.model,
                err.attempts,
                err,
            )
            return None
