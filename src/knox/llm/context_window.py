"""Discover a model's context window from litellm, Ollama, or models.dev.

This module is the canonical way to determine how many tokens a model can
accept and generate. It is used by the LLMClient, the agent layer, and the
doctor command.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import StrEnum

import litellm

from knox.llm.models_dev import resolve_model_limits
from knox.logging import get_logger

logger = get_logger(__name__)


class ContextSource(StrEnum):
    LITELLM = "litellm"
    OLLAMA = "ollama"
    MODELS_DEV = "models_dev"
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


def discover_context_window(
    model: str,
    *,
    fallback_max_input: int = 8192,
    fallback_max_output: int = 4096,
    num_ctx_min: int = 2048,
    num_ctx_max: int = 131072,
    context_window_cap: int | None = None,
) -> ContextWindow:
    """Discover the context window for *model*.

    Strategy (first hit wins; each step fills only the values still missing):
    1. Try ``litellm.get_model_info()`` for static max_input/max_output.
    2. For Ollama models, query ``GET /api/show`` for the runtime
       ``num_ctx`` parameter (also checks ``OLLAMA_NUM_CTX`` env var).
    3. Query the `models.dev <https://models.dev>`_ catalog — a
       community DB that covers custom-provider models (e.g. Ollama
       Cloud) and others absent from litellm's static tables.
    4. Fall back to *fallback_max_input* / *fallback_max_output*,
       clamped to ``[num_ctx_min, num_ctx_max]``, and emit a warning
       when no source could resolve the model.

    Discovered values (steps 1–3) are used **as-is** — they are never
    clamped or capped.  The ``num_ctx_min`` / ``num_ctx_max`` rails apply
    only to the **fallback** path, never to discovered values.

    If *context_window_cap* is set (non-``None``), it overrides discovered
    values by capping ``max_input`` and ``max_output`` to the cap.  This is
    a user-configured limit, not a hardcoded assumption.
    """
    max_input: int | None = None
    max_output: int | None = None
    source = ContextSource.FALLBACK

    custom_prefixes = {f"{e['provider']}/" for e in (litellm.custom_provider_map or []) if e.get("provider")}
    if not any(model.startswith(p) for p in custom_prefixes):
        try:
            info = litellm.get_model_info(model)
            if info:
                max_input = info.get("max_input_tokens")
                max_output = info.get("max_output_tokens")
                if max_input is not None:
                    source = ContextSource.LITELLM
        except Exception:
            logger.debug("litellm has no model info for %s", model, exc_info=True)

    if model.startswith("ollama/"):
        ollama_ctx = _query_ollama_num_ctx(model.removeprefix("ollama/"))
        if ollama_ctx is not None:
            if max_input is None or ollama_ctx < max_input:
                max_input = ollama_ctx
                source = ContextSource.OLLAMA
            if max_output is None or ollama_ctx < max_output:
                max_output = ollama_ctx
                source = ContextSource.OLLAMA

    if max_input is None or max_output is None:
        limits = resolve_model_limits(model)
        if limits is not None:
            md_input, md_output = limits
            if max_input is None:
                max_input = md_input
            if max_output is None:
                max_output = md_output
            source = ContextSource.MODELS_DEV

    if max_input is None:
        logger.warning(
            "%s — context window not found in litellm, Ollama, or models.dev; "
            "using fallback_max_input=%d. Set fallback_max_input or "
            "context_window_cap in config to match your model's actual context window",
            model,
            fallback_max_input,
        )
        max_input = max(num_ctx_min, min(fallback_max_input, num_ctx_max))
        source = ContextSource.FALLBACK
    if max_output is None:
        max_output = max(num_ctx_min, min(fallback_max_output, num_ctx_max))
        source = ContextSource.FALLBACK

    if context_window_cap is not None:
        if source == ContextSource.FALLBACK:
            max_input = context_window_cap
            max_output = context_window_cap
        else:
            max_input = min(max_input, context_window_cap)
            max_output = min(max_output, context_window_cap)

    return ContextWindow(max_input=max_input, max_output=max_output, source=source)
