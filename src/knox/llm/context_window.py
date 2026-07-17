"""Discover a model's context window from Ollama or models.dev.

This module is the canonical way to determine how many tokens a model can
accept and generate. It is used by the agent layer and the doctor command.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import StrEnum

from knox.llm.models import resolve_model_limits
from knox.logging import get_logger

logger = get_logger(__name__)


class ContextSource(StrEnum):
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
    """Query Ollama for the model's context window.

    Checks in priority order (most restrictive wins when multiple are set):
    1. ``OLLAMA_NUM_CTX`` environment variable (global user override).
    2. ``parameters.num_ctx`` from ``GET /api/show`` (per-model Modelfile setting).
    3. ``model_info.<arch>.context_length`` from ``GET /api/show`` (native
       GGUF metadata — the model's full context capability).

    Returns ``None`` if none are available or Ollama is unreachable.
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

    # 2 + 3. Per-model from /api/show.
    per_model: int | None = None
    native_ctx: int | None = None
    try:
        url = "http://localhost:11434/api/show"
        payload = json.dumps({"name": model_name}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read())

        # 2. Modelfile parameter override.
        params = data.get("parameters", {})
        # /api/show may return parameters as a Modelfile string (not a dict)
        # — handle both forms gracefully.
        num_ctx_str = params.get("num_ctx") if isinstance(params, dict) else None
        if num_ctx_str is not None:
            per_model = int(num_ctx_str)

        # 3. Native context from GGUF metadata — any key ending in
        #    ``.context_length`` carries the model's full capability
        #    (e.g. ``qwen35.context_length``, ``llama.context_length``).
        model_info = data.get("model_info", {})
        if isinstance(model_info, dict):
            for key, val in model_info.items():
                if key.endswith(".context_length") and isinstance(val, int) and val > 0:
                    native_ctx = val
                    break
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError):
        # Ollama unreachable or returned malformed data — optional probe, degrade to None.
        logger.debug("Ollama /api/show query failed for %s", model_name, exc_info=True)

    # Most restrictive of the explicit limits wins; native capability is
    # the baseline when no explicit limit is set.
    candidates = [c for c in (env_ctx, per_model, native_ctx) if c is not None]
    return min(candidates) if candidates else None


def discover_context_window(  # noqa: PLR0912
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
    1. For Ollama models, query ``GET /api/show`` for the runtime
       ``num_ctx`` parameter (also checks ``OLLAMA_NUM_CTX`` env var).
    2. Query the `models.dev <https://models.dev>`_ catalog — a
       community DB that covers most providers and custom mirrors.
    3. Fall back to *fallback_max_input* / *fallback_max_output*,
       clamped to ``[num_ctx_min, num_ctx_max]``, and emit a warning
       when no source could resolve the model.

    Discovered values (steps 1–2) are used **as-is** — they are never
    clamped or capped.  The ``num_ctx_min`` / ``num_ctx_max`` rails apply
    only to the **fallback** path, never to discovered values.

    If *context_window_cap* is set (non-``None``), it overrides discovered
    values by capping ``max_input`` and ``max_output`` to the cap.  This is
    a user-configured limit, not a hardcoded assumption.
    """
    max_input: int | None = None
    max_output: int | None = None
    source = ContextSource.FALLBACK

    # Validate model name before attempting discovery.
    # Common mistake: setting KNOX_LOCAL_LLM to "ollama run <model>" (a CLI
    # command) instead of "ollama/<model>" (``provider/model`` format).
    if " " in model:
        msg = (
            f"Invalid model name {model!r}: contains spaces. "
            "Did you include a CLI command (e.g. 'ollama run ...') in the model string?\n"
            "Use 'provider/<model-name>' format (e.g. 'ollama/qwen3')"
        )
        raise ValueError(msg)

    if model.startswith("ollama/"):
        ollama_ctx = _query_ollama_num_ctx(model.removeprefix("ollama/"))
        if ollama_ctx is not None:
            max_input = ollama_ctx
            max_output = ollama_ctx
            source = ContextSource.OLLAMA

    if max_input is None or max_output is None:
        try:
            md_input, md_output = resolve_model_limits(model)
            if max_input is None:
                max_input = md_input
            if max_output is None:
                max_output = md_output
            source = ContextSource.MODELS_DEV
        except (KeyError, LookupError):
            pass

    if max_input is None:
        logger.warning(
            "%s — context window not found in Ollama or models.dev; "
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
