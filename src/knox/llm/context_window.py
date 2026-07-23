"""Discover a model's context window from Ollama or models.dev.

This module is the canonical way to determine how many tokens a model can
accept and generate. It is used by the agent layer and the doctor command.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

import ollama

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
    """Query Ollama for the model's context window via ``ollama.show()``.

    Priority (most restrictive wins when multiple are set):
    1. ``OLLAMA_NUM_CTX`` environment variable (global user override).
    2. ``parameters.num_ctx`` from ``ollama.show()`` — Modelfile override
       (what Ollama is actually configured to enforce at runtime).
    3. ``modelinfo`` from ``ollama.show()`` — native GGUF metadata.
       Prefers ``original_context_length`` (base context before RoPE scaling)
       over the scaled ``context_length`` when both are present.

    Returns ``None`` if none are available or ``ollama.show()`` fails.
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

    # 2. Modelfile parameters from ollama.show() — what Ollama enforces.
    param_ctx: int | None = None
    # 3. Native GGUF metadata from ollama.show().
    native_ctx: int | None = None
    try:
        resp = ollama.show(model_name)

        # 2a. Parse parameters string for num_ctx (Modelfile override).
        params_str: str = resp.parameters or ""
        for line in params_str.split("\n"):
            stripped = line.strip()
            if stripped.startswith("num_ctx") and stripped[len("num_ctx"):].strip().isdigit():
                param_ctx = int(stripped[len("num_ctx"):].strip())
                break

        # 3a. Native GGUF metadata.
        model_info = resp.modelinfo or {}
        # First pass: look for original_context_length (base context before RoPE)
        for key, val in model_info.items():
            if key.endswith(".original_context_length") and isinstance(val, int) and val > 0:
                native_ctx = val
                break
        # Second pass: fall back to scaled context_length
        if native_ctx is None:
            for key, val in model_info.items():
                if key.endswith(".context_length") and isinstance(val, int) and val > 0:
                    native_ctx = val
                    break
    except Exception:
        logger.debug("ollama.show() failed for %s", model_name, exc_info=True)

    candidates = [c for c in (env_ctx, param_ctx, native_ctx) if c is not None]
    return min(candidates) if candidates else None


def discover_context_window(
    model: str,
    *,
    fallback_max_input: int | None = None,
    fallback_max_output: int | None = None,
    context_window_cap: int | None = None,
) -> ContextWindow:
    """Discover the context window for *model*.

    Strategy (first hit wins; each step fills only the values still missing):
    1. For ``ollama/`` models, query via ``ollama.show()`` — checks
       ``OLLAMA_NUM_CTX`` env var then GGUF ``context_length`` metadata.
    2. Query the ``models.dev`` catalog — covers most cloud providers
       and custom OpenAI-compatible mirrors.
    3. If *fallback_max_input* is provided (from ``[compile.llm]`` config),
       use it as-is with **no clamping** — the value the user configured.
       If neither source resolves AND no fallback is provided, raise
       ``LookupError`` — a missing context window usually means the model
       name is wrong, the provider is unreachable, or config needs fixing.

    Values from steps 1–2 are used **as-is** — never clamped or capped
    (except by *context_window_cap* if set).  The old ``num_ctx_min`` /
    ``num_ctx_max`` clamping (now removed) only existed for the hardcoded
    8192 fallback default, which was silently masking misconfigurations.

    If *context_window_cap* is set (non-``None``), it caps both
    ``max_input`` and ``max_output`` to that value.  Use this when you
    want to restrict a model's native context window.
    """
    max_input: int | None = None
    max_output: int | None = None
    source: ContextSource | None = None

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

    # Apply explicit fallback from config if provided.
    if max_input is None and fallback_max_input is not None:
        max_input = fallback_max_input
        source = ContextSource.FALLBACK
    if max_output is None and fallback_max_output is not None:
        max_output = fallback_max_output
        source = ContextSource.FALLBACK

    # Nothing resolved, no fallback configured — this indicates a real problem.
    if max_input is None or max_output is None or source is None:
        msg = (
            f"Context window not found for {model!r}: "
            "Ollama API and models.dev catalog both returned no results, "
            "and no fallback_max_input was configured. "
            "Check that the model name is correct, the provider is reachable, "
            "and/or set fallback_max_input in [compile.llm] in your config."
        )
        raise LookupError(msg)

    if context_window_cap is not None:
        max_input = min(max_input, context_window_cap)
        max_output = min(max_output, context_window_cap)

    return ContextWindow(max_input=max_input, max_output=max_output, source=source)
