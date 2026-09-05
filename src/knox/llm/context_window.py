"""Discover a model's context window from Ollama or models.dev.

This module is the canonical way to determine how many tokens a model can
accept and generate. It is used by the agent layer and the doctor command.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from knox.core.config import LLMConfig
from knox.llm.models import _query_ollama_num_ctx, _user_providers, resolve_model_limits
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


def discover_context_window(  # noqa: PLR0915
    model: str,
    *,
    fallback_max_input: int | None = None,
    fallback_max_output: int | None = None,
    context_window_cap: int | str | None = None,
    llm_config: LLMConfig | None = None,
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

    The typed *llm_config* can be passed instead of individual
    *fallback_max_input* / *fallback_max_output* / *context_window_cap*
    or to supply defaults that explicit parameters override.

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

    # 1b. User-defined provider (from [providers] in knox.toml)?
    # Resolution: per-model context_window → provider-level context_window
    # (old key context_window_cap still honored) — finest granularity wins.
    if max_input is None:
        provider_id, model_name = _split_model(model)
        if provider_id in _user_providers:
            cfg = _user_providers[provider_id]
            model_cfg = cfg.get("models", {}).get(model_name, {})
            per_model = model_cfg.get("context_window") if isinstance(model_cfg, dict) else None
            provider_level = cfg.get("context_window", cfg.get("context_window_cap"))
            override = per_model if per_model is not None else provider_level
            if override is not None:
                max_input = int(override)
                max_output = int(override)
                source = ContextSource.FALLBACK

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

    # Extract values from typed LLMConfig if provided.
    if llm_config is not None:
        if fallback_max_input is None:
            fallback_max_input = llm_config.fallback_max_input
        if fallback_max_output is None:
            fallback_max_output = llm_config.fallback_max_output
        if context_window_cap is None:
            # New canonical key; fall back to the deprecated context_window_cap.
            context_window_cap = llm_config.context_window if llm_config.context_window is not None else llm_config.context_window_cap

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

    cap = _coerce_context_window_cap(context_window_cap)
    if cap is not None:
        max_input = min(max_input, cap)
        max_output = min(max_output, cap)

    return ContextWindow(max_input=max_input, max_output=max_output, source=source)


def _coerce_context_window_cap(value: int | str | None) -> int | None:
    """Coerce a raw config ``context_window_cap`` value to ``int | None``.

    Config values reach this point as strings when loaded from TOML with
    env-var interpolation (e.g. ``context_window_cap = "${CTX_CAP}"``),
    since the surrounding quotes force a string type. Unset env vars
    leave a literal ``${...}`` placeholder, which is treated as "unset"
    (returns ``None``) rather than crashing.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text or "$" in text:
        # Empty string or an unexpanded ``${VAR}`` placeholder — treat as unset.
        logger.warning("context_window_cap %r is not a valid integer; ignoring", value)
        return None
    try:
        return int(text)
    except ValueError:
        logger.warning("context_window_cap %r is not a valid integer; ignoring", value)
        return None


def _split_model(model_string: str) -> tuple[str, str]:
    """Split ``provider/model`` into *(provider, model_name)*."""
    slash = model_string.find("/")
    colon = model_string.find(":")
    candidates = [p for p in (slash, colon) if p > 0]
    if not candidates:
        return ("", model_string)
    pos = min(candidates)
    return model_string[:pos], model_string[pos + 1 :]
