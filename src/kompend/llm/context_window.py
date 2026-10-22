"""Discover a model's context window from Ollama or models.dev.

This module is the canonical way to determine how many tokens a model can
accept and generate. It is used by the agent layer and the doctor command.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from kompend.core.config import LLMConfig
from kompend.llm.models import _query_ollama_num_ctx, _user_providers, resolve_model_limits
from kompend.logging import get_logger

#: Share of the window a planner leaves free for the model's answer. No stage
#: declares its own output ceiling: the answer is bounded by physics (window
#: minus prompt), so the only planning question is how much of the window a
#: request must leave unused for the answer it is about to ask for.
_ANSWER_RESERVE_FRACTION: Final[float] = 0.25
#: Ceiling for that reserve. A larger window does not mean a longer answer —
#: a wiki page or a record set tops out well below this.
_ANSWER_RESERVE_MAX: Final[int] = 8192
#: Floor for that reserve, so a tiny window still gets a usable answer.
_ANSWER_RESERVE_MIN: Final[int] = 512

logger = get_logger(__name__)


class ContextSource(StrEnum):
    USER_PROVIDER = "user_provider"
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

    @property
    def answer_reserve(self) -> int:
        """Tokens a request must leave free for the answer."""
        return answer_reserve_tokens(self.max_input)

    def input_budget(self, reserved_tokens: int = 0) -> int:
        """Tokens available for prompt content after the answer reserve.

        *reserved_tokens* is any further fixed overhead the caller knows
        about (a measured system prompt, tool framing, a safety margin).
        """
        return max(1, self.max_input - self.answer_reserve - reserved_tokens)

    @property
    def history_budget(self) -> int:
        """Token size at which threaded conversation history is compacted.

        Twice the answer reserve: enough history to keep the thread, little
        enough that the next answer still fits beside it.
        """
        return 2 * self.answer_reserve


def answer_reserve_tokens(context_window: int) -> int:
    """Tokens to leave free for the answer when planning a request.

    Derived from the window alone: a quarter of it, bounded to
    ``[_ANSWER_RESERVE_MIN, _ANSWER_RESERVE_MAX]``. It is a *planning*
    assumption, not a limit sent to the provider — kompend sends no
    ``max_tokens``, so a model may use more than this when its prompt ended
    up smaller than planned, and the window is what stops it.
    """
    return max(_ANSWER_RESERVE_MIN, min(_ANSWER_RESERVE_MAX, int(context_window * _ANSWER_RESERVE_FRACTION)))


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
    1. ``[providers.<name>]`` override (from kompend.toml) — explicit user
       config beats any discovered value, the same precedence rule
       ``build_model()`` uses for endpoints. Applies to both user-defined
       and integrated providers (e.g. ``[providers.ollama].context_window``).
       Finest granularity wins: per-model → provider-level.
    2. For ``ollama/`` models, query via ``ollama.show()`` — checks
       ``OLLAMA_CONTEXT_LENGTH`` env var then GGUF ``context_length`` metadata.
    3. Query the ``models.dev`` catalog — covers most cloud providers
       and custom OpenAI-compatible mirrors.
    4. If *fallback_max_input* is provided (from ``[compile.llm]`` config),
       use it as-is with **no clamping** — the value the user configured.
       If neither source resolves AND no fallback is provided, raise
       ``LookupError`` — a missing context window usually means the model
       name is wrong, the provider is unreachable, or config needs fixing.

    The typed *llm_config* can be passed instead of individual
    *fallback_max_input* / *fallback_max_output* / *context_window_cap*
    or to supply defaults that explicit parameters override.

    Values from steps 1–2 are used **as-is** — never clamped or capped
    (except by *context_window_cap* if set).  Silently clamping a
    discovered value to a hardcoded range masks misconfiguration (a wrong
    model name or an unreachable provider then looks like a small model)
    instead of surfacing it.
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

    # 1. Explicit [providers.<name>] override (from kompend.toml) — finest
    # granularity wins: per-model context_window → provider-level
    # context_window.  Checked before the probes so user config always beats
    # auto-discovery, mirroring build_model()'s "explicit user config beats
    # implicit catalog".  (``context_window_cap`` is migrated to
    # ``context_window`` when providers are registered — see
    # kompend.llm.user_providers.)
    provider_id, model_name = _split_model(model)
    cfg = _user_providers.get(provider_id)
    if cfg:
        model_cfg = cfg.get("models", {}).get(model_name, {})
        per_model = model_cfg.get("context_window") if isinstance(model_cfg, dict) else None  # nocheck: provider config schema key
        override = per_model if per_model is not None else cfg.get("context_window")  # nocheck: provider config schema key
        if override is not None:
            max_input = int(override)
            max_output = int(override)
            source = ContextSource.USER_PROVIDER

    # 2. Integrated provider auto-discovery: live Ollama probe, then catalog.
    if max_input is None and model.startswith("ollama/"):
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

    # Extract values from typed LLMConfig if provided.
    if llm_config is not None:
        if fallback_max_input is None:
            fallback_max_input = llm_config.fallback_max_input
        if fallback_max_output is None:
            fallback_max_output = llm_config.fallback_max_output
        if context_window_cap is None:
            context_window_cap = llm_config.context_window

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
    """Coerce a raw config ``context_window`` value to ``int | None``.

    Config values reach this point as strings when loaded from TOML with
    env-var interpolation (e.g. ``context_window = "${CTX_WINDOW}"``),
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
        logger.warning("context_window %r is not a valid integer; ignoring", value)
        return None
    try:
        return int(text)
    except ValueError:
        logger.warning("context_window %r is not a valid integer; ignoring", value)
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
