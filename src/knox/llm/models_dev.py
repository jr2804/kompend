"""Model context-window discovery via litellm.

Returns ``(max_input, max_output)`` token limits for *model_id*, or
``None`` when unknown. Model not found is never fatal — callers fall
through to their own defaults.
"""

from __future__ import annotations

from typing import Any

import litellm

from knox.logging import get_logger

logger = get_logger(__name__)

# ponytail: remove hard-coded fallbacks. If it's not defined, let it burn.
# Common local models litellm doesn't know — keyed by model name (after
# provider/ prefix). Add entries only when users report a context mismatch.
_FALLBACK: dict[str, tuple[int, int]] = {
    "granite-embedding:278m": (8192, 8192),
    "nomic-embed-text": (8192, 8192),
    "all-MiniLM-L6-v2": (512, 512),
}


def resolve_model_limits(
    model_id: str,
    **kwargs: Any,  # noqa: ARG001 — accept `catalog` param for backward compat
) -> tuple[int, int] | None:
    """Resolve ``(max_input, max_output)`` token limits for *model_id*.

    Uses ``litellm.get_model_info()``. Falls back to a small hardcoded map
    for local models litellm doesn't track. Returns ``None`` when unknown.
    """
    # Strip provider prefix and tag for lookup
    _, _, stem = model_id.partition("/")
    base = (stem if stem else model_id).split(":", 1)[0]
    lookup = stem or model_id

    # Try litellm first
    try:
        info = litellm.get_model_info(lookup)
        if isinstance(info, dict):
            max_in = info.get("max_input_tokens") or info.get("max_tokens")
            max_out = info.get("max_output_tokens") or max_in
            if max_in and max_out:
                return int(max_in), int(max_out)
    except Exception:  # noqa: BLE001 — optional probe, never fatal
        logger.debug("litellm.get_model_info(%r) failed", lookup, exc_info=True)

    # Fallback: known local/embedding models
    for key, limits in _FALLBACK.items():
        if key == lookup or key == base:
            return limits

    return None
