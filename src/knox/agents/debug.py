"""Per-LLM-call debug logging for KNOX agents.

When the ``KNOX_DEBUG_LLM_DIR`` environment variable is set, every LLM call
is dumped to a timestamped JSON file containing the full message list, model
settings, and response. This module centralises that logic so it can be
imported without coupling the litellm bridge to the debug-sink details.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from knox.llm.format import salvage_json
from knox.logging import get_logger

logger = get_logger(__name__)


def save_debug_prompt(
    model_string: str,
    litellm_messages: list[dict[str, Any]],
    kwargs: dict[str, Any],
    response_content: str,
    finish_reason: str | None,
    *,
    source_id: str = "",
    stage: str = "",
) -> None:
    """Save the LLM prompt and response to a debug file.

    Triggered by ``KNOX_DEBUG_LLM_DIR`` env var. Each call creates a
    timestamped JSON file with the full message list, model settings,
    and parsed response.

    The dump records the pipeline ``stage`` (compile, wicer, twostage, …)
    and the number of user / assistant turns so sessions that needed
    correction (extra retries, validation feedback) are easy to spot.

    The raw ``response`` is only included when it differs from
    ``response_parsed`` — for a clean one-shot exchange the two are
    redundant, but when the model needed repair or emitted unexpected
    text the divergence is preserved.
    """
    debug_dir = os.environ.get("KNOX_DEBUG_LLM_DIR")
    if not debug_dir:
        return
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_slug = model_string.replace("/", "_").replace(":", "")
    filepath = debug_path / f"llm_{ts}_{model_slug}.json"

    user_turns = sum(1 for m in litellm_messages if m.get("role") == "user")
    assistant_turns = sum(1 for m in litellm_messages if m.get("role") == "assistant")

    parsed = salvage_json(response_content)
    parsed_str = json.dumps(parsed, ensure_ascii=False) if parsed is not None else None

    payload: dict[str, Any] = {
        "model": model_string,
        "stage": stage,
        "source_id": source_id,
        "timestamp": datetime.now().isoformat(),
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "messages": litellm_messages,
        "kwargs": {k: v for k, v in kwargs.items() if k not in ("messages", "model")},
        "response_parsed": parsed_str,
        "finish_reason": finish_reason,
    }
    # Only keep the raw response when it diverges from the parsed result.
    # For a clean one-shot exchange (one prompt, one valid reply) the two are
    # equivalent and the raw field is redundant. When the model needed repair
    # or emitted unexpected text, the divergence is preserved for diagnosis.
    if parsed is None or salvage_json(response_content) != parsed:
        payload["response"] = response_content

    try:
        filepath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Debug prompt saved to %s", filepath)
    except Exception as exc:
        logger.warning("Failed to save debug prompt: %s", exc)
