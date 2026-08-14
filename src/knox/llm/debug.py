"""Per-LLM-call debug logging for KNOX agents.

When the ``KNOX_DEBUG_LLM_DIR`` environment variable is set, every LLM call
is dumped to a timestamped JSON file containing the full message list, model
settings, and response. This module centralises that logic so it can be
imported without coupling the model to the debug-sink details.

The calling agent is passed in directly; its ``stage`` and ``current_source_id``
attributes supply the per-dump context, so callers never hard-code a stage
string or repeat the assembly logic.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic_ai import ModelMessage
from pydantic_ai.messages import (
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)

from knox.llm.format import salvage_json
from knox.logging import get_logger

logger = get_logger(__name__)


@dataclass
class _Payload:
    """One debug dump written to ``KNOX_DEBUG_LLM_DIR``."""

    model: str
    stage: str
    source_id: str
    timestamp: str
    user_turns: int
    assistant_turns: int
    messages: list[dict[str, Any]]
    kwargs: dict[str, Any]
    response_parsed: dict[str, Any] | list[Any] | None
    response: str
    finish_reason: str | None


class _DebugSource(Protocol):
    """Minimal interface ``save_debug_prompt`` needs from an agent."""

    @property
    def stage(self) -> str:
        """Pipeline stage label (e.g. ``extraction``, ``wicer``, ``vlm``)."""

    @property
    def current_source_id(self) -> str:
        """Source id of the most recent ``run()`` call."""


def save_debug_prompt(
    agent: _DebugSource,
    model_string: str,
    messages: list[dict[str, Any]],
    kwargs: dict[str, Any],
    response_content: str,
    finish_reason: str | None,
    *,
    user_turns: int = 0,
    assistant_turns: int = 0,
) -> Path | None:
    """Save the LLM prompt and response to a debug file.

    Triggered by ``KNOX_DEBUG_LLM_DIR`` env var. Each call creates a
    timestamped JSON file with the full message list, model settings,
    and parsed response.

    The dump records the agent's ``stage`` (compile, wicer, twostage, …)
    and the number of user / assistant turns so sessions that needed
    correction (extra retries, validation feedback) are easy to spot.

    The raw ``response`` is only included when it differs from
    ``response_parsed`` — for a clean one-shot exchange the two are
    redundant, but when the model needed repair or emitted unexpected
    text the divergence is preserved.
    """
    # Dev-only debug hook; deliberately read directly, not a runtime config key.
    debug_dir = os.environ.get("KNOX_DEBUG_LLM_DIR")
    if not debug_dir:
        return
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Normalise model string for filename: strip provider prefix (e.g. "ollama/")
    # so the same model always produces the same slug regardless of how it was
    # passed (with or without provider prefix).
    model_slug = model_string.split("/", 1)[-1] if "/" in model_string else model_string
    model_slug = model_slug.replace("/", "_").replace(":", "")
    filepath = debug_dir / f"llm_{ts}_{model_slug}.json"

    parsed = salvage_json(response_content)
    parsed_obj = parsed if isinstance(parsed, (dict, list)) else None

    payload = _Payload(
        model=model_string,
        stage=agent.stage,
        source_id=agent.current_source_id,
        timestamp=datetime.now().isoformat(),
        user_turns=user_turns,
        assistant_turns=assistant_turns,
        messages=messages,
        kwargs={k: v for k, v in kwargs.items() if k not in ("messages", "model")},
        response_parsed=parsed_obj,
        response=response_content,
        finish_reason=finish_reason,
    )

    try:
        # errors="backslashreplace": messages may carry lone surrogates (e.g.
        # re-loaded from parse caches written before ingestion scrubbing);
        # they encode as \udXXX escapes — still valid, round-trippable JSON —
        # instead of aborting the dump with a zero-byte file.
        filepath.write_text(
            json.dumps(asdict(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
            errors="backslashreplace",
        )
        logger.info("Debug prompt saved to %s", filepath)
        return filepath
    except Exception as exc:
        logger.warning("Failed to save debug prompt: %s", exc)
        return None


# ── Message serialisation helpers ───────────────────────────────────────


def messages_to_debug_dicts(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Convert pydantic-ai ModelMessages to debug-friendly dicts.

    Each message is serialised with its role (``ModelRequest`` / ``ModelResponse``)
    and a list of structured part dicts.  Parts are serialised with their type
    (``text``, ``thinking``, ``tool_call``, ``retry_prompt``) and type-specific
    content so the debug file is readable without mental parsing of ``str(part)``.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        kind = type(msg).__name__
        parts_list: list[dict[str, Any]] = []
        for part in getattr(msg, "parts", []):
            parts_list.append(_part_to_debug_dict(part))
        result.append({"role": kind, "parts": parts_list})
    return result


def _part_to_debug_dict(part: Any) -> dict[str, Any]:  # noqa: PLR0911
    """Convert a single ModelPart to a debug-friendly dict."""
    if isinstance(part, SystemPromptPart):
        return {"type": "system_prompt", "content": part.content}
    if isinstance(part, TextPart):
        return {"type": "text", "content": part.content}
    if isinstance(part, ThinkingPart):
        return {"type": "thinking", "content": part.content}
    if isinstance(part, ToolCallPart):
        return {
            "type": "tool_call",
            "tool_name": part.tool_name,
            "args": part.args,
            "tool_call_id": part.tool_call_id,
        }
    if isinstance(part, ToolReturnPart):
        return {
            "type": "tool_return",
            "tool_name": part.tool_name,
            "content": part.content,
        }
    if isinstance(part, RetryPromptPart):
        content = part.content
        # RetryPromptPart.content can be a list of validation errors
        if isinstance(content, list):
            return {"type": "retry_prompt", "errors": content}
        return {"type": "retry_prompt", "content": str(content)}
    return {"type": type(part).__name__, "content": str(part)}
