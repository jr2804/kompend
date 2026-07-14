"""Shared ACP protocol string constants for KNOX.

Centralises literal values used across the ACP agent modules so no
bare string appears in more than one place.
"""

from __future__ import annotations

from typing import Final

# ── ACP protocol stop reasons ──────────────────────────────────────────
STOP_REASON_END_TURN: Final = "end_turn"
STOP_REASON_MAX_TOKENS: Final = "max_tokens"

# ── ACP session update discriminants ───────────────────────────────────
SESSION_UPDATE_MESSAGE_CHUNK: Final = "agent_message_chunk"
SESSION_UPDATE_AVAILABLE_COMMANDS: Final = "available_commands_update"

# ── Content block type discriminants ───────────────────────────────────
CONTENT_TYPE_TEXT: Final = "text"

# ── Message / session ID prefix ────────────────────────────────────────
MSG_ID_PREFIX: Final = "msg_"
SESSION_ID_PREFIX: Final = "sess_"

# ── Internal session store keys ────────────────────────────────────────
SESSION_KEY_CWD: Final = "cwd"

# ── Compile progress event keys (our --json-progress format) ───────────
PROGRESS_EVENT: Final = "event"
PROGRESS_PHASE: Final = "phase"
PROGRESS_CURRENT: Final = "current"
PROGRESS_TOTAL: Final = "total"
PROGRESS_MESSAGE: Final = "message"
PROGRESS_TYPE_PROGRESS: Final = "progress"
PROGRESS_TYPE_DONE: Final = "done"
PROGRESS_TYPE_ERROR: Final = "error"
