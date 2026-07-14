"""Slash command parsing, definitions, and dispatch for the ACP agent.

Maps ``/knox <command> [args]`` to the corresponding KNOX pipeline call.

Command definitions live in :data:`KNOX_COMMANDS` as structured specs so both
the help text and the ACP ``available_commands_update`` notification are
derived from a single source of truth.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    """Structured definition of a single ``/knox`` slash command."""

    name: str
    description: str
    usage: str


KNOX_COMMANDS: list[CommandSpec] = [
    CommandSpec(
        name="compile",
        description="Compile sources into a wiki",
        usage="/knox compile [--start-at-tier <tier>] [--force]",
    ),
    CommandSpec(
        name="query",
        description="Query the compiled wiki",
        usage="/knox query <question> [--tag <name>]",
    ),
    CommandSpec(
        name="capture",
        description="Capture a note",
        usage="/knox capture <note>",
    ),
    CommandSpec(
        name="status",
        description="Show vault health, tier progress, cache stats",
        usage="/knox status",
    ),
    CommandSpec(
        name="help",
        description="Show available commands or details about a specific command",
        usage="/knox help [<command>]",
    ),
]

_COMMAND_INDEX: dict[str, CommandSpec] = {c.name: c for c in KNOX_COMMANDS}


class SlashCommand:
    """A parsed slash command from an ACP prompt."""

    def __init__(self, name: str, args: list[str], kwargs: dict[str, str]) -> None:
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"SlashCommand({self.name}, {self.args}, {self.kwargs})"


def parse_slash_command(text: str) -> SlashCommand | None:
    """Parse a ``/knox <command> [args]`` line from *text*.

    Returns *None* if the text does not start with ``/knox``.
    """
    text = text.strip()
    if not text.startswith("/knox "):
        return None

    rest = text[len("/knox ") :].strip()
    parts = shlex.split(rest)
    if not parts:
        return None

    name = parts[0]
    args: list[str] = []
    kwargs: dict[str, str] = {}

    i = 1
    while i < len(parts):
        part = parts[i]
        if part.startswith("--"):
            key = part[2:]
            if "=" in key:
                # --key=val
                k, _, v = key.partition("=")
                kwargs[k] = v
            elif i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                # --key val (next token is the value)
                i += 1
                kwargs[key] = parts[i]
            else:
                # --flag (boolean flag)
                kwargs[key] = ""
        else:
            args.append(part)
        i += 1

    return SlashCommand(name, args, kwargs)


# ── Help text (derived from KNOX_COMMANDS) ─────────────────────────────


def _build_help_text() -> str:
    """Generate help text from :data:`KNOX_COMMANDS`."""
    lines = [
        "**KNOX — Knowledge Organizer for eXtended documents**",
        "",
        "Available slash commands:",
        "",
    ]
    for cmd in KNOX_COMMANDS:
        lines.append(f"- ``{cmd.usage}``")
        lines.append(f"  {cmd.description}.")
        lines.append("")
    return "\n".join(lines)


HELP_TEXT = _build_help_text()
