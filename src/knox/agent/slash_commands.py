"""Slash command parsing and dispatch for the ACP agent.

Maps ``/knox <command> [args]`` to the corresponding KNOX pipeline call.
"""

from __future__ import annotations

import shlex


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


# ── Help text ──────────────────────────────────────────────────────────

HELP_TEXT = """\
**KNOX — Knowledge Organizer for eXtended documents**

Available slash commands:

- ``/knox compile [--start-at-tier <tier>] [--force]``
  Compile sources into a wiki. Same as ``knox compile``.

- ``/knox query <question> [--tag <name>]``
  Query the compiled wiki. Same as ``knox query``.

- ``/knox capture <note>``
  Capture a note. Same as ``knox capture``.

- ``/knox status``
  Show vault health, tier progress, cache stats.

- ``/knox help [<command>]``
  Show this help or details about a specific command.
"""
