#!/usr/bin/env python
"""The CLI surface follows one rule per input class.

- A **run-level** flag (`--wiki-dir`, `--config`, `--verbose`, `--version`) is
  declared once, on the root callback. `--config` used to exist on 14 of 63
  commands, six more had an undocumented one because Typer turns a bare
  `config: str | None = None` parameter into an option, and
  `kompend --config x lint all` did not parse at all.
- A **command** declares only its own options; `config` is read from the
  Typer context through `_shared.config_file_from_ctx`. (`wiki_dir` keeps its
  documented per-command `WikiDir` alias - a pre-existing surface, out of
  scope here.)
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import click
from typer.main import get_command

from kompend.cli.app import app, register_all_commands

#: Parameter names that belong to the root callback, never to a command.
RUN_LEVEL_PARAMS: frozenset[str] = frozenset({"config", "config_file"})
#: Options the root app must expose.
RUN_LEVEL_OPTIONS: tuple[str, ...] = ("--config", "-c", "--wiki-dir", "-w")


def main() -> int:
    register_all_commands()
    root = get_command(app)
    if not getattr(root, "commands", None):
        print("ERROR: the root command has no sub-commands - is registration broken?", file=sys.stderr)
        return 2

    findings: list[str] = []

    root_options: set[str] = set()
    if root.params:
        for param in root.params:
            root_options.update(param.opts)
            root_options.update(param.secondary_opts)
    for option in RUN_LEVEL_OPTIONS:
        if option not in root_options:
            findings.append(f"root app is missing the run-level option {option}")

    checked = 0
    for path, command in _walk(root):
        checked += 1
        label = " ".join(path)
        for param in command.params:
            options = set(param.opts) | set(param.secondary_opts)
            if param.name in RUN_LEVEL_PARAMS:
                findings.append(
                    f"`kompend {label}` declares a `{param.name}` parameter - read it from the root "
                    f"context (`config_file_from_ctx` / `vault_paths_from_ctx`) instead"
                )
            elif options & {"--config", "-c"}:
                findings.append(f"`kompend {label}` declares its own --config/-c option")

    print(f"check-cli-surface: root options {sorted(root_options)}; {checked} command(s) inspected")
    if findings:
        for finding in findings:
            print(f"  {finding}")
        print(f"\nERROR: {len(findings)} CLI-surface violation(s).", file=sys.stderr)
        return 1
    return 0


def _walk(group: object, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], click.Command]]:
    """Yield every leaf command with its command path.

    Typer vendors click, so ``LazyGroup`` (a ``TyperGroup``) is not a
    ``click.Group``; a command is a group when it exposes ``commands``.
    """
    commands = getattr(group, "commands", None)
    if not isinstance(commands, dict):
        return
    for name, command in commands.items():
        current = (*path, name)
        if getattr(command, "commands", None):
            yield from _walk(command, current)
        else:
            yield current, command


if __name__ == "__main__":
    sys.exit(main())
