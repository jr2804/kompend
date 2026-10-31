#!/usr/bin/env python
"""The CLI surface follows one rule per input class.

- A **run-level** flag (`--wiki-dir`, `--config`, `--verbose`, `--version`) is
  declared once, on the root meta app. `--config` used to exist on 14 of 63
  commands, six more had an undocumented one because Typer turns a bare
  `config: str | None = None` parameter into an option, and
  `kompend --config x lint all` did not parse at all.
- A **command** declares only its own options; `config` is read from the run
  context through `_shared.config_file_from_ctx`. (`wiki_dir` keeps its
  documented per-command `WikiDir` alias - a pre-existing surface, out of
  scope here.)

Reads the Cyclopts app tree directly: every sub-command is an ``App`` whose
``assemble_argument_collection()`` yields the bound ``Argument`` objects, so
the check needs no click internals.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

from cyclopts import App

from kompend.cli.app import app, register_all_commands

#: Parameter names that belong to the root meta app, never to a command.
RUN_LEVEL_PARAMS: frozenset[str] = frozenset({"config", "config_file"})
#: Options the root app must expose.
RUN_LEVEL_OPTIONS: tuple[str, ...] = ("--config", "-c", "--wiki-dir", "-w")
#: Cyclopts registers ``--help``/``--version`` as synthetic commands on every
#: app; they are not part of the authored surface.
_PSEUDO_COMMANDS: frozenset[str] = frozenset({"help_print", "version_print"})


def main() -> int:
    register_all_commands()
    findings: list[str] = []

    root_options = _option_names(app.meta)
    for option in RUN_LEVEL_OPTIONS:
        if option not in root_options:
            findings.append(f"root app is missing the run-level option {option}")

    checked = 0
    for path, command in _walk(app):
        checked += 1
        label = " ".join(path)
        for param in sorted(_param_names(command) & RUN_LEVEL_PARAMS):
            findings.append(
                f"`kompend {label}` declares a `{param}` parameter - read it from the root context (`config_file_from_ctx` / `vault_paths_from_ctx`) instead"
            )
        if _option_names(command) & {"--config", "-c"}:
            findings.append(f"`kompend {label}` declares its own --config/-c option")

    if checked == 0:
        print("ERROR: the app resolved no sub-commands - is registration broken?", file=sys.stderr)
        return 2

    print(f"check-cli-surface: root options {sorted(root_options)}; {checked} command(s) inspected")
    if findings:
        for finding in findings:
            print(f"  {finding}")
        print(f"\nERROR: {len(findings)} CLI-surface violation(s).", file=sys.stderr)
        return 1
    return 0


def _walk(command: App, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], App]]:
    """Yield every leaf command with its command path."""
    for name, child in command.resolved_commands().items():
        if _is_pseudo(child):
            continue
        current = (*path, name)
        if _real_children(child):
            yield from _walk(child, current)
        else:
            yield current, child


def _real_children(command: App) -> list[str]:
    return [name for name, child in command.resolved_commands().items() if not _is_pseudo(child)]


def _is_pseudo(command: App) -> bool:
    return getattr(command.default_command, "__name__", "") in _PSEUDO_COMMANDS


def _option_names(command: App) -> set[str]:
    """Option spellings declared by ``command``."""
    return {name for argument in _arguments(command) for name in getattr(argument, "names", ()) if name.startswith("-")}


def _param_names(command: App) -> set[str]:
    return {name for argument in _arguments(command) for name in getattr(argument.field_info, "names", ())}


def _arguments(command: App) -> Iterator[object]:
    return iter(command.assemble_argument_collection())


if __name__ == "__main__":
    sys.exit(main())
