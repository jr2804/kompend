"""Lazy command group: list every command, import only the invoked one.

Typer materialises a command object when ``@app.command()`` runs, which happens
while its module is imported. This group therefore reports command names from the
registry without importing anything — so ``kompend --help`` stays fast and still
names every command — and imports a command's module on first use, attaching the
command it just registered.

The context parameter of the two overrides below is unused: they decide from the
command name and the registry alone. It is declared ``object`` rather than a
framework type so the signatures do not name something they never read.
"""

from __future__ import annotations

import importlib
from typing import ClassVar, override

import typer
from typer.core import TyperCommand, TyperGroup
from typer.main import get_command_from_info, get_group

from kompend.cli._registry import COMMANDS, PANEL_ORDER, CommandName


class LazyGroup(TyperGroup):
    """A Typer group that imports a command's module only when it is used."""

    #: The application whose registered commands this group serves. Typer
    #: instantiates the group itself, from its own fixed keyword arguments, so the
    #: reference is bound once by ``app.py`` rather than passed in.
    app: ClassVar[typer.Typer | None] = None

    @override
    def list_commands(self, ctx: object) -> list[str]:
        """Command names in workflow order, importing none of their modules."""
        order = {panel: index for index, panel in enumerate(PANEL_ORDER)}
        ordered = sorted(COMMANDS, key=lambda name: (order[COMMANDS[name].panel], name.value))
        return [name.value for name in ordered]

    @override
    def get_command(self, ctx: object, cmd_name: str) -> TyperCommand | TyperGroup | None:
        """Serve ``cmd_name``, importing its module the first time it is asked for."""
        owner = self._owner()
        try:
            name = CommandName(cmd_name)
        except ValueError:
            return None
        if cmd_name not in self.commands:
            importlib.import_module(COMMANDS[name].module.value)
            self._attach(name, owner)
        command = self.commands.get(cmd_name)
        if command is None:
            return None
        if isinstance(command, TyperCommand | TyperGroup):
            command.rich_help_panel = COMMANDS[name].panel.value
            return command
        return None

    @classmethod
    def bind(cls, app: typer.Typer) -> None:
        """Bind the application whose command tree this group serves."""
        cls.app = app

    def _attach(self, name: CommandName, owner: typer.Typer) -> None:
        """Attach the command ``name``'s module just registered on its import."""
        for info in owner.registered_commands:
            command = get_command_from_info(
                info,
                pretty_exceptions_short=True,
                rich_markup_mode=self.rich_markup_mode,
            )
            if command.name == name.value:
                self.add_command(command)
                return
        for group_info in owner.registered_groups:
            if group_info.typer_instance is None:
                continue
            group_name = group_info.name or group_info.typer_instance.info.name or ""
            if group_name == name.value:
                sub_group = get_group(group_info.typer_instance)
                sub_group.name = group_name
                self.add_command(sub_group)
                return

    @classmethod
    def _owner(cls) -> typer.Typer:
        """The bound application, or a failure naming the missing binding."""
        if cls.app is None:
            msg = "LazyGroup.bind(app) must run before the CLI is invoked"
            raise RuntimeError(msg)
        return cls.app
