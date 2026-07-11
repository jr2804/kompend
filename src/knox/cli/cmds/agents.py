"""CLI commands for managing TOML-defined conversational agents.

- ``knox agents list`` — list all agents grouped by task
- ``knox agents use <task> <name>`` — set active agent for a task
- ``knox agents validate <file>`` — validate a TOML agent definition
"""

from __future__ import annotations

from pathlib import Path

import typer

from knox.agent.definitions import AgentDefinition, AgentTask
from knox.agent.registry import AgentRegistry
from knox.cli.app import app
from knox.core.config import resolve_config
from knox.core.config.keys import (
    AGENTS_CAPTURE,
    AGENTS_COMPILE,
    AGENTS_ORGANIZE,
    AGENTS_QUERY,
)
from knox.logging import get_console, get_logger

logger = get_logger(__name__)

_TASK_CONFIG_KEY: dict[AgentTask, str] = {
    AgentTask.COMPILE: AGENTS_COMPILE,
    AgentTask.QUERY: AGENTS_QUERY,
    AgentTask.CAPTURE: AGENTS_CAPTURE,
    AgentTask.ORGANIZE: AGENTS_ORGANIZE,
}


def _resolve_base_dir() -> str:
    """Resolve the vault base_dir from config."""
    cfg = resolve_config()
    return cfg.get("paths.wiki_dir", ".")


@app.command(name="agents")
def agents_cmd(
    action: str = typer.Argument(help="Action: list, use, or validate"),
    task: str | None = typer.Argument(None, help="Task name (for 'use'): compile, query, capture, organize"),
    name: str | None = typer.Argument(None, help="Agent name (for 'use') or file path (for 'validate')"),
) -> None:
    """Manage TOML-defined conversational agents."""
    if action == "list":
        _agents_list()
    elif action == "use":
        if task is None or name is None:
            get_console().print("[red]Usage: knox agents use <task> <name>[/red]")
            raise typer.Exit(1)
        _agents_use(task, name)
    elif action == "validate":
        if task is None:
            get_console().print("[red]Usage: knox agents validate <file>[/red]")
            raise typer.Exit(1)
        _agents_validate(task)
    else:
        get_console().print(f"[red]Unknown action: {action}. Use list, use, or validate.[/red]")
        raise typer.Exit(1)


def _agents_list() -> None:
    """List all agents grouped by task, marking active ones with *."""
    base_dir = _resolve_base_dir()
    registry = AgentRegistry(base_dir=base_dir)
    cfg = resolve_config()
    grouped = registry.list_by_task()

    console = get_console()
    for task in AgentTask:
        agents = grouped.get(task, [])
        if not agents:
            continue
        config_key = _TASK_CONFIG_KEY.get(task, "")
        active_name = cfg.get(config_key) if config_key else None
        console.print(f"\n[bold]{task.value}[/bold]")
        for agent in agents:
            marker = " [green]*[/green]" if agent.name == active_name else "  "
            console.print(f"{marker} {agent.name} — {agent.description}")
    console.print("\n[dim]* = active[/dim]")


def _agents_use(task_str: str, name: str) -> None:
    """Set the active agent for a task."""
    try:
        task = AgentTask(task_str)
    except ValueError:
        get_console().print(f"[red]Unknown task: {task_str}. Use: {', '.join(t.value for t in AgentTask)}[/red]")
        raise typer.Exit(1) from None

    base_dir = _resolve_base_dir()
    registry = AgentRegistry(base_dir=base_dir)
    agent = registry.get(name)
    if agent is None:
        get_console().print(f"[red]Agent not found: {name}[/red]")
        raise typer.Exit(1)

    config_key = _TASK_CONFIG_KEY.get(task)
    if config_key is None:
        get_console().print(f"[red]Cannot set active agent for task: {task}[/red]")
        raise typer.Exit(1)

    # Write to config.toml
    from knox.core.config import validate_config_key  # noqa: PLC0415

    if not validate_config_key(config_key):
        get_console().print(f"[red]Invalid config key: {config_key}[/red]")
        raise typer.Exit(1)

    from knox.core.config._core import set_config_file_value  # noqa: PLC0415
    from knox.core.paths import KNOX_CONFIG_FILE  # noqa: PLC0415

    config_path = Path(base_dir) / KNOX_CONFIG_FILE
    set_config_file_value(config_path, config_key, name)
    get_console().print(f"[green]Active agent for {task.value}: {name}[/green]")


def _agents_validate(file_path: str) -> None:
    """Validate a TOML agent definition file."""
    path = Path(file_path)
    if not path.exists():
        get_console().print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    try:
        agent = AgentDefinition.from_toml(path)
        get_console().print("[green]✓ Valid agent definition[/green]")
        get_console().print(f"  name: {agent.name}")
        get_console().print(f"  task: {agent.task.value}")
        get_console().print(f"  description: {agent.description}")
        if agent.llm.model:
            get_console().print(f"  model: {agent.llm.model}")
        if agent.tools.enabled:
            get_console().print(f"  tools: {', '.join(agent.tools.enabled)}")
        else:
            get_console().print("  tools: all")
    except Exception as exc:
        get_console().print(f"[red]✗ Invalid: {exc}[/red]")
        raise typer.Exit(1) from exc
