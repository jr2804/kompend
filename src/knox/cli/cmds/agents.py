"""CLI commands for managing TOML-defined conversational agents.

- ``knox agents list`` — list all agents grouped by task
- ``knox agents use <task> <name>`` — set active agent for a task
- ``knox agents validate <file>`` — validate a TOML agent definition
- ``knox agents install <file>`` — install an agent TOML into the vault
- ``knox agents remove <name>`` — remove an installed agent from the vault
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from knox.agent.definitions import AgentDefinition, AgentTask
from knox.agent.registry import AgentRegistry
from knox.cli.app import app
from knox.core.config import resolve_config, validate_config_key
from knox.core.config._core import set_config_file_value
from knox.core.config.keys import (
    AGENTS_CAPTURE,
    AGENTS_COMPILE,
    AGENTS_ORGANIZE,
    AGENTS_QUERY,
    PATHS_WIKI_DIR,
)
from knox.core.paths import KNOX_CONFIG_FILE
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
    return cfg.get(PATHS_WIKI_DIR, ".")


@app.command(name="agents")
def agents_cmd(
    action: str = typer.Argument(help="Action: list, use, validate, install, or remove"),
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
    elif action == "install":
        if task is None:
            get_console().print("[red]Usage: knox agents install <file>[/red]")
            raise typer.Exit(1)
        _agents_install(task)
    elif action == "remove":
        if task is None:
            get_console().print("[red]Usage: knox agents remove <name>[/red]")
            raise typer.Exit(1)
        _agents_remove(task)
    else:
        get_console().print(f"[red]Unknown action: {action}. Use list, use, validate, install, or remove.[/red]")
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
    if not validate_config_key(config_key):
        get_console().print(f"[red]Invalid config key: {config_key}[/red]")
        raise typer.Exit(1)

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


def _agents_install(file_path: str) -> None:
    """Install an agent TOML file into the vault."""
    src = Path(file_path)
    if not src.exists():
        get_console().print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    # Validate before installing
    try:
        agent = AgentDefinition.from_toml(src)
    except Exception as exc:
        get_console().print(f"[red]Invalid agent definition: {exc}[/red]")
        raise typer.Exit(1) from exc

    base_dir = _resolve_base_dir()
    vault_dir = Path(base_dir) / ".knox" / "agents"
    vault_dir.mkdir(parents=True, exist_ok=True)
    dest = vault_dir / f"{agent.name}.toml"

    if dest.exists():
        get_console().print(f"[yellow]Overwriting existing agent: {agent.name}[/yellow]")

    shutil.copy2(src, dest)
    get_console().print(f"[green]Installed agent: {agent.name} ({agent.task.value})[/green]")


def _agents_remove(name: str) -> None:
    """Remove an installed agent from the vault."""
    base_dir = _resolve_base_dir()
    vault_dir = Path(base_dir) / ".knox" / "agents"
    agent_file = vault_dir / f"{name}.toml"

    if not agent_file.exists():
        get_console().print(f"[red]Agent not found in vault: {name}[/red]")
        raise typer.Exit(1)

    agent_file.unlink()
    get_console().print(f"[green]Removed agent: {name}[/green]")

    # Clean up empty vault dir
    if vault_dir.exists() and not any(vault_dir.iterdir()):
        vault_dir.rmdir()
