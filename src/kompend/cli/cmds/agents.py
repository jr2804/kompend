"""CLI commands for managing TOML-defined conversational agents.

- ``kompend agents list`` — list all agents grouped by task
- ``kompend agents use <task> <name>`` — set active agent for a task
- ``kompend agents validate <file>`` — validate a TOML agent definition
- ``kompend agents install <file>`` — install an agent TOML into the vault
- ``kompend agents remove <name>`` — remove an installed agent from the vault
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from kompend.acp.definitions import AgentDefinition, AgentTask
from kompend.acp.registry import AgentRegistry
from kompend.cli.app import app
from kompend.cli.args.serving import AgentsAction, AgentsName, AgentsTask
from kompend.core.config import resolve_config, validate_config_key
from kompend.core.config._core import set_config_file_value
from kompend.core.keys import (
    AGENTS_CAPTURE,
    AGENTS_COMPILE,
    AGENTS_ORGANIZE,
    AGENTS_QUERY,
)
from kompend.core.paths import KOMP_CONFIG_FILE
from kompend.logging import get_console, get_logger

_TASK_CONFIG_KEY: dict[AgentTask, str] = {
    AgentTask.COMPILE: AGENTS_COMPILE,
    AgentTask.QUERY: AGENTS_QUERY,
    AgentTask.CAPTURE: AGENTS_CAPTURE,
    AgentTask.ORGANIZE: AGENTS_ORGANIZE,
}

logger = get_logger(__name__)


@app.command(name="agents")
def agents_cmd(
    action: AgentsAction,
    task: AgentsTask = None,
    name: AgentsName = None,
) -> None:
    """Manage TOML-defined conversational agents."""
    if action == "list":
        _agents_list()
    elif action == "use":
        if task is None or name is None:
            get_console().print("[red]Usage: kompend agents use <task> <name>[/red]")
            raise typer.Exit(1)
        _agents_use(task, name)
    elif action == "validate":
        if task is None:
            get_console().print("[red]Usage: kompend agents validate <file>[/red]")
            raise typer.Exit(1)
        _agents_validate(task)
    elif action == "install":
        if task is None:
            get_console().print("[red]Usage: kompend agents install <file>[/red]")
            raise typer.Exit(1)
        _agents_install(task)
    elif action == "remove":
        if task is None:
            get_console().print("[red]Usage: kompend agents remove <name>[/red]")
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

    if not validate_config_key(config_key):
        get_console().print(f"[red]Invalid config key: {config_key}[/red]")
        raise typer.Exit(1)

    config_file = Path(base_dir) / KOMP_CONFIG_FILE
    set_config_file_value(config_file, config_key, name)
    get_console().print(f"[green]Active agent for {task.value}: {name}[/green]")


def _agents_validate(file: str) -> None:
    """Validate a TOML agent definition file."""
    source_path = Path(file)
    if not source_path.exists():
        get_console().print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    try:
        agent = AgentDefinition.from_toml(source_path)
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


def _agents_install(file: str) -> None:
    """Install an agent TOML file into the vault."""
    agent_file = Path(file)
    if not agent_file.exists():
        get_console().print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    try:
        agent = AgentDefinition.from_toml(agent_file)
    except Exception as exc:
        get_console().print(f"[red]Invalid agent definition: {exc}[/red]")
        raise typer.Exit(1) from exc

    base_dir = _resolve_base_dir()
    vault_dir = Path(base_dir) / ".kompend" / "agents"
    vault_dir.mkdir(parents=True, exist_ok=True)
    dest = vault_dir / f"{agent.name}.toml"

    if dest.exists():
        get_console().print(f"[yellow]Overwriting existing agent: {agent.name}[/yellow]")

    shutil.copy2(agent_file, dest)
    get_console().print(f"[green]Installed agent: {agent.name} ({agent.task.value})[/green]")


def _agents_remove(name: str) -> None:
    """Remove an installed agent from the vault."""
    base_dir = _resolve_base_dir()
    vault_dir = Path(base_dir) / ".kompend" / "agents"
    agent_file = vault_dir / f"{name}.toml"

    if not agent_file.exists():
        get_console().print(f"[red]Agent not found in vault: {name}[/red]")
        raise typer.Exit(1)

    agent_file.unlink()
    get_console().print(f"[green]Removed agent: {name}[/green]")

    if vault_dir.exists() and not any(vault_dir.iterdir()):
        vault_dir.rmdir()


def _resolve_base_dir() -> str:
    """Resolve the vault base_dir from config."""
    cfg = resolve_config()
    return cfg.kompend.paths.wiki_dir or "."
