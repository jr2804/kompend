"""Config and scaffolding MCP tools."""

from pathlib import Path
from typing import Any

from knox.core.config import (
    coerce_config_value,
    get_global_config_file,
    get_project_config_file,
    resolve_config,
    set_config_file_value,
    validate_config_key,
)
from knox.logging import get_logger
from knox.scaffolding import ScaffoldError, scaffold_project

logger = get_logger(__name__)


def init_wiki(
    target: str = ".",
    bare: bool = False,
    force: bool = False,
    git: bool = False,
) -> dict[str, Any]:
    """Scaffold a new KNOX wiki project directory."""
    try:
        result = scaffold_project(Path(target), bare=bare, force=force, git=git)
        return {
            "status": "ok",
            "target_dir": str(result.target_dir),
            "created_dirs": [str(d) for d in result.created_dirs],
            "copied_files": [str(f) for f in result.copied_files],
            "skipped_files": list(result.skipped_files) if result.skipped_files else [],
            "git_initialised": result.git_initialised,
        }
    except ScaffoldError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def config_get(
    key: str,
) -> dict[str, Any]:
    """Print a resolved config value."""
    if not validate_config_key(key):
        return {"status": "error", "message": f"Unknown config key: {key}"}
    try:
        cfg = resolve_config()
        value = cfg.get(key)
        return {"status": "ok", "key": key, "value": value}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def config_set(
    key: str,
    value: str,
    global_: bool = False,
) -> dict[str, Any]:
    """Set a config value."""
    if not validate_config_key(key):
        return {"status": "error", "message": f"Unknown config key: {key}"}
    try:
        coerced = coerce_config_value(value, key)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    if global_:
        config_file = get_global_config_file(create=True)
    else:
        config_file = get_project_config_file()
        if not config_file.exists():
            return {"status": "error", "message": "Not in a KNOX vault â€” run `knox init` first or use `--global`"}
    try:
        set_config_file_value(config_file, key, coerced)
        return {"status": "ok", "key": key, "value": coerced, "file": str(config_file)}
    except (OSError, ValueError) as exc:
        return {"status": "error", "message": f"Failed to write {config_file}: {exc}"}
