"""Config and scaffolding MCP tools."""

from pathlib import Path
from typing import Any

from kompend.core.config import (
    check_config_file,
    check_config_files,
    coerce_config_value,
    get_global_config_file,
    get_project_config_file,
    resolve_config,
    set_config_file_value,
    validate_config_key,
)
from kompend.core.keys import PATHS_WIKI_DIR
from kompend.core.utils.dicts import as_int
from kompend.logging import get_logger
from kompend.scaffolding import ScaffoldError, scaffold_project

_ENV_REF_RE = __import__("re").compile(r"^\$\{\w+\}$")

logger = get_logger(__name__)


def init_wiki(
    target: str = ".",
    bare: bool = False,
    force: bool = False,
    git: bool = False,
) -> dict[str, Any]:
    """Scaffold a new kompend wiki project directory."""
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
            return {"status": "error", "message": "Not in a kompend vault â€” run `kompend init` first or use `--global`"}
    try:
        set_config_file_value(config_file, key, coerced)
        return {"status": "ok", "key": key, "value": coerced, "file": str(config_file)}
    except (OSError, ValueError) as exc:
        return {"status": "error", "message": f"Failed to write {config_file}: {exc}"}


def config_check(
    paths: list[str],
    merge: bool = True,
) -> dict[str, Any]:
    """Diff one or more config TOML files against the built-in schema defaults.

    Returns a dict with ``ok`` / ``modified`` / ``error`` counts and an
    ``entries`` list (each entry has ``key``, ``status``, ``default``,
    and where applicable ``current`` and ``message``).
    """
    file_paths = [Path(p) for p in paths]
    try:
        if merge and len(file_paths) > 1:
            report = check_config_files(file_paths)
            payload = [report.to_dict()]
        else:
            payload = [check_config_file(p).to_dict() for p in file_paths]
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    totals: dict[str, int] = {"ok": 0, "modified": 0, "error": 0}
    for report in payload:
        for key in ("ok", "modified", "error"):
            value = as_int(report[key])
            if value is None:
                raise ValueError(f"Config check report is missing the {key!r} count")
            totals[key] += value

    return {
        "status": "ok" if totals["error"] == 0 else "error",
        "totals": totals,
        "reports": payload,
    }


def config_list(
    wiki_dir: str | None = None,
    global_: bool = False,
) -> dict[str, Any]:
    """List all resolved config values with their provenance."""
    overrides = {}
    if wiki_dir:
        overrides[PATHS_WIKI_DIR] = str(wiki_dir)
    try:
        cfg = resolve_config(cli_overrides=overrides)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    entries = []
    for key, value in sorted(cfg.flat().items()):
        source = cfg.source_of(key)
        redacted = _redact_api_key(key, value)
        entries.append({"key": key, "value": redacted, "source": source})
    return {"status": "ok", "count": len(entries), "entries": entries}


def config_path(
    wiki_dir: str | None = None,
    global_: bool = False,
) -> dict[str, Any]:
    """Show the config file path (project or global)."""
    try:
        path = get_global_config_file(create=False) if global_ else get_project_config_file(start_dir=Path(wiki_dir) if wiki_dir else None)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
    return {"status": "ok", "path": str(path)}


def _redact_api_key(key: str, value: Any) -> Any:
    """Hide provider API keys in output; env-var refs (${VAR}) stay visible."""
    if key.rsplit(".", maxsplit=1)[-1] == "api_key" and isinstance(value, str) and not _ENV_REF_RE.match(value):
        return "***"
    return value
