"""Config and scaffolding MCP tools."""

from pathlib import Path
from typing import Any

from kompend.core.config import (
    STATUS_ERROR,
    STATUS_OK,
    check_config_file,
    check_config_files,
    coerce_config_value,
    get_global_config_file,
    get_project_config_file,
    resolve_config,
    set_config_file_value,
    validate_config_key,
)
from kompend.core.keys import PATHS_OUTPUT_DIR, PATHS_SOURCES_DIR, PATHS_WIKI_DIR
from kompend.core.paths import DEFAULT_OUTPUT_DIR_NAME, DEFAULT_SOURCES_DIR_NAME, GITIGNORE_FILE
from kompend.core.utils.dicts import as_int, as_str
from kompend.logging import get_logger
from kompend.mcp.tools._helpers import ToolStatus
from kompend.scaffolding import ScaffoldError, dir_name, is_current, merge_gitignore, render_gitignore, scaffold_project

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
            "status": ToolStatus.OK,
            "target_dir": str(result.target_dir),
            "created_dirs": [str(d) for d in result.created_dirs],
            "copied_files": [str(f) for f in result.copied_files],
            "skipped_files": list(result.skipped_files) if result.skipped_files else [],
            "git_initialised": result.git_initialised,
        }
    except ScaffoldError as exc:
        return {"status": ToolStatus.ERROR, "message": str(exc)}
    except Exception as exc:
        return {"status": ToolStatus.ERROR, "message": str(exc)}


def config_get(
    key: str,
) -> dict[str, Any]:
    """Print a resolved config value."""
    if not validate_config_key(key):
        return {"status": ToolStatus.ERROR, "message": f"Unknown config key: {key}"}
    try:
        cfg = resolve_config()
        value = cfg.get(key)
        return {"status": ToolStatus.OK, "key": key, "value": value}
    except Exception as exc:
        return {"status": ToolStatus.ERROR, "message": str(exc)}


def config_set(
    key: str,
    value: str,
    global_: bool = False,
) -> dict[str, Any]:
    """Set a config value."""
    if not validate_config_key(key):
        return {"status": ToolStatus.ERROR, "message": f"Unknown config key: {key}"}
    try:
        coerced = coerce_config_value(value, key)
    except ValueError as exc:
        return {"status": ToolStatus.ERROR, "message": str(exc)}

    if global_:
        config_file = get_global_config_file(create=True)
    else:
        config_file = get_project_config_file()
        if not config_file.exists():
            return {"status": ToolStatus.ERROR, "message": "Not in a kompend vault â€” run `kompend init` first or use `--global`"}
    try:
        set_config_file_value(config_file, key, coerced)
        return {"status": ToolStatus.OK, "key": key, "value": coerced, "file": str(config_file)}
    except (OSError, ValueError) as exc:
        return {"status": ToolStatus.ERROR, "message": f"Failed to write {config_file}: {exc}"}


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
        return {"status": ToolStatus.ERROR, "message": str(exc)}

    # The report counts are keyed by the config-check vocabulary.
    totals: dict[str, int] = {STATUS_OK: 0, "modified": 0, STATUS_ERROR: 0}
    for report in payload:
        for key in (STATUS_OK, "modified", STATUS_ERROR):
            value = as_int(report[key])
            if value is None:
                raise ValueError(f"Config check report is missing the {key!r} count")
            totals[key] += value

    return {
        "status": ToolStatus.OK if totals[STATUS_ERROR] == 0 else ToolStatus.ERROR,
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
        return {"status": ToolStatus.ERROR, "message": str(exc)}

    entries = []
    for key, value in sorted(cfg.flat().items()):
        source = cfg.source_of(key)
        redacted = _redact_api_key(key, value)
        entries.append({"key": key, "value": redacted, "source": source})
    return {"status": ToolStatus.OK, "count": len(entries), "entries": entries}


def config_path(
    wiki_dir: str | None = None,
    global_: bool = False,
) -> dict[str, Any]:
    """Show the config file path (project or global)."""
    try:
        path = get_global_config_file(create=False) if global_ else get_project_config_file(start_dir=Path(wiki_dir) if wiki_dir else None)
    except Exception as exc:
        return {"status": ToolStatus.ERROR, "message": str(exc)}
    return {"status": ToolStatus.OK, "path": str(path)}


def gitignore_wiki(
    wiki_dir: str | None = None,
    check: bool = False,
) -> dict[str, Any]:
    """Write or check the managed `.gitignore` block of a kompend vault.

    Mirrors ``kompend gitignore``: the managed block is re-rendered from the
    artifact names, every rule the user wrote around it survives, and
    ``check=True`` reports drift without touching the file.
    """
    overrides: dict[str, Any] = {}
    if wiki_dir:
        overrides[PATHS_WIKI_DIR] = str(wiki_dir)
    try:
        cfg = resolve_config(cli_overrides=overrides)
        base_dir = Path(as_str(cfg.get(PATHS_WIKI_DIR)) or str(Path.cwd()))
        rendered = render_gitignore(
            sources_dir=dir_name(cfg.get(PATHS_SOURCES_DIR), fallback=DEFAULT_SOURCES_DIR_NAME),
            output_dir=dir_name(cfg.get(PATHS_OUTPUT_DIR), fallback=DEFAULT_OUTPUT_DIR_NAME),
        )
        path = base_dir / GITIGNORE_FILE
        existing = path.read_text(encoding="utf-8") if path.is_file() else None
        if check:
            current = existing is not None and is_current(existing, rendered)
            return {"status": ToolStatus.OK if current else ToolStatus.ERROR, "path": str(path), "current": current}
        text = rendered if existing is None else merge_gitignore(existing, rendered)
        if text == existing:
            return {"status": ToolStatus.OK, "path": str(path), "changed": False}
        path.write_text(text, encoding="utf-8")
        return {"status": ToolStatus.OK, "path": str(path), "changed": True, "created": existing is None}
    except Exception as exc:
        return {"status": ToolStatus.ERROR, "message": str(exc)}


def _redact_api_key(key: str, value: Any) -> Any:
    """Hide provider API keys in output; env-var refs (${VAR}) stay visible."""
    if key.rsplit(".", maxsplit=1)[-1] == "api_key" and isinstance(value, str) and not _ENV_REF_RE.match(value):
        return "***"
    return value
