"""Event-driven automation hooks for KNOX pipeline operations.

Hooks let users register callbacks that fire on key pipeline events.
Hooks are loaded from the ``[hooks]`` config section as ``module:attribute``
import paths.  A global dict of event→callbacks is kept at module level.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from knox.logging import get_logger

logger = get_logger(__name__)


class HookEvent(StrEnum):
    """Pipeline events that hooks can observe."""

    PRE_COMPILE = "pre_compile"
    POST_COMPILE = "post_compile"
    ON_CAPTURE = "on_capture"
    ON_QUERY = "on_query"
    ON_LINT = "on_lint"


HookCallback = Callable[..., None]

# Module-level registry: event → list of callables
_hooks: dict[HookEvent, list[HookCallback]] = {}


def register_hook(event: HookEvent, callback: HookCallback) -> None:
    """Register a callback for an event."""
    _hooks.setdefault(event, []).append(callback)


def fire_hook(event: HookEvent, **data: Any) -> None:
    """Fire all callbacks registered for *event*.

    Exceptions in individual callbacks are logged and swallowed so a
    faulty hook never breaks the pipeline.
    """
    for callback in _hooks.get(event, []):
        try:
            callback(**data)
        except Exception:
            logger.exception("Hook %s for %s raised", callback, event)


def clear_hooks() -> None:
    """Remove all registered hooks."""
    _hooks.clear()


def get_registry() -> dict[HookEvent, list[HookCallback]]:
    """Return the default global registry (for testing/introspection)."""
    return _hooks


def load_hooks_from_config(config: dict[str, Any]) -> int:
    """Load hooks from a resolved config dict.

    The ``[hooks]`` section maps event names to lists of callable paths::

        [hooks]
        post_compile = ["mypkg.hooks:auto_lint"]

    Each callable uses ``module:attribute`` syntax.

    Returns the number of hooks successfully loaded.
    """
    raw_hooks = config.get("hooks")
    if not isinstance(raw_hooks, dict):
        return 0

    loaded = 0
    for event_str, paths in raw_hooks.items():
        try:
            event = HookEvent(event_str)
        except ValueError:
            logger.warning("Unknown hook event '%s' — skipping", event_str, exc_info=True)
            continue

        if isinstance(paths, str):
            path_list = [paths]
        elif isinstance(paths, list):
            path_list = paths
        else:
            logger.warning("Hook event '%s' must be a list of paths", event_str)
            continue

        for dotted_name in path_list:
            callback = _import_callable(str(dotted_name))
            if callback is not None:
                register_hook(event, callback)
                loaded += 1

    return loaded


def _import_callable(dotted_name: str) -> HookCallback | None:
    """Import a callable from a ``module:attribute`` string."""
    if ":" not in dotted_name:
        logger.warning(
            "Hook dotted_name '%s' must use 'module:attribute' syntax — skipping",
            dotted_name,
        )
        return None

    module_path, _, attr = dotted_name.rpartition(":")
    if not module_path or not attr:
        logger.warning("Invalid hook dotted_name '%s' — skipping", dotted_name)
        return None

    try:
        module = importlib.import_module(module_path)
        callback = getattr(module, attr)
    except (ImportError, AttributeError):
        logger.exception("Failed to import hook '%s'", dotted_name)
        return None

    if not callable(callback):
        logger.warning("Hook dotted_name '%s' does not resolve to a callable — skipping", dotted_name)
        return None

    return callback
