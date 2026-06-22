"""Event-driven automation hooks for KNOX pipeline operations.

Hooks let users register callbacks that fire on key pipeline events
(pre/post compile, capture, query, lint).  Hooks are loaded from the
``[hooks]`` config section as ``module:attribute`` import paths.

Example ``knox.toml``::

    [hooks]
    post_compile = ["mypkg.hooks:auto_lint"]
    on_capture = ["mypkg.hooks:notify"]
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class HookEvent(StrEnum):
    """Pipeline events that hooks can observe."""

    PRE_COMPILE = "pre_compile"
    POST_COMPILE = "post_compile"
    ON_CAPTURE = "on_capture"
    ON_QUERY = "on_query"
    ON_LINT = "on_lint"


@dataclass
class HookContext:
    """Context passed to every hook callback."""

    event: HookEvent
    data: dict[str, Any] = field(default_factory=dict)


HookCallback = Callable[[HookContext], None]


class HookRegistry:
    """Registry of event-to-callback mappings."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookCallback]] = {}

    def register(self, event: HookEvent, callback: HookCallback) -> None:
        """Register a callback for an event."""
        self._hooks.setdefault(event, []).append(callback)

    def clear(self) -> None:
        """Remove all registered hooks."""
        self._hooks.clear()

    def fire(self, event: HookEvent, **data: Any) -> None:
        """Fire all callbacks registered for *event*.

        Exceptions in individual callbacks are logged and swallowed so a
        faulty hook never breaks the pipeline.
        """
        ctx = HookContext(event=event, data=data)
        for callback in self._hooks.get(event, []):
            try:
                callback(ctx)
            except Exception:
                logger.exception("Hook %s for %s raised", callback, event)


# ── Module-level convenience API ────────────────────────────────────────

_default_registry = HookRegistry()


def register_hook(event: HookEvent, callback: HookCallback) -> None:
    """Register a hook on the default global registry."""
    _default_registry.register(event, callback)


def fire_hook(event: HookEvent, **data: Any) -> None:
    """Fire all hooks for *event* on the default global registry."""
    _default_registry.fire(event, **data)


def clear_hooks() -> None:
    """Clear all hooks on the default global registry."""
    _default_registry.clear()


def get_registry() -> HookRegistry:
    """Return the default global registry (for testing/introspection)."""
    return _default_registry


# ── Config-driven loading ──────────────────────────────────────────────


def load_hooks_from_config(config: dict[str, Any]) -> int:
    """Load hooks from a resolved config dict.

    The ``[hooks]`` section maps event names to lists of callable paths::

        [hooks]
        post_compile = ["mypkg.hooks:auto_lint"]

    Each callable dotted_name uses ``module:attribute`` syntax.

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
