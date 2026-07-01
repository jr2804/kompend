"""Test hooks module placeholder — the event-based hook system was removed.

The module is now a documentation placeholder. Future hook systems
should add tests here.
"""

from __future__ import annotations

import importlib


def test_hooks_module_importable() -> None:
    importlib.import_module("knox.core.hooks")
