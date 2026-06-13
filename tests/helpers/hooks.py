"""Test helper callables for hook import tests."""

from __future__ import annotations

from knox.core.hooks import HookContext


def query_logger(ctx: HookContext) -> None:
    """Example hook callable for testing config-driven hook loading."""
