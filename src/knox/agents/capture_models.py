"""Pydantic output models for capture and distill agents."""

from __future__ import annotations

from pydantic import BaseModel


class CapturedItem(BaseModel):
    """A single knowledge item extracted from text."""

    kind: str = "note"
    title: str = ""
    content: str = ""
    tags: list[str] = []


class CaptureOutput(BaseModel):
    """List of extracted knowledge items.

    When the input contains nothing extractable (greetings, chit-chat,
    gibberish, or a single trivial fragment), the agent sets ``refused``
    to ``True`` and explains why in ``refusal_reason`` instead of emitting
    empty or hallucinated items.
    """

    items: list[CapturedItem] = []
    refused: bool = False
    refusal_reason: str = ""


class DistillOutput(BaseModel):
    """A synthesized wiki page from grouped notes."""

    title: str = ""
    content: str = ""
    tags: list[str] = []
