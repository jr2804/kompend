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
    """List of extracted knowledge items."""

    items: list[CapturedItem] = []


class DistillOutput(BaseModel):
    """A synthesized wiki page from grouped notes."""

    title: str = ""
    content: str = ""
    tags: list[str] = []
