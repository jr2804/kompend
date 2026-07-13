"""Pydantic output model for the categorize agent."""

from __future__ import annotations

from pydantic import BaseModel


class CategoryNode(BaseModel):
    """A single node in the proposed category tree."""

    name: str = ""
    files: list[str] = []
    subcategories: list[CategoryNode] = []


class CategoryTree(BaseModel):
    """Root container returned by the LLM."""

    categories: list[CategoryNode] = []
    unfiled: list[str] = []
