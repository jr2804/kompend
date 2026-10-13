"""Abstract base for KOMPEND output writers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from kompend.core.models import CompiledWiki


class BaseOutputWriter(ABC):
    """Output writer interface for KOMPEND Tier 5.

    Subclasses must set ``name`` to a unique identifier used for
    registry lookup (e.g. ``"markdown"``, ``"json"``, ``"typst"``).

    Use :meth:`register` / :meth:`get` for the plugin registry.
    """

    name: ClassVar[str]
    _registry: ClassVar[dict[str, type[BaseOutputWriter]]] = {}

    def __init__(self, flavor: str | None = None, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        flavor:
            Markdown flavor (``"obsidian"``, ``"gfm"``, etc.).
            Subclasses that produce Markdown use this via
            :class:`DocumentRenderer`. Others ignore it.
        """
        self.flavor = flavor

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "name", None)
        if name is not None and not getattr(cls, "__abstractmethods__", None):
            cls._registry[name] = cls

    @abstractmethod
    def write(self, wiki: CompiledWiki, output_dir: Path) -> Path:
        """Write compiled wiki to the output directory and return the output path."""
        ...

    @classmethod
    def get(cls, name: str) -> type[BaseOutputWriter] | None:
        """Look up a registered writer by name."""
        return cls._registry.get(name)

    @classmethod
    def available_formats(cls) -> list[str]:
        """Return sorted list of registered format names."""
        return sorted(cls._registry)
