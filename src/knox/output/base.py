"""Abstract base for KNOX output writers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from knox.core.models import CompiledWiki


class BaseOutputWriter(ABC):
    """Output writer interface for KNOX Tier 5."""

    @abstractmethod
    def write(self, wiki: CompiledWiki, output_dir: Path) -> Path:
        """Write compiled wiki to the output directory and return the output path."""
        ...
