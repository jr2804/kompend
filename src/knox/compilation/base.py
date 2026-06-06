"""Abstract base for KNOX compilers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from knox.core.models import CompiledWiki, ResolvedChunk


class BaseCompiler(ABC):
    """Compiler interface for KNOX Tier 4."""

    @abstractmethod
    def compile(self, chunks: list[ResolvedChunk]) -> CompiledWiki:
        """Compile resolved chunks into a knowledge graph wiki."""
        ...
