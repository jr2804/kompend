"""KOMPEND output tier.

Importing this package registers all built-in output writers
with the :class:`BaseOutputWriter` registry.
"""

from __future__ import annotations

from kompend.output.json_writer import JsonWriter
from kompend.output.markdown import MarkdownWriter
from kompend.output.typst import TypstWriter

__all__ = ["JsonWriter", "MarkdownWriter", "TypstWriter"]
