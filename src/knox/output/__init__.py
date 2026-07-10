"""KNOX output tier.

Importing this package registers all built-in output writers
with the :class:`BaseOutputWriter` registry.
"""

from __future__ import annotations

from knox.output.json_writer import JsonWriter
from knox.output.markdown import MarkdownWriter
from knox.output.okf_writer import OkfWriter
from knox.output.typst import TypstWriter

__all__ = ["JsonWriter", "MarkdownWriter", "OkfWriter", "TypstWriter"]
