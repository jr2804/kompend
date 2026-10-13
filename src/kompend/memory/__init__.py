"""KOMPEND Memory layer.

Two concerns live here:

- **Query memory** (``store``): episodic storage of query/answer/feedback
  triples for semantic retrieval and long-term agent learning.
- **Compile memory** (``lifecycle``): confidence scoring, tier promotion,
  supersession, and retention/decay for wiki nodes across compiles.
"""

from kompend.memory.lifecycle import update_memory_for_compile
from kompend.memory.store import MemoryField, MemoryStore

__all__ = ["MemoryField", "MemoryStore", "update_memory_for_compile"]
