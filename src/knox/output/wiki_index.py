"""Auto-generated index, log, and overview pages for the compiled wiki.

Re-export shim — actual implementations live in :mod:`knox.output.pages`.

KNOX generates three complementary pages after every compile:

- ``index.md`` — Table of contents + wiki stats (:mod:`pages.index`)
- ``log.md`` — Append-only activity journal (:mod:`pages.log`)
- ``overview.md`` — LLM-generated thematic summary (:mod:`pages.overview`)
"""

from knox.output.pages.index import generate_index, write_index
from knox.output.pages.log import (
    generate_log_entry,
    generate_stage_log_entry,
    prepend_log,
)
from knox.output.pages.overview import generate_overview, write_overview

__all__ = [
    "generate_index",
    "generate_log_entry",
    "generate_overview",
    "generate_stage_log_entry",
    "prepend_log",
    "write_index",
    "write_overview",
]
