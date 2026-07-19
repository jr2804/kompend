"""Page generation submodules for compiled wiki output.

- :mod:`index` — ``index.md`` TOC + stats
- :mod:`log` — ``log.md`` activity journal entries
- :mod:`overview` — ``overview.md`` LLM-generated summary
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
