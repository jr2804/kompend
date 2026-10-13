"""Auto-generated index, log, and overview pages for the compiled wiki.

Re-export shim — actual implementations live in :mod:`kompend.output.pages`.

kompend generates three complementary pages after every compile:

- ``index.md`` — Table of contents + wiki stats (:mod:`pages.index`)
- ``log.md`` — Append-only activity journal (:mod:`pages.log`)
- ``overview.md`` — LLM-generated thematic summary (:mod:`pages.overview`)
"""

from kompend.output.pages.index import generate_index, write_index
from kompend.output.pages.log import LogStage, append_log_entry, generate_log_entry, log_stage
from kompend.output.pages.overview import generate_overview, write_overview

__all__ = [
    "LogStage",
    "append_log_entry",
    "generate_index",
    "generate_log_entry",
    "generate_overview",
    "log_stage",
    "write_index",
    "write_overview",
]
