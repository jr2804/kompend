"""Ingest, downloader, organize, and ignore models."""

from __future__ import annotations

import os
from typing import Final

from pydantic import Field

from kompend.core.config.models._base import _KompendBaseModel
from kompend.core.keys import EnvVar

_DEFAULT_DOCLING_THREADS: Final[int] = min(4, os.cpu_count() or 4)


class IngestConfig(_KompendBaseModel):
    parse_timeout_s: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=EnvVar.INGEST_PARSE_TIMEOUT_S,
        description=(
            "Wall-clock budget for parsing a single source, in seconds. `0` (default) parses in-process "
            "with no bound, exactly as before. Any positive value runs each parse in a child process that "
            "is killed when the budget expires, so one wedged backend cannot stall an ingest: the file is "
            "counted as failed, the run continues, and the next run retries it. Each bounded parse pays "
            "the child's interpreter start-up, so prefer the in-process path for corpora you trust."
        ),
    )
    parsecraft_docling_threads: int | None = Field(
        default=_DEFAULT_DOCLING_THREADS,
        ge=1,
        validation_alias=EnvVar.INGEST_PARSECRAFT_DOCLING_THREADS,
        description=(
            "ONNX thread count of the docling engine (`DOCLING_RS_PDF_THREADS`), applied at parser "
            "construction whether or not the run routes anything to docling. Defaults to 4 (or fewer on a "
            "smaller host). It is a bound, not a target: the engine still spreads work across its own "
            "pool, and an explicit `DOCLING_RS_PDF_THREADS` in the environment always wins. Measured on "
            "Windows with docling-rs 2.65.1 (2026-11-04), one 8-page PDF: engine default 7.1 s / "
            "2790 MB peak commit, 4 threads 7.4 s / 1830 MB, 1 thread 16.3 s / 1533 MB — 4 is the knee. "
            "The thread count changes neither the extracted text nor the routing, so it is not part of "
            "the parse-cache fingerprint."
        ),
    )
    parse_min_free_commit_mb: int = Field(
        default=10240,
        ge=0,
        validation_alias=EnvVar.INGEST_PARSE_MIN_FREE_COMMIT_MB,
        description=(
            "Free commit charge (MB) a parse that may load the heavy conversion engine requires. The "
            "default (10240) sits above the measured peak of one docling conversion (about 8.0 GB) so the "
            "run stops with an actionable message instead of the OS aborting an allocation inside the "
            "native extension. `0` disables the check; the concurrency cap stays. Text sources never take "
            "this budget — only the formats that may load the PDF engine do."
        ),
    )
    max_concurrent_heavy_parses: int = Field(
        default=1,
        ge=1,
        validation_alias=EnvVar.INGEST_MAX_CONCURRENT_HEAVY_PARSES,
        description=(
            "How many parses that may load the heavy conversion engine may run at once. `1` (default) "
            "serializes them: the engine's cost is a fixed several gigabytes per conversion, so N workers "
            "cost N times the host's commit headroom. Light sources are unaffected and keep parsing in "
            "parallel."
        ),
    )


class DownloaderConfig(_KompendBaseModel):
    undetected: bool = Field(default=True, validation_alias=EnvVar.DOWNLOADER_UNDETECTED)


class OrganizeConfig(_KompendBaseModel):
    """Configuration for ``kompend organize`` LLM-based file categorization."""

    max_depth: int = Field(default=3, validation_alias=EnvVar.ORGANIZE_MAX_DEPTH)
    min_subcategories: int = Field(default=2, validation_alias=EnvVar.ORGANIZE_MIN_SUBCATEGORIES)
    min_files_per_category: int = 2
    collapse_single_file: bool = Field(default=True, validation_alias=EnvVar.ORGANIZE_COLLAPSE_SINGLE_FILE)


class IgnoreConfig(_KompendBaseModel):
    patterns: list[str] = Field(default_factory=list)
