from __future__ import annotations

import logging
from pathlib import Path

from knox.cli._sources import download_urls, ensure_in_sources, resolve_files

logger = logging.getLogger(__name__)


def ingest(
    target: str,
    *,
    sources_dir: Path,
    in_place: bool = False,
) -> Path | None:
    """Ingest a file or URL into the sources directory.

    When *in_place* is ``True``, local files are kept at their original
    location instead of being copied into *sources_dir*.
    """
    if not in_place:
        sources_dir.mkdir(parents=True, exist_ok=True)

    if _looks_like_url(target):
        downloaded = download_urls([target], sources_dir)
        if downloaded:
            logger.info("Ingested URL -> %s", downloaded[0])
            return downloaded[0]
        logger.warning("Failed to download URL: %s", target)
        return None

    resolved = resolve_files(target)
    if not resolved:
        logger.warning("No files found for: %s", target)
        return None

    copied = ensure_in_sources(resolved, sources_dir, in_place=in_place)
    if copied:
        logger.info("Ingested file -> %s", copied[0])
        return copied[0]
    return None


def _looks_like_url(target: str) -> bool:
    return target.startswith(("http://", "https://", "ftp://"))
