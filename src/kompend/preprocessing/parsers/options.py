"""Construction inputs for kompend's one parser.

:class:`ParserOptions` is the bridge between resolved kompend config and the
parsecraft adapter's constructor. It exists so the pipeline, the CLI and the MCP
server build the parser identically: they share one parse cache, and a site that
built a narrower option set could serve a document produced under different
settings — ``kompend ingest`` once passed only ``images_dir`` while ``kompend
compile`` also passed the config-derived ``image_mode``.

The routing vocabulary (preference, OCR permission, engine list, backend pins,
per-backend options, and which judge model to ask) is NOT here: it belongs to
parsecraft, and its files are the ones that hold it (see
:mod:`kompend.preprocessing.parsers.parsecraft_routing` for the precedence
order). Only what is genuinely kompend's remains below — how much of the host a
docling pass may use, and where figures land.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from kompend.core.config import ImageMode, ResolvedConfig
from kompend.core.paths import PARSECRAFT_PROJECT_FILE, VaultPaths
from kompend.preprocessing.parsers.base import IMAGE_SCALE_DEFAULT
from kompend.preprocessing.parsers.parsecraft_routing import ParsecraftRoutingConfig

#: The parser's name in a parsed document's metadata and in ``list-parsers``.
#: One backend, so this is a constant, not a vocabulary: nothing selects between
#: parsers any more.
PARSER_PARSECRAFT: Final[str] = "parsecraft"


@dataclass(frozen=True)
class ParserOptions:
    """Everything the parser needs that does not come from parsecraft's config."""

    images_dir: Path | None = None
    image_mode: ImageMode = ImageMode.FILE
    images_scale: float = IMAGE_SCALE_DEFAULT
    #: parsecraft's own config files for this run, lowest priority first: the
    #: vault's ``.kompend/parsecraft.toml``, then ``./parsecraft.toml``.
    #: parsecraft's engine layers its global file below them and the
    #: ``PARSECRAFT_CONFIG_*`` environment above them.
    parsecraft_config_files: tuple[Path, ...] = ()
    #: An already-loaded parsecraft config. ``None`` (the normal path) loads it
    #: from ``parsecraft_config_files`` on the adapter's first use.
    parsecraft_config: ParsecraftRoutingConfig | None = None
    #: ONNX thread count of the docling engine. `None` keeps the engine's
    #: own thread count; a value bounds the engine's memory use (the ONNX
    #: arena scales with the thread count) at the price of wall-clock time.
    parsecraft_docling_threads: int | None = None
    #: Commit charge the planned engines must leave free
    #: (``ingest.parse_min_free_commit_mb``, in megabytes, same unit as the
    #: config key; ``0`` disables the check). The parser checks it against the
    #: routing plan, because only the plan reveals which engine a source
    #: reaches — see
    #: :func:`kompend.preprocessing.host_memory.ensure_engine_headroom`.
    parse_min_free_commit_mb: int = 0

    @classmethod
    def from_config(cls, cfg: ResolvedConfig, *, base_dir: Path) -> ParserOptions:
        """Build the parser options implied by a resolved kompend config.

        ``base_dir`` is the vault root: the image cache and the
        ``parsecraft.toml`` location both derive from it, and deriving them
        here keeps every caller from re-computing the vault layout.
        ``images_scale`` is not a config key, so it takes the field default.
        """
        kompend = cfg.kompend
        paths = VaultPaths(base_dir)
        return cls(
            images_dir=paths.images_dir,
            image_mode=kompend.compile.image_mode,
            parsecraft_config_files=(paths.parsecraft_config_file, Path.cwd() / PARSECRAFT_PROJECT_FILE),
            parsecraft_docling_threads=kompend.ingest.parsecraft_docling_threads,
            parse_min_free_commit_mb=kompend.ingest.parse_min_free_commit_mb,
        )
