"""rumdl config resolution for a vault.

`compile.output.rumdl_config` points at the per-vault rumdl config; when
unset, KNOX defaults to ``<base_dir>/.knox/rumdl.toml`` and materializes it
from the shipped default on first use so users can edit it.
"""

from __future__ import annotations

import importlib.resources as ir
from pathlib import Path

from knox.core.paths import RUMDL_CONFIG_FILE
from knox.logging import get_logger

logger = get_logger(__name__)


def resolve_rumdl_config(
    base_dir: Path,
    configured: Path | str | None = None,
) -> Path | None:
    """Resolve the rumdl config path for a vault rooted at *base_dir*.

    - Explicit *configured* wins (relative paths resolve against *base_dir*).
    - Otherwise default to ``<base_dir>/.knox/rumdl.toml``, materializing it
      from the shipped default on first use.

    Returns ``None`` only when the shipped default cannot be located.
    """
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = base_dir / path
        return path

    target = base_dir / RUMDL_CONFIG_FILE
    if target.exists():
        return target

    try:
        data = (ir.files("knox.lint") / "rumdl.toml").read_bytes()
    except (ModuleNotFoundError, FileNotFoundError, AttributeError):
        logger.warning("Shipped rumdl default not found — rumdl config disabled")
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    logger.info("Wrote default rumdl config to %s (edit to override)", target)
    return target
