"""rumdl config resolution for a vault.

`compile.output.rumdl_config` points at the per-vault rumdl config; when
unset, kompend defaults to ``<base_dir>/.kompend/rumdl.toml`` and materializes it
from the shipped default on first use so users can edit it.

When the on-disk copy is stale (its content hash no longer matches the
shipped default), the file is refreshed and the user is warned. Custom
user edits are detected by an ``# kompend-managed: <hash>`` comment that
``resolve_rumdl_config`` writes on first materialization. Edits below that
marker survive the refresh; everything above is overwritten.
"""

from __future__ import annotations

import hashlib
import importlib.resources as ir
import re
from pathlib import Path

from kompend.core.paths import RUMDL_CONFIG_FILE
from kompend.logging import get_logger

#: Marker comment written into materialized configs so we can detect user
#: edits below it. Format: ``# kompend-managed: <sha256-prefix>``.
_MANAGED_MARKER_RE = re.compile(r"^# kompend-managed:\s*([0-9a-f]+)\s*$", re.MULTILINE)

logger = get_logger(__name__)


def _shipped_hash() -> str:
    """SHA-256 prefix of the shipped default config."""
    data = _shipped_default_bytes() or b""
    return hashlib.sha256(data).hexdigest()[:16]


def resolve_rumdl_config(
    base_dir: Path,
    configured: Path | str | None = None,
) -> Path | None:
    """Resolve the rumdl config path for a vault rooted at *base_dir*.

    - Explicit *configured* wins (relative paths resolve against *base_dir*).
    - Otherwise default to ``<base_dir>/.kompend/rumdl.toml``, materializing it
      from the shipped default on first use.

    When the on-disk copy was materialized by an OLDER kompend version, the
    shipped default is re-written on top. User edits are preserved if the
    user kept them below the ``# kompend-managed: <hash>`` marker; otherwise
    the user is warned that their edits have been overwritten.

    Returns ``None`` only when the shipped default cannot be located.
    """
    if configured:
        configured_path = Path(configured)
        if not configured_path.is_absolute():
            configured_path = base_dir / configured_path
        return configured_path

    target = base_dir / RUMDL_CONFIG_FILE
    shipped = _shipped_default_bytes()
    if shipped is None:
        logger.warning("Shipped rumdl default not found — rumdl config disabled")
        return None
    current_hash = hashlib.sha256(shipped).hexdigest()[:16]

    if target.exists():
        existing = target.read_bytes()
        marker = _MANAGED_MARKER_RE.search(existing.decode("utf-8", errors="replace"))
        if marker and marker.group(1) == current_hash:
            return target  # up to date
        # Stale or never marked — refresh.
        user_section = _extract_user_section(existing)
        new_content = _build_managed_config(shipped, user_section, current_hash)
        target.write_bytes(new_content)
        if user_section is not None:
            logger.warning(
                "Refreshed stale rumdl config at %s (shipped default updated); user-edited section preserved below the kompend-managed marker",
                target,
            )
        else:
            logger.info("Refreshed stale rumdl config at %s", target)
        return target

    # First-use materialization.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_build_managed_config(shipped, None, current_hash))
    logger.info("Wrote default rumdl config to %s (edit below the marker to override)", target)
    return target


def _shipped_default_bytes() -> bytes | None:
    """Load the bundled ``src/kompend/lint/rumdl.toml`` (read-only)."""
    try:
        return (ir.files("kompend.lint") / "rumdl.toml").read_bytes()
    except (ModuleNotFoundError, FileNotFoundError, AttributeError):
        return None


def _build_managed_config(shipped: bytes, user_section: str | None, current_hash: str) -> bytes:
    """Append the ``# kompend-managed`` marker and (optional) user section to the shipped default."""
    suffix = f"\n\n# kompend-managed: {current_hash}\n"
    if user_section is not None:
        suffix += f"# User edits below — preserved across refreshes.\n{user_section}\n"
    return shipped.rstrip(b"\n") + suffix.encode("utf-8")


def _extract_user_section(content: bytes) -> str | None:
    """Return text below the managed marker, or None if the file is untracked/old-format."""
    text = content.decode("utf-8", errors="replace")
    marker = _MANAGED_MARKER_RE.search(text)
    if marker is None:
        return None  # Old-format file — no marker, can't preserve (warn caller).
    return text[marker.end() :].strip() or None
