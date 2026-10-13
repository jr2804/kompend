"""Pack management MCP tools."""

from typing import Any

from kompend.logging import get_logger
from kompend.packs.base import get_registry
from kompend.packs.builtins import academic_pack, technical_pack

_PACK_MAP = {
    "academic": academic_pack,
    "technical": technical_pack,
}

logger = get_logger(__name__)


def pack_list() -> dict[str, Any]:
    """List installed contribution packs."""
    registry = get_registry()
    if not registry.names:
        return {"status": "ok", "packs": [], "count": 0, "message": "No packs registered."}
    return {
        "status": "ok",
        "count": len(registry.names),
        "packs": [{"name": pack.name, "version": pack.version, "description": pack.description} for pack in registry.all],  # nocheck: MCP response payload
    }


def pack_install(name: str) -> dict[str, Any]:
    """Install a built-in contribution pack (academic | technical)."""
    pack = _PACK_MAP.get(name.lower())
    if pack is None:
        available = ", ".join(sorted(_PACK_MAP))
        return {"status": "error", "message": f"Unknown pack '{name}'. Available: {available}"}
    registry = get_registry()
    registry.register(pack)
    return {"status": "ok", "message": f"Installed pack '{pack.name}' v{pack.version}"}
