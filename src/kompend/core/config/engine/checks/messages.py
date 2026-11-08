"""Status vocabulary and report rows for ``kompend config check``."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from kompend.types import JSONDict, JSONList, JSONValue

# ── Status kind ─────────────────────────────────────────────────────────────

STATUS_OK = "ok"
STATUS_MODIFIED = "modified"
STATUS_ERROR = "error"
STATUS_REMOVED = "removed"  # key present in TOML but removed from schema
STATUS_DEPRECATED = "deprecated"  # renamed key, still migrated at resolve time


@dataclasses.dataclass(frozen=True)
class ConfigCheckEntry:
    """One row in a config-check report.

    `current` is the user's value when the key is set in the TOML, or
    ``None`` when the user did not set it and we are reporting the
    default-resolved value.
    """

    key: str
    status: str  # one of STATUS_OK / STATUS_MODIFIED / STATUS_ERROR
    current: JSONValue  # None when not set in user TOML
    default: JSONValue  # value in DEFAULTS (None for keys unknown to schema)
    set_in_toml: bool = False  # False → "not set; default resolves here"
    message: str | None = None  # populated only for error rows

    def to_dict(self) -> JSONDict:
        out: JSONDict = {
            "key": self.key,
            "status": self.status,
            "default": self.default,
        }
        if self.set_in_toml or self.status != STATUS_OK:
            out["current"] = self.current
        if self.message is not None:
            out["message"] = self.message
        return out


@dataclasses.dataclass(frozen=True)
class ConfigCheckReport:
    """Full result for one config file (or merged set)."""

    path: Path | None  # None when the report covers multiple files merged
    entries: tuple[ConfigCheckEntry, ...]
    files_scanned: tuple[Path, ...] = ()

    @property
    def ok_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_OK)

    @property
    def modified_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_MODIFIED)

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_ERROR)

    @property
    def removed_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_REMOVED)

    @property
    def deprecated_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_DEPRECATED)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def to_dict(self) -> JSONDict:
        entries: JSONList = [e.to_dict() for e in self.entries]
        out: JSONDict = {
            "ok": self.ok_count,
            "modified": self.modified_count,
            "error": self.error_count,
            "removed": self.removed_count,
            "entries": entries,
        }
        if self.path is not None:
            out["path"] = str(self.path)
        if self.files_scanned:
            out["files"] = [str(p) for p in self.files_scanned]
        return out
