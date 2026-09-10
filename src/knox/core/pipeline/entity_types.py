"""Per-vault entity-type registry.

Stable per-concept entity kinds, persisted across recompiles so the
folder assignment under ``entities/<type>/<id>.md`` does not flicker
when the model gives a slightly different ``type`` value for the same
concept on different runs.

File shape::

    {
      "polqa": "standard",
      "itu-t": "organization",
      "john-doe": "person",
      ...
    }

Keys are :func:`knox.compilation.concept_keys.normalize_concept_key`
of the concept name; values are sanitized folder names (see
:func:`knox.compilation.flows.generation._sanitize_entity_type`).

Persistence semantics: on first read, the file is loaded once. On
write, the entire dict is rewritten atomically (tmp-file + rename)
so a crash mid-write does not corrupt the registry.

See p863 wiki roundup (2026-09-10) for the motivating bug — the
model returned different ``type`` values for POLQA across runs,
causing its page to land in ``entities/standard/`` one time and
``entities/polqa/`` the next.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from knox.logging import get_logger

logger = get_logger(__name__)


class EntityTypeRegistry:
    """JSON-backed registry mapping concept keys to sanitized entity kinds.

    Parameters
    ----------
    entity_types_file:
        File to read/write. Created lazily on first save.
    """

    def __init__(self, entity_types_file: Path) -> None:
        self._file = entity_types_file
        self._data: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        """Read the registry file into memory.

        A missing file means a fresh start (empty registry). A corrupt
        file is logged and treated as empty rather than crashing — the
        next compile will rebuild it from the model's output.
        """
        if not self._file.exists():
            self._data = {}
            return
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Entity-type registry at %s is unreadable (%s); treating as empty",
                self._file,
                exc,
            )
            self._data = {}
            return
        if not isinstance(raw, dict):
            logger.warning(
                "Entity-type registry at %s has unexpected shape (%s); treating as empty",
                self._file,
                type(raw).__name__,
            )
            self._data = {}
            return
        # Coerce values to str (defensive against bad writes).
        self._data = {str(k): str(v) for k, v in raw.items() if isinstance(v, (str, int))}

    def save(self) -> None:
        """Persist the in-memory registry to disk atomically.

        Uses ``tempfile`` + ``os.replace`` so a crash mid-write doesn't
        leave a partially-written file.
        """
        self._file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=self._file.parent, prefix=".entity_types.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_path, self._file)
        except Exception:
            # Clean up the tmp file on failure.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get(self, concept_key: str) -> str | None:
        """Return the registered type for *concept_key*, or None."""
        return self._data.get(concept_key)

    def record(self, concept_key: str, entity_type: str) -> bool:
        """Record *entity_type* for *concept_key*.

        Returns True when the type changed (caller may want to log
        regressions), False when it was already set to the same value
        or unset.

        Empty/None keys or types are ignored — the model occasionally
        returns blank ``type`` fields and those should not pollute the
        registry.
        """
        if not concept_key or not entity_type:
            return False
        existing = self._data.get(concept_key)
        if existing == entity_type:
            return False
        self._data[concept_key] = entity_type
        return existing is not None  # True = override

    def all(self) -> dict[str, str]:
        """Snapshot of the registry (copy)."""
        return dict(self._data)

    def __len__(self) -> int:
        return len(self._data)
