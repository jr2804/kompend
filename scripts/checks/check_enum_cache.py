#!/usr/bin/env python
"""Every StrEnum reachable from a cached dataclass must be registered.

The cache codec tags an enum by class name and refuses an unknown tag;
``CacheStore.get`` treats that as a miss. A missing registration therefore
throws away work that was already paid for - an unregistered ``EdgeType`` cost
a 65-minute recompile before this check existed.
"""

from __future__ import annotations

import dataclasses
import sys
import typing
from enum import StrEnum

from kompend.core.cache import store


def main() -> int:
    reachable = _reachable_enums()
    registered = set(store._CACHE_ENUMS)
    missing = sorted(set(reachable) - registered)

    print(f"check-enum-cache: {len(reachable)} enum(s) reachable, {len(registered)} registered")
    for name, where in sorted(reachable.items()):
        mark = "ok" if name in registered else "MISSING"
        print(f"  {mark:7} {name:16} ({where})")

    if missing:
        print(
            f"\nERROR: {len(missing)} StrEnum(s) travel inside a cached payload but are not in "
            f"_CACHE_ENUMS: {', '.join(missing)}\n"
            "Add them to src/kompend/core/cache/store.py::_CACHE_ENUMS, or the entry becomes unreadable.",
            file=sys.stderr,
        )
        return 1
    return 0


def _reachable_enums() -> dict[str, str]:
    """Map every enum name reachable from a cached dataclass to where it sits."""
    found: dict[str, str] = {}
    pending = list(store._CACHE_DATACLASSES.values())
    seen: set[type] = set()
    while pending:
        cls = pending.pop()
        if cls in seen:
            continue
        seen.add(cls)
        try:
            hints = typing.get_type_hints(cls)
        except Exception as exc:  # noqa: BLE001 - a broken hint is a finding, not a crash
            print(f"ERROR: cannot resolve annotations of {cls.__name__}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        for field_name, hint in hints.items():
            for item in _walk_annotation(hint):
                if isinstance(item, type) and issubclass(item, StrEnum) and item is not StrEnum:
                    found.setdefault(item.__name__, f"{cls.__name__}.{field_name}")
                elif isinstance(item, type) and (dataclasses.is_dataclass(item) or typing.is_typeddict(item)):
                    pending.append(item)
    return found


def _walk_annotation(hint: object) -> typing.Iterator[object]:
    """Yield *hint* and everything nested inside it."""
    yield hint
    for arg in typing.get_args(hint):
        if arg is type(None):
            continue
        yield from _walk_annotation(arg)


if __name__ == "__main__":
    sys.exit(main())
