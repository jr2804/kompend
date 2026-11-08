"""Diff a user-supplied config TOML against the built-in :data:`DEFAULTS`.

Used by the ``kompend config check`` CLI command. For each provided file:

- Walks every leaf key in the live :class:`KompendConfig` schema (so nested
  model fields like ``compile.llm.rlm.enable`` are reachable even when the
  parent defaults to ``None``).
- For every key the user actually set in the TOML, report:
    * ``ok``         — value matches the schema default
    * ``modified``   — value is set and differs from the default
    * ``error``      — value fails pydantic validation
- Reports ``ok`` for every schema key the user did *not* set (the resolved
  value equals the default).
- Adds one ``error`` row per leaf key present in the TOML but unknown to
  the schema (catches typos and removed fields).

Fields whose keys the *user* names are namespaces, not typos: ``providers``,
``compile.skills`` and ``providers.<name>.models`` hold models keyed by name,
and ``categories``/``frontmatter``/``hooks`` are free-form tables. A key under
one of those is matched segment-wise against the schema instead of compared to a
fixed list, so ``providers.ollama.base_url`` is a normal row while
``providers.ollama.base_uri`` is still an error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kompend.core.config.engine._loading import load_toml
from kompend.core.config.engine._schema import DEFAULTS, DEPRECATED_ALIASES
from kompend.core.config.engine.checks.cross_key import collect_validation_errors, overlay_merge
from kompend.core.config.engine.checks.deprecated import (
    deprecated_rows,
    removed_ancestor,
    removed_leaf_keys,
    removed_rows,
)
from kompend.core.config.engine.checks.messages import (
    STATUS_DEPRECATED,
    STATUS_ERROR,
    STATUS_MODIFIED,
    STATUS_OK,
    STATUS_REMOVED,
    ConfigCheckEntry,
    ConfigCheckReport,
)
from kompend.core.config.engine.checks.schema import (
    _DYNAMIC,
    matches_schema_pattern,
    open_namespace_leaves,
    schema_defaults,
    schema_leaf_keys,
    schema_lookup,
)
from kompend.core.utils.dicts import flat_keys, get_dot, resolve_dotted

__all__ = [
    "STATUS_DEPRECATED",
    "STATUS_ERROR",
    "STATUS_MODIFIED",
    "STATUS_OK",
    "STATUS_REMOVED",
    "ConfigCheckEntry",
    "ConfigCheckReport",
    "check_config_file",
    "check_config_files",
]


# ── Public API ──────────────────────────────────────────────────────────────


def check_config_file(path: Path) -> ConfigCheckReport:
    """Check one config file against the live :class:`KompendConfig` schema.

    A row is emitted for every leaf key known to the schema; unknown keys
    present in the file surface as additional error rows. Keys the user
    did not set are reported as ``ok`` (default).
    """
    if not path.exists():
        entries = _all_defaults_as_errors(path, "file does not exist")
        return ConfigCheckReport(path=path, entries=entries)
    raw = load_toml(path)
    return _check_loaded(raw=raw, path=path, files_scanned=(path,))


def check_config_files(paths: list[Path]) -> ConfigCheckReport:
    """Check multiple files as a merged overlay (later wins).

    Useful for ``kompend config check ./base.toml ./local.toml`` — the local
    file overrides the base, and the diff is against the resulting merged
    config.
    """
    raw: dict[str, object] = {}
    files_scanned: list[Path] = []
    for path in paths:
        if not path.exists():
            entries = _all_defaults_as_errors(path, "file does not exist")
            return ConfigCheckReport(path=None, entries=entries, files_scanned=tuple(paths))
        raw = overlay_merge(raw, load_toml(path))
        files_scanned.append(path)
    return _check_loaded(
        raw=raw,
        path=None,
        files_scanned=tuple(files_scanned),
    )


def _all_defaults_as_errors(path: Path, message: str) -> tuple[ConfigCheckEntry, ...]:
    """Return one ``error`` entry per default key when the file can't be loaded."""
    return tuple(
        ConfigCheckEntry(
            key=k,
            status=STATUS_ERROR,
            current=None,
            default=get_dot(DEFAULTS, k),
            message=f"{path}: {message}",
        )
        for k in sorted(schema_leaf_keys())
        if _DYNAMIC not in k
    )


def _check_loaded(
    *,
    raw: dict[str, Any],
    path: Path | None,
    files_scanned: tuple[Path, ...],
) -> ConfigCheckReport:
    """Diff ``raw`` against schema defaults and validate via :class:`KompendConfig`."""
    all_schema_keys = schema_leaf_keys()
    # Keys with a ``*`` segment describe a user-named namespace; they are matched
    # segment-wise, never compared for equality.
    schema_patterns = tuple(sorted(k for k in all_schema_keys if _DYNAMIC in k))
    schema_keys = {k for k in all_schema_keys if _DYNAMIC not in k}
    schema_default_values = schema_defaults()
    user_keys = set(flat_keys(raw))
    retired_leaf_keys = removed_leaf_keys()

    def _known_schema_key(key: str) -> bool:
        """True when *key* is a schema leaf, in a namespace, or under a table."""
        if key in schema_keys:
            return True
        for pattern in schema_patterns:
            if matches_schema_pattern(key, pattern):
                return True
        # A free-form table's keys are the user's data (``categories.foo``).
        return any(key.startswith(f"{leaf}.") for leaf in open_namespace_leaves())

    def _removed_ancestor(key: str) -> str | None:
        return removed_ancestor(key, retired_leaf_keys)

    unknown_keys = sorted(k for k in user_keys if not _known_schema_key(k) and _removed_ancestor(k) is None and k not in DEPRECATED_ALIASES)
    removed_keys = sorted({a for k in user_keys if k not in schema_keys and (a := _removed_ancestor(k)) is not None})
    deprecated_keys = sorted(k for k in user_keys if k in DEPRECATED_ALIASES)

    # Validation pass via pydantic to surface type/value errors.
    validation_error_keys = collect_validation_errors(raw, user_keys)

    entries: list[ConfigCheckEntry] = []
    for key in sorted(schema_keys):
        default_value = schema_default_values.get(key, get_dot(DEFAULTS, key))
        in_user_toml = key in user_keys
        user_value = get_dot(raw, key) if in_user_toml else None

        # Leaves under an optional nested model (e.g. ``compile.llm.rlm.enable``)
        # are only relevant when the user actually wrote through the
        # parent. If neither the parent nor the leaf is in the user's TOML
        # we silently skip them — we are not going to manufacture rows for
        # ``ok`` (= default = False) under a parent the user did not touch.
        if not in_user_toml and not _parent_key_present(raw, key):
            continue

        if key in validation_error_keys:
            entries.append(
                ConfigCheckEntry(
                    key=key,
                    status=STATUS_ERROR,
                    current=user_value,
                    default=default_value,
                    set_in_toml=in_user_toml,
                    message=validation_error_keys[key],
                ),
            )
            continue

        if in_user_toml and not _values_equal(user_value, default_value):
            entries.append(
                ConfigCheckEntry(
                    key=key,
                    status=STATUS_MODIFIED,
                    current=user_value,
                    default=default_value,
                    set_in_toml=True,
                ),
            )
            continue

        entries.append(
            ConfigCheckEntry(
                key=key,
                status=STATUS_OK,
                current=user_value,
                default=default_value,
                set_in_toml=in_user_toml,
            ),
        )

    # Rows for keys the user set inside a user-named namespace. They are not in
    # ``schema_keys`` (that set is pattern-free), so without this pass a valid
    # ``providers.ollama.base_url`` would silently vanish from the report.
    for key in sorted(k for k in user_keys if k not in schema_keys and _known_schema_key(k)):
        found, default_value = schema_lookup(key, schema_default_values, schema_patterns)
        if not found:
            continue
        found_value, user_value = resolve_dotted(raw, key)
        user_value = user_value if found_value else None
        status = STATUS_OK
        message = None
        if key in validation_error_keys:
            status, message = STATUS_ERROR, validation_error_keys[key]
        elif not _values_equal(user_value, default_value):
            status = STATUS_MODIFIED
        entries.append(
            ConfigCheckEntry(
                key=key,
                status=status,
                current=user_value,
                default=default_value,
                set_in_toml=True,
                message=message,
            ),
        )

    # Add error rows for unknown keys the user wrote.
    for key in unknown_keys:
        entries.append(
            ConfigCheckEntry(
                key=key,
                status=STATUS_ERROR,
                current=get_dot(raw, key),
                default=None,
                set_in_toml=True,
                message="unknown config key (not in schema)",
            ),
        )

    # Add deprecation rows for keys removed from the schema (older vaults).
    entries.extend(removed_rows(raw, set(removed_keys)))

    # Renamed keys: still migrated at resolve time, so report the migration
    # rather than an unknown-key error.
    entries.extend(deprecated_rows(raw, set(deprecated_keys)))

    return ConfigCheckReport(
        path=path,
        entries=tuple(entries),
        files_scanned=files_scanned,
    )


def _parent_key_present(raw: dict[str, Any], key: str) -> bool:
    """Return ``True`` if every section path from the root to *key*'s
    immediate parent is present in *raw*.

    For ``compile.llm.rlm.enable`` the chain is
    ``["compile", "compile.llm", "compile.llm.rlm"]`` — the parent of
    the leaf is ``compile.llm.rlm``. We only surface nested-model leaves
    when the user actually opened those sections in the TOML.
    """
    parts = key.split(".")
    # All parent prefixes (length 1..len-1 inclusive); we want every one present.
    for i in range(1, len(parts)):
        section = parts[:i]
        cur: Any = raw
        present = True
        for part in section:
            if not isinstance(cur, dict) or part not in cur:
                present = False
                break
            cur = cur[part]
        if not present:
            return False
    return True


def _values_equal(a: Any, b: Any) -> bool:
    """Structural equality, agnostic to list ordering.

    Lists are sorted first so reordering doesn't change the diff result;
    nested dicts are JSON-serialized with sorted keys to keep order
    deterministic.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, bool) != isinstance(b, bool):
        return False

    def _walk(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _walk(inner) for k, inner in sorted(v.items())}
        if isinstance(v, list):
            return sorted(_walk(x) for x in v)
        if isinstance(v, Path):
            return str(v)
        return v

    try:
        return bool(
            json.dumps(_walk(a), sort_keys=True, default=str) == json.dumps(_walk(b), sort_keys=True, default=str),
        )
    except Exception:
        return a == b
