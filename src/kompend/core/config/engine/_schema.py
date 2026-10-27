"""Config schema introspection: DEFAULTS, env map, validation."""

from __future__ import annotations

import contextlib
from typing import Any, get_origin

from pydantic import BaseModel, TypeAdapter
from pydantic.fields import FieldInfo

from kompend.core.config._configs import KompendConfig
from kompend.core.keys import (
    COMPILE_CACHE_DIR,
    COMPILE_RESPECT_GITIGNORE,
    COMPILE_START_AT_TIER,
    COMPILE_STOP_AT_TIER,
    COMPILE_TIER_STATE_FILE,
    EnvVar,
)
from kompend.core.utils.dicts import flat_keys
from kompend.logging import get_logger
from kompend.types import ConfigValue

logger = get_logger(__name__)


def _build_defaults() -> dict[str, Any]:
    """Generate DEFAULTS from the :class:`KompendConfig` schema model."""
    return KompendConfig().model_dump()


def _build_env_map() -> dict[str, str]:
    """Auto-generate _ENV_MAP from :class:`KompendConfig` validation aliases."""
    env_map: dict[str, str] = {}

    def _walk(cls: type[BaseModel], prefix: str = "") -> None:
        for name, info in cls.model_fields.items():
            dotted = f"{prefix}.{name}" if prefix else name
            ann = info.annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                _walk(ann, dotted)
            else:
                alias = info.validation_alias
                if isinstance(alias, str):
                    env_map[dotted] = alias

    _walk(KompendConfig)
    # Env aliases not expressible as field validation aliases (surfaced per-run
    # via CLI overrides). Keys live in ``kompend.core.keys``.
    env_map.update(
        {
            COMPILE_START_AT_TIER: EnvVar.START_AT_TIER,
            COMPILE_STOP_AT_TIER: EnvVar.STOP_AT_TIER,
            COMPILE_TIER_STATE_FILE: EnvVar.TIER_STATE_PATH,
            COMPILE_CACHE_DIR: EnvVar.COMPILE_CACHE_DIR,
            COMPILE_RESPECT_GITIGNORE: EnvVar.RESPECT_GITIGNORE,
        }
    )
    return env_map


DEFAULTS: dict[str, Any] = _build_defaults()


ENV_MAP: dict[str, str] = _build_env_map()

# Keys removed from the schema but still present in older vault configs.
# Stripped (with a warning) before validation so old vaults don't crash on
# upgrade; ``kompend config check`` reports them as deprecations.
REMOVED_KEYS: dict[str, list[str]] = {  # nocheck: str-registry — frozen list of historic config keys, gone from the schema
    "compile.budget": [
        "concepts_per_batch",
        # Output ceilings. `max_tokens` is no longer sent to a model: an
        # answer is bounded by the context window minus its prompt, so the
        # planners derive the room they leave free from the window itself.
        "agent_output_max",
        "extract_output_max",
        "gen_output_max",
        "overview_output_max",
        "reference_output_max",
        "verify_output_max",
        "lint_output_max",
        "wicer_probe_output_max",
        "wicer_eval_output_max",
    ],
    "search.synthesis": ["tokens"],
    "compile.llm": ["role_models", "fallback_max_retries"],
    # `compile.mode` selected a pipeline mode, but the tier flags
    # (`start_at_tier` / `stop_at_tier`) cover the same ground and are the
    # only ones the pipeline reads.
    "compile": ["mode"],
    # The whole [compile.pdf] section. It configured the kompend-owned docling and
    # liteparse backends, which parsecraft supersedes; nothing reads it now.
    # The keys are listed individually so an older vault is stripped with a
    # deprecation warning instead of rejected by the strict schema.
    "compile.pdf": [
        "max_pages_before_chunking",
        "page_chunk_overlap",
        "page_chunk_size",
        "table_structure",
        "table_mode",
        "formula_enrichment",
        "code_enrichment",
        "ocr",
        "ocr_min_text_chars_per_page",
        "num_threads",
    ],
    # `[ingest] parser` selected among kompend's own parser backends. All of those
    # are deleted — parsecraft is the only backend and routes internally — so the
    # selector had nothing left to select between. `parser_fallback` was its chain
    # companion and went when the key became single-valued.
    "ingest": ["parser", "parser_fallback"],
    # The whole [eval] section. It held a score threshold and a history size
    # for an evaluation loop that no longer exists.
    "eval": ["threshold", "history_size"],
    # Display limits without a reader: `error_chars` / `json_chars` were
    # never read (the error and JSON renderers use their own formatting),
    # `query_expansions` has no expander behind it, and `wiki_topics` only
    # fed an index section that was dropped.
    "display": ["error_chars", "json_chars", "query_expansions", "wiki_topics"],
}

# Deprecated config keys → their canonical replacement.  The single source of
# truth for renames: the resolve-time migration (``_precedence``) and
# ``kompend config check`` both read it, so a rename cannot be migrated in one
# place and misreported in another.
#
# A key here may still be a schema field (kept for a release) or already gone
# from the schema.  Either way it is *migrated*, not ignored — which is what
# separates it from REMOVED_KEYS.
DEPRECATED_ALIASES: dict[str, str] = {  # nocheck: str-registry — historic rename map; keys exist only in old vault files
    "search.embed_attachments": "attachments.embed",
    "compile.llm.context_window_cap": "compile.llm.context_window",
}


def validate_config_key(key: str) -> bool:
    """Return ``True`` if *key* is a known leaf config key."""
    return key in _flat_schema()


def _flat_schema() -> set[str]:
    """Return all leaf dotted keys present in :data:`DEFAULTS`."""
    return set(flat_keys(DEFAULTS))


def coerce_config_value(value: str, key: str) -> ConfigValue:
    """Coerce a CLI string to the config schema type for *key*.

    Raises:
        ValueError: when *value* cannot be coerced to the expected type.
    """
    value = value.strip()
    field_info = resolve_field_info(key)
    if field_info is None:
        # The field is unknown to the schema — try int, then float, else keep
        # the raw string. (Each ``suppress`` can fall through independently, so
        # the float branch is reachable, e.g. "1.5".)
        for _conv in (int, float):
            with contextlib.suppress(ValueError):
                return _conv(value)
        return value

    ann = field_info.annotation
    if get_origin(ann) is list:
        return [v.strip() for v in value.split(",") if v.strip()]
    if ann is None:
        # A known key always carries an annotation, so reaching here means the
        # schema is malformed. Coercing against no type would silently accept
        # anything, which is the failure this function exists to prevent, so
        # name the key instead.
        raise ValueError(f"config key {key!r} has no annotation to coerce {value!r} against")

    try:
        return TypeAdapter(ann).validate_python(value)
    except Exception as e:
        raise ValueError(f"cannot coerce {value!r} to {ann} for {key}") from e


def resolve_field_info(key: str) -> FieldInfo | None:
    """Resolve a dotted config key to its pydantic FieldInfo in :class:`KompendConfig`."""
    parts = key.split(".")
    cls: type[BaseModel] = KompendConfig
    for part in parts[:-1]:
        info = cls.model_fields.get(part)
        if info is None:
            return None
        ann = info.annotation
        if not (isinstance(ann, type) and issubclass(ann, BaseModel)):
            return None
        cls = ann
    return cls.model_fields.get(parts[-1])
