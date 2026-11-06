"""``kompend index`` — rebuild the vector index without recompiling."""

from __future__ import annotations

from kompend.cli.args import CliSet, EmbedModel, WikiDir
from kompend.cli.settings import build_cli_overrides, config_file_from_ctx, parse_cli_set, vault_paths_from_ctx
from kompend.core.cache.wiki import write_wiki_cache
from kompend.core.config import resolve_config
from kompend.core.models import MetadataField, load_cached_wiki, load_wiki_from_files
from kompend.core.paths import vault_paths
from kompend.logging import get_console, get_err_console
from kompend.rag import build_vector_index


def index_cmd(
    *,
    embed_model: EmbedModel = None,
    wiki_dir: WikiDir = None,
    cli_set: CliSet = None,
) -> None:
    """Rebuild the vector index for semantic search.

    Uses the cached :class:`CompiledWiki` from the ``wiki_snapshot`` table in ``.kompend/kompend.db``.
    This is a recovery command — a normal ``kompend compile`` already builds
    the index automatically via post-compile actions.

    Use it when the index is missing, stale, or you changed the embedding
    model (``compile.embedding.model``) or the ``attachments.embed``
    setting and want to re-embed without a full LLM recompile.

    It does **not** re-ingest attachments or recompile the wiki. To ingest
    newly-added attachments, run ``kompend compile --force --start-at-tier chunk``
    first.
    """
    vp = vault_paths_from_ctx()
    if wiki_dir:
        vp = vault_paths(wiki_dir)

    cli_overrides = build_cli_overrides(cli_set=parse_cli_set(cli_set))
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=cli_overrides, config_file=config_file_from_ctx())
    output_dir = vp.output_dir

    # Index the markdown bundle — the node set ``kompend query`` judges staleness
    # against (a files-loaded wiki carries every page; the snapshot graph may
    # not, e.g. per-source summaries).
    cached = load_cached_wiki(vp.base_dir)
    wiki = load_wiki_from_files(output_dir) or cached
    if wiki is None:
        get_err_console().print(f"No compiled wiki found in {output_dir}. Run 'kompend compile' first.")
        raise SystemExit(1)

    # Resolve embed model (CLI flag > config > compile.embedding.model).
    resolved_model = embed_model or cfg.kompend.compile.embedding.model
    if not resolved_model:
        get_err_console().print(
            "No embedding model configured. Set compile.embedding.model in config or pass --embed-model.",
        )
        raise SystemExit(1)

    embed_attachments = cfg.kompend.attachments.embed

    try:
        count = build_vector_index(
            wiki=wiki,
            output_dir=output_dir,
            embed_model=resolved_model,
            base_dir=vp.base_dir,
            embed_attachments=embed_attachments,
        )
    except Exception as exc:
        get_err_console().print(f"Failed to build vector index: {exc}")
        raise SystemExit(1) from exc

    # Persist embedding model metadata so ``kompend query`` can open the store.
    # Written through the snapshot wiki, which is what the cache holds (the
    # files-loaded wiki above has no graph metrics to cache).
    if cached is not None:
        cached.metadata[str(MetadataField.EMBEDDING_MODEL)] = resolved_model
        write_wiki_cache(cached, vp.base_dir)

    if count > 0:
        get_console().print(f"Indexed {count} nodes in the vector store for semantic search.")
    else:
        get_err_console().print("Warning: no nodes indexed — vector index is empty.")
