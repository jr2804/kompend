"""Tier 1 (parse + chunk) orchestration helpers."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from kompend.agents.vlm import VlmAgent
from kompend.core.config.rumdl import resolve_rumdl_config
from kompend.core.models import (
    ChunkTree,
    DocumentElement,
    ElementType,
    MetadataField,
    ParsedDocument,
    WikiField,
)
from kompend.core.models.attachments import AttachmentManifest, adopt_orphan_attachments
from kompend.core.models.elements import element_text
from kompend.core.paths import VaultPaths
from kompend.core.pipeline.attachments import (
    ingest_attachment,
    iter_ingestible_attachments,
)
from kompend.core.pipeline.config import PipelineConfig
from kompend.core.pipeline.ingest import (
    PARSE_CACHE_PARSER_KEY,
    parse_cache_is_current,
    preserve_cached_metadata,
    relativize_image_paths,
)
from kompend.core.pipeline.parse_cache import (
    content_hash_prefix,
    read_document,
    write_document,
)
from kompend.core.pipeline.runtime import PipelineRuntime
from kompend.core.store import vault_db_path
from kompend.core.text.encoding import normalize_to_utf8_path
from kompend.core.text.rumdl import normalize_element_markdown
from kompend.core.verbosity import is_verbose
from kompend.llm.models import resolve_vlm_model
from kompend.llm.user_providers import effective_workers
from kompend.logging import get_logger
from kompend.preprocessing.dedup import DedupStore
from kompend.preprocessing.parsers.base import FileExtension, scrub_document
from kompend.resolution.base import BaseResolver
from kompend.resolution.chain import ResolverChain
from kompend.resolution.vlm_enricher import VLMEnricher
from kompend.trust import filter_sensitive
from kompend.types import JSONDict

logger = get_logger(__name__)


def parse_and_chunk(pipeline: PipelineRuntime) -> list[ChunkTree]:
    """Parse documents and chunk them, with optional caching and parallelism.

    When tier state is available, sources that already match the
    persisted hash and tier are skipped to keep the per-tier
    execution incremental.

    When content-level duplicate detection has run during a prior ingest,
    the DedupStore is checked. Sources flagged as ``identical`` are
    silently skipped. Sources flagged as ``similar`` get a duplication
    warning attached to their metadata.
    """
    source_files = list(pipeline.iter_sources())
    if not source_files:
        return []

    pipeline.emit("parse", 0, len(source_files), "starting")

    # Auto-adopt orphaned attachments (files under attachments/<slug>/ with
    # no manifest entry — recovery after manifest loss, or files dropped in
    # manually). Flags derive from extension classes + global embed config.
    try:
        adopted = adopt_orphan_attachments(
            Path(pipeline.base_directory),
            ingest_adopted=pipeline.config.attachments_ingest_adopted,
        )
        if adopted:
            logger.info("Adopted %d orphaned attachment(s): %s", len(adopted), [a.path for a in adopted])
    except OSError as exc:
        logger.warning("Attachment adoption failed — continuing: %s", exc)

    dedup_skip, dedup_similar = _load_dedup_state(pipeline)
    if dedup_skip:
        logger.info("Dedup: %d source(s) flagged as identical — will skip", len(dedup_skip))
        source_files = [f for f in source_files if f.stem not in dedup_skip]
        if not source_files:
            pipeline.emit("parse", 0, 0, "all skipped by dedup")
            return []

    workers = max(1, pipeline.config.workers)
    # Clip against the LLM provider's client-side connection budget
    # ([providers.<name>].max_connections — overridable for integrated
    # providers like ollama). Silent throttling is impossible: a clip logs
    # a warning naming both values.
    workers = effective_workers(pipeline.config.llm.default_model or "", workers)

    trees: list[ChunkTree] = []
    if workers > 1 and len(source_files) > 1:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_single_source, pipeline, sf, dedup_similar): sf for sf in source_files}
            for future in as_completed(futures):
                sf = futures[future]
                try:
                    tree = future.result()
                    if tree is not None:
                        trees.append(tree)
                except Exception as exc:
                    logger.error("Worker failed for %s: %s", sf, exc, exc_info=is_verbose())
                finally:
                    completed += 1
                    pipeline.emit("parse", completed, len(source_files), sf.name)
    else:
        for idx, sf in enumerate(source_files):
            tree = _process_single_source(pipeline, sf, dedup_similar)
            if tree is not None:
                trees.append(tree)
            pipeline.emit("parse", idx + 1, len(source_files), sf.name)

    pipeline.emit("parse", len(source_files), len(source_files), "done")

    # ── Process ingestible attachments ──────────────────────────────
    att_trees = _process_attachments(pipeline)
    if att_trees:
        trees.extend(att_trees)

    if pipeline.summarizer is not None:
        for tree in trees:
            pipeline.summarizer.summarize(tree)

    return trees


def _process_single_source(
    pipeline: PipelineRuntime,
    source_file: Path,
    dedup_similar: dict[str, dict[str, object]],
) -> ChunkTree | None:
    """Parse a single source file (cache-aware) and chunk it.

    Returns the :class:`ChunkTree` or *None* if the file was skipped
    (no parser, cache hit with no tree, or parse failure).
    """
    stem = source_file.stem
    use_cache = not pipeline.config.no_cache
    content_hash = ""

    if use_cache and not pipeline.config.force:
        content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()

    parsed = _try_load_parsed_cache(pipeline, source_file, stem, content_hash, use_cache)

    if parsed is None:
        parsed = _parse_source(pipeline, source_file, use_cache, content_hash)
        if parsed is None:
            return None
        if use_cache and not content_hash:
            content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()

    # Normalize source_file to relative POSIX on parsed doc.
    parsed.source_file = Path(pipeline.rel_file(parsed.source_file))
    for el in parsed.elements:
        if el.source_file is not None:
            el.source_file = Path(pipeline.rel_file(el.source_file))

    # Portable cache: absolute image paths -> relative to the vault. Must run
    # BEFORE chunking so asset ids (which hash figure image_ref) stay
    # consistent between the parse cache and the asset store.
    relativize_image_paths(parsed, Path(pipeline.base_directory))

    if pipeline.config.privacy_filter:
        apply_privacy_filter(parsed.elements)

    if not pipeline.chunker:
        return None

    # Attach dedup similarity warning if applicable.
    _attach_dedup_warning(parsed, stem, dedup_similar)

    # Chunk cache lookup (Tier 1).
    tree = _try_load_chunk_cache(pipeline, source_file, content_hash, use_cache)
    if tree is not None:
        return tree

    # Chunk and cache.
    tree = pipeline.chunker.chunk(parsed)
    pipeline.apply_tags_to_tree(tree, source_file)

    if pipeline.enricher is not None:
        tree = pipeline.enricher.enrich_tree(tree, parsed)

    if content_hash:
        pipeline.state_store.record(pipeline.state, stem, Path(pipeline.rel_file(source_file)), content_hash, tier=1)
        chunk_deps = _chunk_cache_deps(pipeline, content_hash)
        pipeline.cache_store.set("chunk", chunk_deps, tree.to_dict())
    return tree


def apply_privacy_filter(elements: list[DocumentElement]) -> None:
    """Apply privacy filter in-place to document elements."""
    for el in elements:
        if isinstance(el.content, str):
            el.content, _ = filter_sensitive(el.content)
        elif isinstance(el.content, list):
            el.content = [[filter_sensitive(cell)[0] for cell in row] for row in el.content]


def _try_load_parsed_cache(pipeline: PipelineRuntime, source_file: Path, stem: str, content_hash: str, use_cache: bool) -> ParsedDocument | None:
    """Try to load a parsed document from cache. Returns *None* on miss."""
    if not use_cache:
        return None
    parsed = _load_cache(pipeline, source_file)
    if parsed is not None:
        return parsed
    if pipeline.config.force:
        return None
    cached_entry = pipeline.state.entries.get(stem)
    if cached_entry and cached_entry.content_hash == content_hash and cached_entry.current_tier >= 1:
        return _load_cache(pipeline, source_file)
    return None


def _parse_source(pipeline: PipelineRuntime, source_file: Path, use_cache: bool, content_hash: str) -> ParsedDocument | None:
    """Parse a source file via the router and optionally cache the result."""
    if pipeline.router is None:
        return None
    try:
        selected = pipeline.router.route(source_file)
        if selected is None:
            logger.warning("No parser backend for %s, skipping", source_file)
            return None
        # Encoding normalization only helps text-format sources; binary
        # containers (PDF/DOCX/raster) are handled byte-wise by their parser
        # and would raise ValueError("binary") if funneled through normalize_to_utf8_path.
        suffix = FileExtension(source_file.suffix.lower())
        parse_target, _encoding = normalize_to_utf8_path(source_file, pipeline.cache_directory()) if suffix.is_text else (source_file, None)
        parsed = selected.parse(parse_target)
        _ = len(parsed.elements)
    except Exception as exc:
        logger.error("Failed to parse %s: %s", source_file, exc, exc_info=is_verbose())
        return None
    # Portable cache: absolute image paths -> relative to the vault, BEFORE the
    # cache write so cache and in-memory doc agree (asset ids hash figure
    # image_ref; the same doc is chunked after this function returns).
    relativize_image_paths(parsed, Path(pipeline.base_directory))
    # Canonical markdown at the part level (one rumdl call per document), so
    # backend formatting differences are levelled before anything downstream
    # reads the content. Content-only: page/reading-order/metadata are kept.
    normalize_element_markdown(
        parsed.elements,
        rumdl_config=resolve_rumdl_config(Path(pipeline.base_directory), pipeline.config.output.rumdl_config),
        flavor=pipeline.config.output.flavor,
    )
    if use_cache:
        _save_cache(pipeline, source_file, parsed)
    return parsed


def _attach_dedup_warning(parsed: ParsedDocument, stem: str, dedup_similar: dict[str, dict[str, object]]) -> None:
    """Attach a dedup similarity warning to the parsed document's metadata."""
    similar_entry = dedup_similar.get(stem)
    if not similar_entry:
        return
    warning = (
        f"> [!warning] Possible duplicate\n"
        f"> This document is similar to **{similar_entry.get('closest_match', 'unknown')}** "
        f"> (score={similar_entry.get('closest_score', 0):.2f}).\n"
    )
    if not parsed.metadata:
        parsed.metadata = {}
    parsed.metadata["_dedup_warning"] = warning


def _try_load_chunk_cache(pipeline: PipelineRuntime, source_file: Path, content_hash: str, use_cache: bool) -> ChunkTree | None:
    """Try to load a chunked tree from cache. Returns *None* on miss."""
    if not (use_cache and content_hash):
        return None
    chunk_deps = _chunk_cache_deps(pipeline, content_hash)
    cached = pipeline.cache_store.get_dict("chunk", chunk_deps)
    if cached is None:
        return None
    stem = source_file.stem
    pipeline.state_store.record(pipeline.state, stem, Path(pipeline.rel_file(source_file)), content_hash, tier=1)
    tree = ChunkTree.from_dict(cached)
    pipeline.apply_tags_to_tree(tree, source_file)
    return tree


def _chunk_cache_deps(pipeline: PipelineRuntime, content_hash: str) -> dict[str, str]:
    """Cache identity for a chunked tree.

    ``source`` + chunker size identify the chunking; ``parser`` carries the
    parser-chain fingerprint so a parser switch re-chunks instead of reusing
    a tree built from the previous backend's output.
    """
    return {
        "source": content_hash,
        "chunk_effective_max": str(getattr(pipeline.chunker, "_effective_max", 0)),
        "parser": pipeline.router.fingerprint() if pipeline.router is not None else "",
        "contextual": "1" if (pipeline.config.search.contextual.enabled and pipeline.enricher is not None) else "0",
    }


def _load_cache(pipeline: PipelineRuntime, source_file: Path) -> ParsedDocument | None:
    """Load a cached ParsedDocument from the store if available."""
    db_path = vault_db_path(Path(pipeline.base_directory))
    key = cache_key(source_file)
    data = read_document(db_path, key)
    if data is None:
        return None
    if pipeline.router is not None and not parse_cache_is_current(data, pipeline.router.fingerprint()):
        # Entry produced by a different parser chain — re-parse so a parser
        # switch takes effect instead of silently serving stale output.
        return None
    try:
        elements = [
            DocumentElement(
                type=ElementType(el["type"]),
                content=el["content"],
                page=el.get("page"),
                reading_order=el.get(MetadataField.READING_ORDER),
                heading_level=el.get("heading_level"),
                metadata=el.get("metadata", {}),
            )
            for el in cast("list[dict[str, Any]]", data.get("elements") or [])
        ]
        # scrub_document: caches written before surrogate scrubbing carry
        # lone \\udXXX escapes that json.loads restores as surrogate chars.
        return scrub_document(
            ParsedDocument(
                source_file=pipeline.abs_file(cast("str", data.get(MetadataField.SOURCE_FILE, ""))),
                elements=elements,
                title=cast("str", data.get("title", "")),
                author=cast("str", data.get("author", "")),
                language=cast("str", data.get("language", "en")),
                page_count=cast("int", data.get("page_count", 0)),
                metadata=cast("dict[str, Any]", data.get(WikiField.METADATA) or {}),
            )
        )
    except (KeyError, ValueError, TypeError) as exc:
        # Corrupt entry = miss: re-parse. Warn so it's visible.
        logger.warning("Corrupt parse cache for %s — reparsing: %s", key, exc, exc_info=is_verbose())
        return None


def _save_cache(pipeline: PipelineRuntime, source_file: Path, doc: ParsedDocument) -> None:
    """Write a ParsedDocument to the store."""
    db_path = vault_db_path(Path(pipeline.base_directory))
    key = cache_key(source_file)
    try:
        # Preserve user-set tags/attributes across the overwrite (a --no-cache
        # compile, or an automatic re-parse after a parser change).
        preserve_cached_metadata(doc, read_document(db_path, key))
        data = {
            str(MetadataField.SOURCE_FILE): pipeline.rel_file(doc.source_file),
            PARSE_CACHE_PARSER_KEY: pipeline.router.fingerprint() if pipeline.router is not None else "",
            "title": doc.title,
            "author": doc.author,
            "language": doc.language,
            "page_count": doc.page_count,
            "metadata": doc.metadata,
            "elements": [
                {
                    "type": str(el.type),
                    # element_text, not str(): bytes would land in the cache (and
                    # every later read of it) as a repr blob.
                    "content": el.content if isinstance(el.content, list) else element_text(el.content),
                    "page": el.page,
                    MetadataField.READING_ORDER: el.reading_order,
                    "heading_level": el.heading_level,
                    "metadata": el.metadata,
                }
                for el in doc.elements
            ],
        }
        write_document(db_path, key, cast("JSONDict", data))
    except Exception as exc:
        logger.debug("Cache write failed for %s — ignoring: %s", key, exc, exc_info=is_verbose())


def _load_dedup_state(pipeline: PipelineRuntime) -> tuple[set[str], dict[str, dict[str, object]]]:
    """Best-effort load of dedup store.

    Returns ``(dedup_skip, dedup_similar)``.
    """
    dedup_skip: set[str] = set()
    dedup_similar: dict[str, dict[str, object]] = {}
    try:
        ds = DedupStore(pipeline.dedup_store_file())
        ds.load()
        if ds.is_fresh():
            for slug in ds.slugs:
                entry = ds.get(slug, {})
                if entry.get("verdict") == "identical":
                    dedup_skip.add(slug)
                elif entry.get("verdict") == "similar":
                    dedup_similar[slug] = entry
    except Exception as exc:
        logger.debug("Dedup store not available — skipping dedup check: %s", exc, exc_info=is_verbose())
    return dedup_skip, dedup_similar


def cache_key(source_file: Path) -> str:
    """Content-discriminated cache key — collision-proof across same-stem files."""
    return f"{source_file.stem}-{content_hash_prefix(source_file.read_bytes())}"


def _process_attachments(pipeline: PipelineRuntime) -> list[ChunkTree]:
    """Process ingestible attachments and return their ChunkTrees.

    Attachments are parsed via the same parser router as sources.
    Image attachments (PNG/JPG) are captioned via VLM if configured.
    Each resulting tree carries ``source_asset=True`` and ``parent_source``
    metadata so the rest of the pipeline can distinguish them.
    """
    vp = VaultPaths(pipeline.base_directory)
    manifest = AttachmentManifest(vp.attachments_manifest_file)
    if not manifest.exists():
        return []

    # Global gate: ``attachments.ingest_adopted`` ANDs with the per-attachment
    # manifest flag. When the gate is off, nothing is ingested regardless of
    # how many attachments were explicitly marked ingestible.
    if not pipeline.config.attachments_ingest_adopted:
        logger.info("Attachment ingestion disabled (attachments.ingest_adopted=false) — skipping %d manifest entry(s)", len(manifest.read_all()))
        return []

    ingestible = iter_ingestible_attachments(vp)
    if not ingestible:
        return []

    router = pipeline.router
    if router is None:
        logger.warning("No parser router available — skipping attachment ingestion")
        return []
    vlm_captioner = None
    vlm_model = resolve_vlm_model(pipeline.config.llm.default_model, pipeline.config.llm.vlm_model)
    if vlm_model:
        try:
            vlm_captioner = VlmAgent(
                vlm_model,
                timeout=pipeline.config.llm.timeout,
            )
        except Exception as exc:
            logger.debug("VLM captioner not available for attachments: %s", exc, exc_info=is_verbose())

    trees: list[ChunkTree] = []
    pipeline.emit("parse", 0, len(ingestible), "starting")
    for idx, att in enumerate(ingestible):
        pipeline.emit("parse", idx + 1, len(ingestible), f"asset: {att.filename}")

        try:
            doc = ingest_attachment(vp, att, router, vlm_captioner=vlm_captioner)
            if doc is None:
                continue

            if not pipeline.chunker:
                continue

            tree = pipeline.chunker.chunk(doc)

            # Stamp source-asset metadata on all leaves so downstream
            # resolve/index stages can distinguish attachment content.
            for node in tree.leaves():
                if node.chunk is None:
                    continue
                node.chunk.metadata["source_asset"] = True
                node.chunk.metadata["parent_source"] = att.source
                node.chunk.metadata["asset_file"] = att.path

            if pipeline.enricher is not None:
                tree = pipeline.enricher.enrich_tree(tree, doc)

            trees.append(tree)
            logger.info("Processed attachment: %s → %d chunk(s)", att.filename, len(list(tree.leaves())))

        except Exception as exc:
            logger.debug("Failed to process attachment %s: %s", att.filename, exc, exc_info=is_verbose())

    if trees:
        logger.info("Processed %d ingestible attachment(s)", len(trees))

    return trees


def build_resolver_chain(config: PipelineConfig) -> BaseResolver:
    """Construct the resolver chain.

    When a VLM model is configured, a :class:`VLMEnricher` enriches
    image chunks with generated captions.
    """
    resolvers: list[BaseResolver] = []

    vlm_model = resolve_vlm_model(config.llm.default_model, config.llm.vlm_model)
    if vlm_model:
        try:
            vlm_agent = VlmAgent(
                vlm_model,
                timeout=config.llm.timeout,
            )
            resolvers.append(
                VLMEnricher(
                    agent=vlm_agent,
                    min_image_bytes=config.vlm_min_image_bytes,
                    min_image_dimension=config.vlm_min_image_dimension,
                )
            )
            logger.debug("VLMEnricher added to resolver chain (model=%s)", vlm_model)
        except Exception as exc:
            logger.debug("VLMEnricher not available: %s", exc, exc_info=is_verbose())

    if not resolvers:
        logger.info("No resolvers configured — chunks will pass through without resolution")

    return ResolverChain(resolvers)
