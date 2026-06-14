"""KNOX CLI commands."""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

# Silence noisy third-party loggers that corrupt the rich progress bar
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

from knox.capture.capture import capture
from knox.capture.crystallize import crystallize
from knox.capture.ingest import ingest
from knox.capture.learn import learn
from knox.cli._progress import PipelineProgress
from knox.cli.app import app
from knox.cli.args import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPILE_FORMAT,
    DEFAULT_COMPILE_MODE,
    DEFAULT_COMPILER,
    DEFAULT_IMAGE_MODE,
    DEFAULT_LLM_FALLBACK_MAX_RETRIES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_BUDGET,
    DEFAULT_SOURCES_DIR,
    BudgetConcepts,
    BudgetGenSegment,
    BypassReview,
    CaptureText,
    ChunkSize,
    CompileFiles,
    CompileLint,
    CompileMode,
    CompilerChoice,
    ConfigPath,
    EmbedDim,
    EmbedModel,
    ForceTier,
    Frontmatter,
    Ignore,
    ImageMode,
    IndexStore,
    IngestTarget,
    JsonOutput,
    KindOption,
    LearnTags,
    LearnText,
    LintFix,
    LintLlm,
    LintOutputDir,
    LintRule,
    LintSourcesDir,
    LintThreshold,
    ListTiers,
    LlmContextBudget,
    LlmDefaultModel,
    LlmFallbackChain,
    LlmFallbackMaxRetries,
    LlmRoleModel,
    LlmTimeout,
    MemoryDecayThreshold,
    MemoryEnabled,
    MemoryHalfLifeDays,
    NoCacheFlag,
    NoPrivacy,
    OnlineUrls,
    OutputDir,
    OutputFlavor,
    ParseFiles,
    PrintElements,
    Private,
    Progress,
    PromptBudget,
    Provenance,
    QueryFormat,
    QueryNoClassify,
    QueryRaw,
    QuerySearchMode,
    QuerySourcesDir,
    QueryText,
    QueryTopK,
    Recursive,
    RelationConfidenceThreshold,
    ResolveRelations,
    ReviewMode,
    ServeHost,
    ServeOpen,
    ServePort,
    SourcesDir,
    SynthesizeFlag,
    TargetFormat,
    UrlFile,
    Verbose,
    Verify,
    VlmModel,
    Workers,
)
from knox.core.config import resolve_config
from knox.core.hooks import HookEvent, clear_hooks, fire_hook, load_hooks_from_config
from knox.core.models import CompiledWiki, load_cached_wiki
from knox.core.pipeline import PipelineConfig, build_pipeline
from knox.lint.llm_rules import LLM_RULE_BY_NAME
from knox.lint.runner import LintRunner, report_to_json, report_to_text
from knox.llm.client import LLMClient
from knox.mcp.server import main as mcp_main
from knox.output.markdown import MarkdownWriter
from knox.output.models import Document
from knox.output.wiki_index import write_index_and_log, write_overview
from knox.packs.base import get_registry
from knox.packs.builtins import academic_pack, technical_pack
from knox.preprocessing.parsers._registry import build_default_router, get_registered_parsers
from knox.provenance import annotate_wiki_with_provenance
from knox.rag import build_vector_index
from knox.rag.zvec_store import ZvecWikiStore
from knox.review import ReviewQueue
from knox.search.classifier import classify
from knox.search.index import build_chunk_index
from knox.search.retriever import HybridRetriever, TokenOverlapRetriever
from knox.trust import mark_wiki_private
from knox.verification import verify_wiki
from knox.web.server import serve

from ._sources import download_urls, ensure_in_sources, resolve_files
from .doctor import print_doctor_results, run_doctor
from .status import collect_stats, print_status

logger = logging.getLogger(__name__)

# ── parse ───────────────────────────────────────────────────────────────


def _uses_llm_rule(rule: list[str] | None) -> bool:
    """Return True if any requested rule name is LLM-powered."""
    if not rule:
        return False
    return any(r in LLM_RULE_BY_NAME for r in rule)


def _load_hooks() -> None:
    """Load event hooks from the resolved config into the global registry."""
    clear_hooks()
    try:
        cfg = resolve_config()
        load_hooks_from_config(cfg.raw)
    except Exception:
        logger.debug("Could not load hooks from config", exc_info=True)


def _make_progress_callback(tracker: PipelineProgress) -> Callable[[str, int, int, str], None]:
    """Build a progress callback that translates pipeline events into Rich updates."""

    def _cb(phase: str, current: int, total: int, detail: str) -> None:
        if phase == "parse":
            if current == 0:
                tracker.start()
            elif detail == "done":
                tracker.message(f"Parsed {total} file(s)")
            else:
                tracker.start_file(detail)
                tracker.step("Parse")
                tracker.step("Chunk")
                tracker.finish_file(detail, "OK")
        elif phase == "index":
            if current == 0:
                tracker.start()
                tracker.message("Downloading sources…")
            else:
                tracker.message("Download complete")
        elif phase == "resolve":
            if current == 0:
                tracker.message("Resolving chunks…")
            else:
                tracker.message("Resolve complete")
        elif phase == "compile":
            if current == 0:
                tracker.message("Compiling wiki…")
            else:
                tracker.message("Compile complete")
        elif phase == "output":
            if current == 0:
                tracker.message("Writing output…")
            else:
                tracker.message("Output complete")

    return _cb


def _resolve_compile_inputs(
    files: CompileFiles,
    online_urls: list[str],
    url_file: UrlFile,
    sources_dir_path: Path,
    recursive: Recursive,
    ignore: Ignore,
) -> list[Path]:
    """Resolve file arguments and URLs into a deduplicated input file list."""
    if url_file is not None:
        for raw_line in url_file.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#"):
                online_urls.append(stripped)

    resolved = resolve_files(*files, recursive=recursive, ignore_patterns=ignore)

    sources_dir_path.mkdir(parents=True, exist_ok=True)
    local_files = ensure_in_sources(resolved, sources_dir_path)
    downloaded = download_urls(online_urls, sources_dir_path)

    all_input_files = list(dict.fromkeys(local_files + downloaded))
    if not all_input_files:
        all_input_files = resolve_files(
            str(sources_dir_path),
            recursive=recursive,
            ignore_patterns=ignore,
        )
    return all_input_files


@app.command(name="parse")
def parse_cmd(
    files: ParseFiles,
    recursive: Recursive = False,
    print_elements: PrintElements = False,
) -> None:
    """Parse documents and show their structure.

    Accepts one or more file paths, glob patterns (e.g. *.pdf), or directories.
    Use --recursive to descend into subdirectories.
    """
    resolved = resolve_files(*files, recursive=recursive)
    if not resolved:
        typer.echo("No supported files found.", err=True)
        raise typer.Exit(1)

    router = build_default_router()

    for input_file in resolved:
        try:
            doc = router.parse(input_file)
            typer.echo(f"\n{'─' * 60}")
            typer.echo(f"  {doc.title or input_file.name}")
            typer.echo(f"  Pages: {doc.page_count}  |  Elements: {len(doc.elements)}")
            if print_elements:
                for el in doc.elements[:20]:
                    typer.echo(f"    [{el.type.name:10s}] {str(el.content)[:120]}")
                if len(doc.elements) > 20:
                    typer.echo(f"    ... ({len(doc.elements) - 20} more)")
        except Exception as exc:
            typer.echo(f"Error parsing {input_file}: {exc}", err=True)


# ── compile ─────────────────────────────────────────────────────────────


@app.command(name="compile")
def compile_cmd(
    files: CompileFiles,
    sources_dir: SourcesDir = None,
    output_dir: OutputDir = None,
    config: ConfigPath = None,
    online_urls: OnlineUrls = None,
    url_file: UrlFile = None,
    recursive: Recursive = False,
    target_format: TargetFormat = DEFAULT_COMPILE_FORMAT,
    chunk_size: ChunkSize = DEFAULT_CHUNK_SIZE,
    image_mode: ImageMode = DEFAULT_IMAGE_MODE,
    index_store: IndexStore = False,
    embed_model: EmbedModel = None,
    ignore: Ignore = None,
    show_progress: Progress = None,
    mode: CompileMode = DEFAULT_COMPILE_MODE,
    no_cache: NoCacheFlag = False,
    workers: Workers = 1,
    force: ForceTier = False,
    list_tiers: ListTiers = False,
    prompt_budget: PromptBudget = DEFAULT_PROMPT_BUDGET,
    review: ReviewMode = False,
    bypass_review: BypassReview = False,
    provenance: Provenance = False,
    private: Private = False,
    no_privacy: NoPrivacy = False,
    verify: Verify = False,
    flavor: OutputFlavor = None,
    frontmatter: Frontmatter = None,
    kind: KindOption = None,
    compiler: CompilerChoice = DEFAULT_COMPILER,
    synthesize: SynthesizeFlag = False,
    llm_fallback_chain: LlmFallbackChain = None,
    llm_fallback_max_retries: LlmFallbackMaxRetries = DEFAULT_LLM_FALLBACK_MAX_RETRIES,
    llm_default_model: LlmDefaultModel = None,
    vlm_model: VlmModel = None,
    resolve_relations: ResolveRelations = False,
    relation_confidence_threshold: RelationConfidenceThreshold = 0.0,
    memory_enabled: MemoryEnabled = True,
    memory_half_life_days: MemoryHalfLifeDays = 90.0,
    memory_decay_threshold: MemoryDecayThreshold = 0.3,
    llm_role_model: LlmRoleModel = None,
    compile_lint: CompileLint = False,
    budget_gen_segment: BudgetGenSegment = None,
    budget_concepts: BudgetConcepts = None,
    embed_dim: EmbedDim = None,
    llm_timeout: LlmTimeout = None,
) -> None:
    """Compile documents into a knowledge wiki.

    Source files are gathered from the given paths (files, globs, directories)
    and consolidated into the sources directory. If a file is already there it
    stays in place; otherwise it is copied in. URLs are downloaded into the
    sources directory as well.

    The sources directory and output directory default to 'sources' and 'wiki'
    in the current working directory respectively.
    """
    if online_urls is None:
        online_urls = []
    sources_dir_path = Path(sources_dir) if sources_dir else Path(DEFAULT_SOURCES_DIR)
    output_dir_path = Path(output_dir) if output_dir else Path(DEFAULT_OUTPUT_DIR)

    all_input_files = _resolve_compile_inputs(
        files, online_urls, url_file, sources_dir_path, recursive, ignore,
    )
    if not all_input_files:
        typer.echo(f"No supported files found in {sources_dir_path}.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Compiling {len(all_input_files)} document(s) -> {output_dir_path}")

    # 6. Resolve config: CLI > env > project config > global config > defaults
    cli_overrides = _build_cli_overrides(
        sources_dir=sources_dir,
        output_dir=output_dir,
        target_format=target_format,
        chunk_size=chunk_size,
        image_mode=image_mode,
        mode=mode,
        no_cache=no_cache,
        workers=workers,
        force=force,
        review=review,
        provenance=provenance,
        private=private,
        no_privacy=no_privacy,
        verify=verify,
        synthesize=synthesize,
        index_store=index_store,
        compile_lint=compile_lint,
        flavor=flavor,
        frontmatter=frontmatter,
        ignore=ignore,
        llm_default_model=llm_default_model,
        vlm_model=vlm_model,
        resolve_relations=resolve_relations,
        relation_confidence_threshold=relation_confidence_threshold,
        memory_enabled=memory_enabled,
        memory_half_life_days=memory_half_life_days,
        memory_decay_threshold=memory_decay_threshold,
        llm_fallback_chain=llm_fallback_chain,
        llm_fallback_max_retries=llm_fallback_max_retries,
        llm_role_model=llm_role_model,
        kind=kind,
        compiler=compiler,
        budget_gen_segment=budget_gen_segment,
        budget_concepts=budget_concepts,
        embed_dim=embed_dim,
        embed_model=embed_model,
        llm_timeout=llm_timeout,
    )
    cfg = resolve_config(cli_overrides=cli_overrides, config_path=config)
    pipeline_config = PipelineConfig.from_resolved(
        cfg,
        source_dir=sources_dir_path,
        output_dir=output_dir_path,
        input_files=all_input_files,
    )

    pipeline = build_pipeline(pipeline_config)
    _load_hooks()

    if list_tiers:
        _print_tier_listing(pipeline)
        return

    # ── Progress bar ──────────────────────────────────────────────────
    tracker = PipelineProgress.create(
        total_files=len(all_input_files),
        force_progress=show_progress,
    )
    progress_cb = _make_progress_callback(tracker) if tracker.enabled else None

    try:
        start_time = time.monotonic()
        wiki = pipeline.run(progress_callback=progress_cb)
        elapsed = time.monotonic() - start_time

        # Enrich wiki metadata with timing and source info
        wiki.metadata["duration_seconds"] = round(elapsed, 2)
        wiki.metadata["source_count"] = len(all_input_files)
        wiki.metadata["source_names"] = [Path(f).name for f in all_input_files[:20]]

        tracker.finish()
        typer.echo(f"Compiled wiki written to {output_dir_path}")

        _post_compile_actions(
            wiki=wiki,
            pipeline_config=pipeline_config,
            output_dir_path=output_dir_path,
            sources_dir_path=sources_dir_path,
            review=review,
            bypass_review=bypass_review,
            provenance=provenance,
            private=private,
            verify=verify,
            synthesize=synthesize,
            index_store=index_store,
            compile_lint=compile_lint,
            embed_model=embed_model,
        )

    except Exception as exc:
        tracker.stop()
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


# ── doctor ──────────────────────────────────────────────────────────────


@app.command(name="doctor")
def doctor_cmd(
    verbose: Verbose = False,
    json_output: JsonOutput = False,
) -> None:
    """Validate configuration and connectivity.

    Runs health checks for source/output directories, parser availability,
    environment variables, and optionally LLM connectivity.
    """
    sources_dir = Path(DEFAULT_SOURCES_DIR) if Path(DEFAULT_SOURCES_DIR).exists() else None
    output_dir = Path(DEFAULT_OUTPUT_DIR) if Path(DEFAULT_OUTPUT_DIR).exists() else None
    results = run_doctor(
        sources_dir=sources_dir,
        output_dir=output_dir,
        verbose=verbose,
    )
    exit_code = print_doctor_results(results, json_output=json_output)
    if exit_code:
        raise typer.Exit(1)


# ── status ──────────────────────────────────────────────────────────────


@app.command(name="status")
def status_cmd(
    json_output: JsonOutput = False,
) -> None:
    """Show per-wiki statistics.

    Displays source count, article count, word count, last compile time,
    and config path.
    """
    sources_dir = Path(DEFAULT_SOURCES_DIR) if Path(DEFAULT_SOURCES_DIR).exists() else None
    output_dir = Path(DEFAULT_OUTPUT_DIR) if Path(DEFAULT_OUTPUT_DIR).exists() else None
    stats = collect_stats(sources_dir=sources_dir, output_dir=output_dir)
    print_status(stats, json_output=json_output)


# ── parser subcommand group ─────────────────────────────────────────────


@app.command(name="list-parsers")
def list_parsers_cmd(
    json_output: JsonOutput = False,
) -> None:
    """List all registered parsers (built-in + external + HTTP)."""
    # Ensure built-in parsers are registered, then read from the registry
    build_default_router()

    parsers = get_registered_parsers()

    if json_output:
        json.dump(parsers, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    if not parsers:
        typer.echo("No parsers registered.")
        return

    typer.echo(f"Registered parsers ({len(parsers)}):")
    for p in parsers:
        exts = ", ".join(p.get("extensions", []) or [])
        ptype = p.get("type", "built-in")
        typer.echo(f"  • {p['name']} ({ptype})  [{exts}]")


# ── write (Q18) ─────────────────────────────────────────────────────────


@app.command(name="write")
def write_cmd(
    page_title: str = typer.Argument(..., help="Title of the page to create."),
    source: list[str] = typer.Option([], "--source", "-s", help="Source file(s) to include (repeatable)."),  # noqa: B008
    frontmatter: Frontmatter = None,
) -> None:
    """Create a one-shot wiki page from a title and optional sources.

    Example:
        knox write "Transformer architecture" --source attention.pdf --source bert.pdf --frontmatter language=en
    """
    # Parse frontmatter from key=value pairs
    fm: dict[str, str] = {}
    if frontmatter:
        for kv in frontmatter:
            if "=" in kv:
                k, v = kv.split("=", 1)
                fm[k.strip()] = v.strip()

    # Collect source content
    body_parts: list[str] = []
    if source:
        source_files = resolve_files(*source)
        for sf in source_files:
            try:
                content = sf.read_text(encoding="utf-8", errors="replace")[:2000]
                body_parts.append(f"From {sf.name}:\n\n{content}\n")
            except Exception as exc:
                typer.echo(f"Warning: could not read {sf}: {exc}", err=True)

    body = "\n\n".join(body_parts) if body_parts else "_(Content pending)_"

    # Render via flvor-aware MarkdownWriter

    writer = MarkdownWriter()

    doc = Document(title=page_title, body=body, frontmatter=fm)
    rendered = writer.renderer.render(doc)

    output_path = Path.cwd() / "wiki" / f"{page_title.lower().replace(' ', '-')}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    typer.echo(f"Written: {output_path}")


# ── provenance (Q16) ────────────────────────────────────────────────────


@app.command(name="provenance")
def provenance_cmd(
    page_slug: str = typer.Argument(..., help="Page slug or title to show provenance for."),
    with_content: bool = typer.Option(False, "--with-content", help="Show cited content snippets."),
    json_output: JsonOutput = False,
) -> None:
    """Show source-to-article provenance for a compiled wiki page.

    Example:
        knox provenance self-attention
        knox provenance self-attention --with-content
    """
    # Read the compiled wiki
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    base_dir = output_dir.parent
    wiki_file = output_dir / "wiki.md"
    if not wiki_file.exists():
        typer.echo(f"No compiled wiki found at {wiki_file}. Run 'knox compile' first.", err=True)
        raise typer.Exit(1)

    # Try loading the wiki from cache
    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        typer.echo("Could not load compiled wiki metadata.", err=True)
        raise typer.Exit(1)

    # Find the matching node
    nodes = [n for n in wiki.nodes if n.id == page_slug or n.title.lower() == page_slug.lower()]
    if not nodes:
        typer.echo(f"Page '{page_slug}' not found in compiled wiki.", err=True)
        raise typer.Exit(1)

    node = nodes[0]
    source_chunks = node.source_chunks or []

    if json_output:
        data = {"page": node.title, "slug": node.id, "sources": source_chunks}
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    typer.echo(f"# {node.title}\n")
    if source_chunks:
        for chunk_id in source_chunks:
            typer.echo(f"  Source: {chunk_id}")
    else:
        typer.echo("  (no provenance tracked)")


# ── review (N8) ─────────────────────────────────────────────────────────


@app.command(name="review")
def review_cmd(
    action: Annotated[
        str,
        typer.Argument(help="Action: list | approve | reject | lock | unlock"),
    ],
    slug: Annotated[
        str | None,
        typer.Argument(help="Page slug (required for approve/reject/lock/unlock)."),
    ] = None,
    reason: Annotated[
        str | None,
        typer.Option("--reason", help="Rejection reason (for 'reject')."),
    ] = None,
    approve_all: Annotated[
        bool,
        typer.Option("--all", help="Approve every pending page."),
    ] = False,
    output_dir: OutputDir = None,
    json_output: JsonOutput = False,
) -> None:
    """Manage the review queue for compiled pages."""
    out = Path(output_dir) if output_dir else Path(DEFAULT_OUTPUT_DIR)
    base_dir = out.parent
    queue = ReviewQueue(review_dir=base_dir / ".knox" / "review")
    _dispatch_review_action(queue, action, slug, reason, approve_all, json_output)


def _review_list(queue: object, json_output: bool) -> None:
    entries = queue.list_entries()
    if json_output:
        typer.echo(json.dumps([e.to_dict() for e in entries], indent=2, default=str))
        return
    if not entries:
        typer.echo("No pages in the review queue.")
        return
    for e in entries:
        typer.echo(f"  [{e.status:9s}] {e.slug}  ({e.title})")


def _review_lock(queue: object, slug: str | None) -> None:
    if not slug:
        typer.echo("Lock requires a slug.", err=True)
        raise typer.Exit(1)
    if queue.lock(slug):
        typer.echo(f"Locked: {slug}")
        return
    typer.echo(f"Cannot lock {slug!r}: no approved page found.", err=True)
    raise typer.Exit(1)


def _review_unlock(queue: object, slug: str | None) -> None:
    lock_path = queue.review_dir / f"{slug}.locked" if slug else None
    if lock_path and lock_path.exists():
        lock_path.unlink()
        typer.echo(f"Unlocked: {slug}")
        return
    typer.echo(f"No lock for {slug!r}.", err=True)
    raise typer.Exit(1)


def _dispatch_review_action(
    queue: object,
    action: str,
    slug: str | None,
    reason: str | None,
    approve_all: bool,
    json_output: bool,
) -> None:
    """Dispatch a review command action. Extracted to keep ruff PLR0911 happy."""
    if action == "list":
        _review_list(queue, json_output)
        return

    if action == "approve":
        approved = queue.approve(slug or "", approve_all=approve_all)
        if not approved:
            typer.echo("No matching pages to approve.", err=True)
            raise typer.Exit(1)
        for e in approved:
            typer.echo(f"Approved: {e.slug}")
        return

    if action == "reject":
        if not slug:
            typer.echo("Reject requires a slug.", err=True)
            raise typer.Exit(1)
        rejected = queue.reject(slug, reason=reason or "")
        if not rejected:
            typer.echo(f"No pending page with slug {slug!r}.", err=True)
            raise typer.Exit(1)
        for e in rejected:
            typer.echo(f"Rejected: {e.slug}")
        return

    if action == "lock":
        _review_lock(queue, slug)
        return

    if action == "unlock":
        _review_unlock(queue, slug)
        return

    typer.echo(f"Unknown review action: {action!r}. Use list|approve|reject|lock|unlock.", err=True)
    raise typer.Exit(1)


# ── lint ───────────────────────────────────────────────────────────────


@app.command(name="lint")
def lint_cmd(
    output_dir: LintOutputDir = None,
    sources_dir: LintSourcesDir = None,
    fix: LintFix = False,
    llm: LintLlm = False,
    rule: LintRule = None,
    threshold: LintThreshold = None,
    json_output: JsonOutput = False,
    llm_default_model: LlmDefaultModel = None,
    llm_context_budget: LlmContextBudget = None,
) -> None:
    """Lint a compiled wiki for quality issues.

    Runs 8 structural rules by default. Add --llm to also run 3
    LLM-powered rules (slow, expensive). Use --rule to run only specific
    rules and --fix to auto-fix supported issues.

    Examples:
        knox lint                       # structural rules only
        knox lint --llm                 # structural + LLM rules
        knox lint --fix                 # auto-fix what's possible
        knox lint --rule broken-wikilinks --rule orphan-page
        knox lint --threshold 80        # exit 1 if score < 80
    """
    output_dir_path = Path(output_dir) if output_dir else Path(DEFAULT_OUTPUT_DIR)
    base_dir = output_dir_path.parent
    sources_dir_str = sources_dir if sources_dir else (DEFAULT_SOURCES_DIR if Path(DEFAULT_SOURCES_DIR).exists() else None)

    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        typer.echo(f"No compiled wiki found in {base_dir}. Run 'knox compile' first.", err=True)
        raise typer.Exit(1)

    llm_client = None
    if llm or _uses_llm_rule(rule):
        if not llm_default_model:
            typer.echo(
                "LLM-powered lint rules require --llm-default-model (e.g. --llm-default-model openai/gpt-4o).",
                err=True,
            )
            raise typer.Exit(1)

        llm_client = LLMClient(default_model=llm_default_model)

    runner = LintRunner(
        rules=list(rule) if rule else None,
        enable_llm=llm,
        llm_client=llm_client,
        lint_page_budget=llm_context_budget,
        fix=fix,
        sources_dir=sources_dir_str,
    )
    report = runner.run(wiki)
    _load_hooks()
    fire_hook(HookEvent.ON_LINT, report=report, wiki=wiki)

    if json_output:
        typer.echo(report_to_json(report))
    else:
        typer.echo(report_to_text(report))

    if threshold is not None and report.health_score < threshold:
        typer.echo(f"\nHealth score {report.health_score:.0f}% is below threshold {threshold}%.", err=True)
        raise typer.Exit(1)


# ── config show ───────────────────────────────────────────────────────


@app.command(name="config")
def config_cmd(
    config: ConfigPath = None,
    show_source: bool = typer.Option(False, "--source", help="Show where each value comes from."),
    json_output: JsonOutput = False,
) -> None:
    """Show resolved KNOX configuration.

    Displays the merged configuration from all layers:
    CLI > env vars > project knox.toml > global config > defaults.

    Examples:
        knox config                # show resolved config
        knox config --source       # show where each value comes from
        knox config --json         # machine-readable output
    """
    cfg = resolve_config(config_path=config)
    flat = cfg.flat()

    if json_output:
        data = {k: {"value": v, "source": str(cfg.source_of(k))} for k, v in flat.items()}
        typer.echo(json.dumps(data, indent=2, default=str))
        return

    max_key_len = max(len(k) for k in flat) if flat else 20
    for key in sorted(flat):
        value = flat[key]
        if show_source:
            source = cfg.source_of(key)
            typer.echo(f"  {key:<{max_key_len}}  {str(value):<30}  [{source}]")
        else:
            typer.echo(f"  {key:<{max_key_len}}  {value}")

    if cfg.project_config_path:
        typer.echo(f"\n  Project config: {cfg.project_config_path}")
    if cfg.global_config_path:
        typer.echo(f"  Global config:  {cfg.global_config_path}")


# ── mcp server ──────────────────────────────────────────────────────────


@app.command(name="mcp")
def mcp_cmd() -> None:
    """Start the KNOX MCP server over stdio transport.

    Connects to Claude Code, Cursor, and other MCP-compatible clients.
    Uses the official FastMCP SDK with JSON-RPC 2.0 protocol.

    Examples:
        knox mcp                        # start stdio server
        knox mcp --help                 # this help
    """
    mcp_main()


# ── pack commands ───────────────────────────────────────────────────────


@app.command(name="pack-list")
def pack_list_cmd() -> None:
    """List installed contribution packs.

    Examples:
        knox pack-list
    """
    registry = get_registry()
    if not registry.names:
        typer.echo("No packs registered.")
        return
    typer.echo(f"Registered packs ({len(registry.names)}):")
    for pack in registry.all:
        typer.echo(f"  {pack.name:<20} v{pack.version}  {pack.description}")


@app.command(name="pack-install")
def pack_install_cmd(
    name: str = typer.Argument(..., help="Pack name to install (e.g. 'academic')."),
) -> None:
    """Install a built-in contribution pack.

    Examples:
        knox pack-install academic
        knox pack-install technical
    """
    registry = get_registry()
    pack_map = {
        "academic": academic_pack,
        "technical": technical_pack,
    }
    pack = pack_map.get(name.lower())
    if pack is None:
        available = ", ".join(sorted(pack_map))
        typer.echo(f"Unknown pack '{name}'. Available: {available}", err=True)
        raise typer.Exit(1)
    registry.register(pack)
    typer.echo(f"Installed pack '{pack.name}' v{pack.version}")


# ── query ────────────────────────────────────────────────────────────────


def _print_raw_results(results: list[Any], format: str) -> None:
    if format == "json":
        output = [
            {
                "article": r.title,
                "source": r.source_file,
                "snippet": r.snippet,
                "score": round(r.score, 3),
            }
            for r in results
        ]
        typer.echo(json.dumps(output, indent=2))
    else:
        for r in results:
            typer.echo(f"  [{r.score:.3f}] {r.title}")
            if r.source_file:
                typer.echo(f"        Source: {r.source_file}")
            typer.echo(f"        {r.snippet}")
            typer.echo("")


def _classify_intent(query: str, no_classify: bool, format: str) -> bool:
    """Run intent classification and respond directly if classified.

    Returns True if the classifier produced a response (caller should return).
    """
    if no_classify:
        return False

    result = classify(query)
    if result.response is None:
        return False
    if format == "json":
        typer.echo(
            json.dumps(
                {
                    "intent": result.intent.value,
                    "response": result.response,
                },
                indent=2,
            )
        )
    else:
        typer.echo(result.response)
    return True


def _execute_query_search(
    retriever: HybridRetriever,
    query: str,
    search_mode: str,
    sources_dir_path: Path | None,
    top_k: int,
) -> tuple[list[Any], bool]:
    """Dispatch to the requested search mode and return results + raw flag."""
    raw = False
    expand_graph = search_mode == "graph"
    mode = "hybrid" if search_mode == "graph" else search_mode

    if mode == "raw":
        if sources_dir_path is None:
            typer.echo("No sources directory found. Use --sources-dir to specify.", err=True)
            raise typer.Exit(1)
        results = retriever.search_raw(query, sources_dir_path, top_k=top_k)
        raw = True
    elif mode == "hybrid":
        results = retriever.search_hybrid(query, sources_dir_path, top_k=top_k, expand_graph=expand_graph)
    else:
        results = retriever.search_wiki(query, top_k=top_k)
    return results, raw


@app.command(name="query")
def query_cmd(
    query: QueryText,
    top_k: QueryTopK = 5,
    raw: QueryRaw = False,
    format: QueryFormat = "text",
    output_dir: OutputDir = None,
    config: ConfigPath = None,
    no_classify: QueryNoClassify = False,
    search_mode: QuerySearchMode = "hybrid",
    sources_dir: QuerySourcesDir = None,
) -> None:
    """Ask a natural language question against the compiled wiki.

    Retrieves relevant context from the last compiled wiki and optionally
    synthesizes an answer using the configured LLM. Use --raw to skip LLM
    synthesis and see the source context directly.
    """
    output_dir_path = Path(output_dir) if output_dir else Path(DEFAULT_OUTPUT_DIR)
    base_dir = output_dir_path.parent
    _load_hooks()

    if _classify_intent(query, no_classify, format):
        return

    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        typer.echo(
            f"No compiled wiki found in {base_dir}. Run 'knox compile' first.",
            err=True,
        )
        raise typer.Exit(1)

    sources_dir_path: Path | None = Path(sources_dir) if sources_dir else (Path(DEFAULT_SOURCES_DIR) if Path(DEFAULT_SOURCES_DIR).is_dir() else None)

    vector_store = _open_vector_store(base_dir, wiki)
    try:
        retriever = HybridRetriever(
            wiki,
            index_path=base_dir / ".knox" / "search.db",
            vector_store=vector_store,
            sources_dir=sources_dir_path,
        )
        results, raw = _execute_query_search(retriever, query, search_mode, sources_dir_path, top_k)
    finally:
        if vector_store is not None:
            vector_store.close()

    fire_hook(HookEvent.ON_QUERY, query=query, results=results, search_mode=search_mode)

    if not results:
        typer.echo("No relevant content found.")
        raise typer.Exit(0)

    if raw:
        _print_raw_results(results, format)
        return

    llm_client = _try_build_llm_client(config)
    if llm_client is None:
        typer.echo("Warning: No LLM configured. Showing raw context instead.", err=True)
        _print_raw_results(results, "text")
        return

    try:
        synthesizer = TokenOverlapRetriever(wiki)
        synthesis = synthesizer.synthesize(query, results, llm_client=llm_client)
    except Exception as exc:
        typer.echo(f"LLM synthesis failed: {exc}", err=True)
        typer.echo("Showing raw context instead:", err=True)
        _print_raw_results(results, "text")
        raise typer.Exit(1)

    if format == "json":
        typer.echo(format_json_output(synthesis, results))
    else:
        typer.echo(synthesis.answer)
        typer.echo("")
        typer.echo("Sources:")
        c: dict[str, Any]
        for i, c in enumerate(synthesis.citations, 1):
            source_str: str = c.get("source") or "wiki"
            typer.echo(f"  [{i}] {c['article']} ({source_str})")


# ── capture ──────────────────────────────────────────────────────────────


@app.command(name="capture")
def capture_cmd(
    text: CaptureText,
    sources_dir: SourcesDir = None,
) -> None:
    """Extract knowledge items from text and save as source files.

    Uses LLM to extract decisions, discoveries, corrections, and references
    from conversation text or notes. Results are written to sources/_captured/.

    Examples:
        knox capture "We decided to use QKV projections with RoPE"
    """
    sources_dir_path = Path(sources_dir) if sources_dir else Path(DEFAULT_SOURCES_DIR)

    try:
        cfg = resolve_config()
        default_model = cfg.get("compile.llm.default_model")
        llm_client = LLMClient(default_model=default_model)
    except Exception:
        llm_client = None

    paths = capture(text, sources_dir=sources_dir_path, llm_client=llm_client)
    typer.echo(f"Captured {len(paths)} item(s) to {sources_dir_path / '_captured'}")
    for p in paths:
        typer.echo(f"  {p.name}")


# ── learn ────────────────────────────────────────────────────────────────


@app.command(name="learn")
def learn_cmd(
    text: LearnText,
    tag: LearnTags = None,
    sources_dir: SourcesDir = None,
) -> None:
    """Store a single fact directly as a source file.

    Unlike 'capture', this stores the text verbatim without LLM extraction.
    Results are written to sources/_captured/.

    Examples:
        knox learn "The transformer uses d_model=512 and h=8"
        knox learn "QKV with RoPE" --tag attention --tag architecture
    """
    sources_dir_path = Path(sources_dir) if sources_dir else Path(DEFAULT_SOURCES_DIR)

    path = learn(text, sources_dir=sources_dir_path, tags=tag)
    typer.echo(f"Learned fact -> {path.name}")


# ── crystallize ───────────────────────────────────────────────────────────


@app.command(name="crystallize")
def crystallize_cmd(
    sources_dir: SourcesDir = None,
) -> None:
    """Synthesize captured notes into coherent wiki pages.

    Reads all notes from sources/_captured/, groups them by tag, and uses an
    LLM to merge each group into a single well-structured page written to
    sources/_crystallized/. Run 'knox compile' afterwards to incorporate
    the crystallized pages into the wiki.

    Examples:
        knox crystallize
    """
    sources_dir_path = Path(sources_dir) if sources_dir else Path(DEFAULT_SOURCES_DIR)

    try:
        cfg = resolve_config()
        default_model = cfg.get("compile.llm.default_model")
        llm_client = LLMClient(default_model=default_model)
    except Exception:
        llm_client = None

    paths = crystallize(sources_dir_path, llm_client=llm_client)
    if not paths:
        typer.echo("No captured notes found to crystallize.")
        return
    typer.echo(f"Crystallized {len(paths)} page(s) to {sources_dir_path / '_crystallized'}")
    for p in paths:
        typer.echo(f"  {p.name}")


# ── ingest ────────────────────────────────────────────────────────────────


@app.command(name="ingest")
def ingest_cmd(
    target: IngestTarget,
    sources_dir: SourcesDir = None,
) -> None:
    """Download a URL or copy a file into the sources directory.

    URLs are downloaded via HTTP. Local files are copied in.
    After ingestion, run 'knox compile' to process the new source.

    Examples:
        knox ingest https://arxiv.org/pdf/1234.56789
        knox ingest notes/meeting-notes.md
    """
    sources_dir_path = Path(sources_dir) if sources_dir else Path(DEFAULT_SOURCES_DIR)

    result = ingest(target, sources_dir=sources_dir_path)
    if result is None:
        typer.echo(f"Failed to ingest: {target}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Ingested -> {result}")


# ── serve ─────────────────────────────────────────────────────────────────


@app.command(name="serve")
def serve_cmd(
    host: ServeHost = "127.0.0.1",
    port: ServePort = 8765,
    open_browser: ServeOpen = False,
) -> None:
    """Start the KNOX web UI server.

    Opens a browser-based dashboard for browsing, searching, and managing
    the compiled wiki. Binds to localhost by default.

    Examples:
        knox serve
        knox serve --port 8080 --open
    """
    typer.echo(f"Starting KNOX web UI at http://{host}:{port}")
    if host != "127.0.0.1":
        typer.echo("Warning: Binding to non-localhost exposes the server to the network.", err=True)
    serve(host=host, port=port, open_browser=open_browser)


# ── Internal helpers ────────────────────────────────────────────────────


def format_json_output(synthesis: Any, results: list[Any]) -> str:  # noqa: ANN401
    """Format query result as JSON string."""
    output = {
        "answer": synthesis.answer,
        "citations": [
            {
                "article": c["article"],
                "source": c.get("source"),
                "snippet": c["snippet"],
                "score": c["score"],
            }
            for c in synthesis.citations
        ],
    }
    return json.dumps(output, indent=2)


def _try_build_llm_client(config: str | None = None) -> LLMClient | None:
    """Build an LLMClient from config if a default model is set."""
    cfg = resolve_config(config_path=config)
    default_model: str | None = cfg.get("compile.llm.default_model")
    if not default_model:
        return None

    return LLMClient(
        default_model=default_model,
        fallback_chain=list(cfg.get("compile.llm.fallback_chain", [])),
        role_models=dict(cfg.get("compile.llm.role_models", {})),
    )


def _open_vector_store(base_dir: Path, wiki: CompiledWiki) -> ZvecWikiStore | None:
    """Open the zvec sidecar index built during compile, if it exists."""
    db_path = base_dir / ".knox" / "vectors"
    if not db_path.exists():
        return None

    embed_model = wiki.metadata.get("embedding_model", "hash")
    embed_dim = wiki.metadata.get("embedding_dim", 384)
    try:
        return ZvecWikiStore.open(db_path, embedding_model=embed_model or "hash", embedding_dim=embed_dim)
    except Exception as exc:
        logger.warning("Could not open vector store at %s: %s", db_path, exc)
        return None


def _post_compile_actions(
    *,
    wiki: CompiledWiki,
    pipeline_config: PipelineConfig,
    output_dir_path: Path,
    sources_dir_path: Path,
    review: bool | None,
    bypass_review: bool | None,
    provenance: bool | None,
    private: bool | None,
    verify: bool | None,
    synthesize: bool | None,
    index_store: bool | None,
    compile_lint: bool | None,
    embed_model: str | None,
) -> None:
    """Run all post-compilation steps (review, provenance, index, lint, etc.)."""
    base_dir = pipeline_config.resolved_base_dir()
    _write_wiki_cache(wiki, base_dir)

    if review and not bypass_review:
        _send_to_review_queue(wiki, base_dir)
        typer.echo(f"Pages pending review in {base_dir / '.knox' / 'review'}")
    elif bypass_review and review:
        typer.echo("--bypass-review overrides --review; writing pages directly.")

    if provenance:
        _annotate_provenance(wiki, output_dir_path)

    if private:
        _mark_private(wiki, output_dir_path)

    if verify:
        resolved_model = pipeline_config.llm.default_model
        _verify_claims(wiki, output_dir_path, model=resolved_model, verify_budget=pipeline_config.budget.verify_output_max * 4)

    generated = write_index_and_log(wiki, output_dir_path, base_dir=base_dir)
    typer.echo(f"Index: {generated['index']}")
    typer.echo(f"Log:   {generated['log']}")

    if synthesize:
        resolved_model = pipeline_config.llm.default_model
        overview_path = write_overview(
            wiki,
            base_dir,
            model=resolved_model,
            overview_budget=pipeline_config.budget.overview_output_max * 4,
            timeout=pipeline_config.llm.timeout,
        )
        if overview_path:
            typer.echo(f"Overview: {overview_path}")
        else:
            typer.echo("Overview generation skipped (LLM client unavailable).", err=True)

    if index_store:
        search_db_path = base_dir / ".knox" / "search.db"
        build_chunk_index(wiki, search_db_path)
        wiki.metadata["embedding_model"] = embed_model
        wiki.metadata["embedding_dim"] = pipeline_config.embedding.dim
        _cli_build_vector_index(wiki, output_dir_path, pipeline_config.embedding.model, base_dir=base_dir)

    if compile_lint:
        _run_post_compile_lint(wiki, sources_dir_path, json_output=False)

    _write_wiki_cache(wiki, base_dir)


def _build_cli_overrides(
    *,
    sources_dir: str | None = None,
    output_dir: str | None = None,
    target_format: str | None = None,
    chunk_size: int | None = None,
    image_mode: str | None = None,
    mode: str | None = None,
    no_cache: bool | None = None,
    workers: int | None = None,
    force: bool | None = None,
    review: bool | None = None,
    provenance: bool | None = None,
    private: bool | None = None,
    no_privacy: bool | None = None,
    verify: bool | None = None,
    synthesize: bool | None = None,
    index_store: bool | None = None,
    compile_lint: bool | None = None,
    flavor: str | None = None,
    frontmatter: list[str] | None = None,
    ignore: list[str] | None = None,
    llm_default_model: str | None = None,
    vlm_model: str | None = None,
    resolve_relations: bool | None = None,
    relation_confidence_threshold: float | None = None,
    memory_enabled: bool | None = None,
    memory_half_life_days: float | None = None,
    memory_decay_threshold: float | None = None,
    llm_fallback_chain: list[str] | None = None,
    llm_fallback_max_retries: int | None = None,
    llm_role_model: list[str] | None = None,
    kind: str | None = None,
    compiler: str | None = None,
    budget_concepts: int | None = None,
    budget_gen_segment: int | None = None,
    embed_model: str | None = None,
    embed_dim: int | None = None,
    llm_timeout: float | None = None,
) -> dict[str, Any]:
    """Build a flat dotted-key dict of non-None CLI overrides."""
    overrides: dict[str, Any] = {}

    _simple: list[tuple[str, Any]] = [
        ("compile.sources_dir", sources_dir),
        ("compile.output_dir", output_dir),
        ("compile.format", target_format),
        ("compile.chunk_size", chunk_size),
        ("compile.image_mode", image_mode),
        ("compile.mode", mode),
        ("compile.workers", workers),
        ("compile.output.flavor", flavor),
        ("compile.llm.default_model", llm_default_model),
        ("compile.llm.vlm_model", vlm_model),
        ("compile.relation_confidence_threshold", relation_confidence_threshold),
        ("compile.memory_half_life_days", memory_half_life_days),
        ("compile.memory_decay_threshold", memory_decay_threshold),
        ("compile.page_kind", kind),
        ("compile.compiler", compiler),
        ("compile.budget.concepts_per_batch", budget_concepts),
        ("compile.budget.gen_output_max", budget_gen_segment),
        ("compile.embedding.model", embed_model),
        ("compile.embedding.dim", embed_dim),
        ("compile.llm.timeout", llm_timeout),
    ]
    for key, val in _simple:
        if val is not None:
            overrides[key] = val

    _flags: list[tuple[str, bool | None]] = [
        ("compile.no_cache", no_cache),
        ("compile.force", force),
        ("compile.review", review),
        ("compile.provenance", provenance),
        ("compile.private", private),
        ("compile.verify", verify),
        ("compile.synthesize", synthesize),
        ("compile.index", index_store),
        ("compile.lint", compile_lint),
        ("compile.resolve_relations", resolve_relations),
        ("compile.memory_enabled", memory_enabled),
    ]
    for key, val in _flags:
        if val:
            overrides[key] = val

    if no_privacy:
        overrides["trust.privacy_filter"] = False
    if llm_fallback_chain is not None:
        overrides["compile.llm.fallback_chain"] = list(llm_fallback_chain)
    if llm_fallback_max_retries is not None:
        overrides["compile.llm.fallback_max_retries"] = llm_fallback_max_retries
    if llm_role_model is not None:
        parsed: dict[str, str] = {}
        for item in llm_role_model:
            if "=" in item:
                role, model = item.split("=", 1)
                parsed[role.strip()] = model.strip()
        if parsed:
            overrides["compile.llm.role_models"] = parsed
    return overrides


def _cli_build_vector_index(
    wiki: CompiledWiki,
    output_dir_path: Path,
    embed_model: str | None,
    *,
    base_dir: Path | None = None,
) -> None:
    """Thin CLI wrapper — delegates to knox.rag and prints status."""
    if not embed_model:
        typer.echo("Error: --embed-model must be provided when --index is used.", err=True)
        raise typer.Exit(1)

    count = build_vector_index(wiki, output_dir_path, embed_model=embed_model, base_dir=base_dir)
    if count > 0:
        typer.echo(f"Indexed {count} nodes in zvec store for semantic search")
    else:
        typer.echo("Warning: no nodes indexed — vector index skipped", err=True)


# ── New feature helpers (Q20/N3/N8/N4/Q21) ──────────────────────────────


def _print_tier_listing(pipeline: object) -> None:
    """Print the tier listing from a WikiPipeline."""
    if not hasattr(pipeline, "list_tiers"):
        typer.echo("Pipeline does not support tier listing.", err=True)
        return

    typer.echo(json.dumps(pipeline.list_tiers(), indent=2, default=str))


def _send_to_review_queue(wiki: CompiledWiki, base_dir: Path) -> None:
    """Send compiled wiki nodes to the review queue."""
    queue = ReviewQueue(review_dir=base_dir / ".knox" / "review")
    for node in wiki.nodes:
        queue.enqueue(node)


def _annotate_provenance(wiki: CompiledWiki, output_dir_path: Path) -> None:
    """Annotate compiled wiki with claim-level provenance citations."""
    annotated = annotate_wiki_with_provenance(wiki, output_dir_path)
    typer.echo(f"Annotated {len(annotated.nodes)} nodes with provenance")


def _mark_private(wiki: CompiledWiki, output_dir_path: Path) -> None:
    """Mark all wiki pages as private via frontmatter."""
    mark_wiki_private(wiki, output_dir_path)


def _verify_claims(wiki: CompiledWiki, output_dir_path: Path, *, model: str, verify_budget: int = 3000) -> None:
    """Run grounding verification over wiki claims."""
    result = verify_wiki(wiki, output_dir_path, model=model, verify_budget=verify_budget)
    typer.echo(f"Verified {result.verified}/{result.total} claims (avg confidence {result.average_confidence:.2f})")


def _run_post_compile_lint(wiki: CompiledWiki, sources_dir_path: Path, *, json_output: bool = False) -> None:
    """Run lint after compilation and print the report."""
    runner = LintRunner(sources_dir=str(sources_dir_path) if sources_dir_path.exists() else None)
    report = runner.run(wiki)
    typer.echo("\n" + report_to_text(report))


def _resolve_role_models(cfg: Any) -> dict[str, str]:
    """Extract per-role model overrides from resolved config (Q13).

    Reads ``compile.llm.role_models`` from the merged config.
    """
    role_models: dict[str, str] = {}
    raw = cfg.get("compile.llm.role_models", {})
    if isinstance(raw, dict):
        for role, model in raw.items():
            if isinstance(model, str) and model.strip():
                role_models[role] = model.strip()
    return role_models


def _write_wiki_cache(wiki: CompiledWiki, base_dir: Path) -> None:
    """Serialize the compiled wiki to ``.knox/last_wiki.json``.

    This lets standalone ``knox lint`` load the wiki without requiring
    a fresh compile. The cache is overwritten on every successful compile.
    """
    cache_dir = base_dir / ".knox"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "last_wiki.json"

    data = {
        "nodes": [dataclasses.asdict(n) for n in wiki.nodes],
        "edges": list(wiki.edges),
        "toc": list(wiki.toc),
        "quality_score": wiki.quality_score,
        "iteration": wiki.iteration,
        "metadata": dict(wiki.metadata),
    }
    cache_file.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
