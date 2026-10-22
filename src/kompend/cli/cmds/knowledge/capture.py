"""Capture, learn, distill, and write commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from kompend.agents.capture import CaptureAgent
from kompend.agents.types import WriteDeps
from kompend.agents.write import make_agentic_write_agent
from kompend.capture.capture import capture
from kompend.capture.consolidate import consolidate
from kompend.capture.learn import learn
from kompend.cli._distill_helpers import distill_result_lines, run_distill
from kompend.cli._shared import build_cli_overrides, vault_paths_from_ctx
from kompend.cli._sources import resolve_files
from kompend.cli.app import app
from kompend.cli.args import (
    WRITE_SOURCES_DEFAULT,
    CaptureText,
    ConsolidateDryRun,
    ConsolidateNoArchive,
    DistillNoArchive,
    Frontmatter,
    LearnText,
    SourceAttrs,
    SourceTags,
    WritePageTitle,
    WriteSourceFiles,
)
from kompend.core import source_metadata
from kompend.core.config import resolve_config
from kompend.core.config._configs import BudgetConfig, LLMConfig
from kompend.core.models import MetadataField, load_cached_wiki
from kompend.core.models.enums import NodeType, OkfField
from kompend.core.models.metadata import meta_str_list
from kompend.core.paths import CAPTURED_SUBDIR
from kompend.core.text.encoding import read_text_auto
from kompend.core.text.matching import SemanticMatcher
from kompend.core.text.slug import slugify
from kompend.core.utils.time import iso_local
from kompend.llm.format import strip_reasoning
from kompend.logging import get_console, get_err_console, get_logger
from kompend.output.markdown import MarkdownWriter
from kompend.output.models import Document
from kompend.rag.zvec_store import make_provider

logger = get_logger(__name__)


@app.command(name="capture")
def capture_cmd(
    ctx: typer.Context,
    text: CaptureText,
    tag: SourceTags = None,
    attr: SourceAttrs = None,
) -> None:
    """Extract knowledge items from text via LLM; save each as a source file.

    Sends the text to an LLM, which pulls out decisions, discoveries,
    corrections, references, and notes — one file per extracted item.
    This is the LLM-extraction counterpart to 'learn' (which stores text
    verbatim without an LLM). Results are written to sources/_captured/.

    Examples:
        kompend capture "We decided to use QKV projections with RoPE"
        kompend capture "Discussion about layer norm placement" --tags architecture --tags training
        kompend capture "..." --attr version=2 --attr status=confirmed
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    sources_dir = vp.sources_dir

    default_model = cfg.kompend.compile.llm.default_model
    capture_agent = None
    if default_model:
        try:
            capture_agent = CaptureAgent(
                default_model,
                timeout=float(cfg.kompend.compile.llm.timeout),
                llm=cfg.kompend.compile.llm,
                budget=cfg.kompend.compile.budget,
            )
        except Exception:
            capture_agent = None

    paths = capture(
        text,
        sources_dir=sources_dir,
        capture_agent=capture_agent,
        tags=tag,
        attributes=_parse_attrs(attr),
    )
    if not paths:
        get_console().print("[yellow]Nothing captured — the input contained no extractable knowledge items.[/]")
        return
    get_console().print(f"Captured {len(paths)} item(s) to {sources_dir / CAPTURED_SUBDIR}")
    for p in paths:
        get_console().print(f"  {p.name}")


@app.command(name="learn")
def learn_cmd(
    ctx: typer.Context,
    text: LearnText,
    tag: SourceTags = None,
    attr: SourceAttrs = None,
) -> None:
    """Store a single fact verbatim as a source file — no LLM involved.

    The text is written exactly as given (with optional tags/attributes) to
    sources/_learned/. Use this when you already know the fact you want
    recorded; use 'capture' when you want an LLM to pull items out of
    free-form text.

    Examples:
        kompend learn "The transformer uses d_model=512 and h=8"
        kompend learn "QKV with RoPE" --tags attention --tags architecture
        kompend learn "..." --attr version=2
    """
    vp = vault_paths_from_ctx(ctx)
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())

    path = learn(
        text,
        sources_dir=vp.sources_dir,
        tags=tag,
        attributes=_parse_attrs(attr),
    )
    get_console().print(f"Learned fact -> {path.name}")


def _parse_attrs(attr: list[str] | None) -> dict[str, str]:
    """Parse ``KEY=VALUE`` --attr flags into a dict."""
    out: dict[str, str] = {}
    for item in attr or []:
        if "=" not in item:
            raise typer.BadParameter(f"Use KEY=VALUE format for attributes, got: '{item}'")
        key, _, value = item.partition("=")
        out[key.strip()] = value.strip()
    return out


@app.command(name="consolidate")
def consolidate_cmd(
    ctx: typer.Context,
    dry_run: ConsolidateDryRun = False,
    no_archive: ConsolidateNoArchive = False,
) -> None:
    """Find and archive redundant captured/learned/distilled notes.

    Scans sources/_captured/, sources/_learned/ and sources/_distilled/ for
    near-duplicate notes (rapidfuzz token similarity, plus embeddings when
    an embedding model is configured). For each duplicate group the newest
    or longest note is kept; the rest are archived to
    sources/_archive/<kind>/ by default.

    Examples:
        kompend consolidate --dry-run
        kompend consolidate
        kompend consolidate --no-archive
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())

    matcher: SemanticMatcher | None = None
    embed_model = cfg.kompend.compile.embedding.model
    if embed_model and embed_model != "hash":
        try:
            matcher = SemanticMatcher(embedder=make_provider(embed_model))
        except Exception as exc:
            logger.warning("consolidate: embedding tier unavailable (%s); string similarity only", exc)

    groups = consolidate(vp.sources_dir, matcher=matcher, dry_run=dry_run, archive=not no_archive)
    if not groups:
        get_console().print("No duplicate notes found.")
        return
    for group in groups:
        keeper = group.keeper.source_file.name if group.keeper.source_file else "?"
        get_console().print(f"Group: keeper={keeper}")
        for dup in group.duplicates:
            name = dup.source_file.name if dup.source_file else "?"
            action = "would archive" if (dry_run or no_archive) else "archived"
            get_console().print(f"  {action}: {name}")


@app.command(name="distill")
def distill_cmd(
    ctx: typer.Context,
    no_archive: DistillNoArchive = False,
) -> None:
    """Merge captured notes into coherent wiki pages via LLM.

    Reads all notes from sources/_captured/ and sources/_learned/, groups
    them by tag, and uses an LLM to merge each group into a single
    well-structured page written to sources/_distilled/. Existing pages for
    the same group are updated in place (no duplicate timestamped files).
    By default the distilled source notes are archived to
    sources/_archive/<kind>/; pass --no-archive to keep them.
    Run 'kompend compile' afterwards to incorporate the distilled pages into
    the wiki.

    Examples:
        kompend distill
        kompend distill --no-archive
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())

    paths = run_distill(vp.sources_dir, cfg, archive=not no_archive)
    for line in distill_result_lines(paths, vp.sources_dir):
        get_console().print(line)


@app.command(name="write")
def write_cmd(
    ctx: typer.Context,
    page_title: WritePageTitle,
    source: WriteSourceFiles = WRITE_SOURCES_DEFAULT,
    frontmatter: Frontmatter = None,
) -> None:
    """Create a wiki page synthesising content from one or more sources.

    Sources are resolved through the parse cache (.kompend/cache/) so PDF/DOCX
    files are read as clean text, not raw bytes. When an LLM is configured,
    the page is written as continuous prose with a References section; otherwise
    source texts are concatenated.

    The page is written to wiki/write/ with default frontmatter.

    Example::

        kompend write "Transformer architecture" --source attention.pdf --source bert.pdf --frontmatter language=en
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    base_dir = vp.base_dir

    # ── Resolve sources through the parse cache ──────────────────────────
    source_texts: list[tuple[str, str]] = []  # (stem, clean_text)
    if source:
        source_files = resolve_files(*source, sources_dir=vp.sources_dir)
        for sf in source_files:
            stem = sf.stem
            try:
                cached = source_metadata.cached_text(vp.cache_dir, sf.name)
                text = cached if cached is not None else read_text_auto(sf)
                source_texts.append((stem, text.strip()))
            except Exception as exc:
                get_err_console().print(f"Warning: could not read {sf}: {exc}")

    # ── Load related wiki nodes from the last compile ────────────────────
    related_titles = _load_related_titles(base_dir, source_texts)
    related_titles = related_titles[: cfg.kompend.display.related_titles]

    # ── Build body ────────────────────────────────────────────────────────
    fm: dict[str, Any] = {}
    if frontmatter:
        for kv in frontmatter:
            if "=" in kv:
                k, v = kv.split("=", 1)
                fm[k.strip()] = v.strip()

    compiled_at = iso_local()
    fm.setdefault(OkfField.TITLE, page_title)
    fm.setdefault(OkfField.KIND, NodeType.PAGE.value)
    fm.setdefault(OkfField.COMPILED_AT, compiled_at)
    if source_texts:
        fm.setdefault(OkfField.SOURCES, [s for s, _ in source_texts])

    if not source_texts:
        body = "_(Content pending — run with --source to populate)_"
    else:
        # Try LLM synthesis; fall back to concatenation.
        llm_model = cfg.kompend.compile.llm.default_model
        body = _synthesize_write_page(
            page_title=page_title,
            source_texts=source_texts,
            related_titles=related_titles,
            model=llm_model,
            timeout=float(cfg.kompend.compile.llm.timeout),
            source_char_max=cfg.kompend.compile.budget.capture_source_char_max,
            llm=cfg.kompend.compile.llm,
            budget=cfg.kompend.compile.budget,
        )

    # ── Write to wiki/write/ ─────────────────────────────────────────────
    slug = slugify(page_title)
    write_dir = vp.output_dir / "write"
    write_dir.mkdir(parents=True, exist_ok=True)
    output_file = write_dir / f"{slug}.md"

    writer = MarkdownWriter(description_max_chars=cfg.kompend.display.description_max_chars)
    doc = Document(title=page_title, body=body, frontmatter=fm, node_type=NodeType.PAGE)
    rendered = writer.renderer.render(doc)
    output_file.write_text(rendered, encoding="utf-8")
    get_console().print(f"Written: {output_file}")


def _load_related_titles(
    base_dir: Path,
    source_texts: list[tuple[str, str]],
) -> list[str]:
    """Return titles of wiki nodes that reference any of the given sources."""
    related: list[str] = []
    if not source_texts:
        return related
    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        return related
    source_stems = {s for s, _ in source_texts}
    for node in wiki.nodes:
        prov = meta_str_list((node.metadata or {}).get(MetadataField.PROVENANCE))
        node_sources = {Path(p).stem for p in prov if p}
        if node_sources & source_stems:
            related.append(str(node.title))
    return related


def _synthesize_write_page(
    *,
    page_title: str,
    source_texts: list[tuple[str, str]],
    related_titles: list[str],
    model: str | None,
    timeout: float,
    source_char_max: int,
    llm: LLMConfig | None = None,
    budget: BudgetConfig | None = None,
) -> str:
    """Return continuous prose for the write page.

    When an LLM is available, asks it to synthesize the sources into a
    coherent essay with a References section. Falls back to a structured
    text join when the LLM is unavailable or returns nothing.

    The sources are passed in full — ``source_char_max`` sizes one
    ``read_source()`` window, and the agent pages through a source it
    wants to read completely. No source text is dropped.
    """
    references_section = _build_references_section(source_texts, related_titles)

    if not model:
        raise RuntimeError("No LLM model configured for write-page synthesis")

    agent = make_agentic_write_agent(
        model_string=model,
        timeout=timeout,
        llm=llm,
        budget=budget,
    )
    deps = WriteDeps(
        page_title=page_title,
        source_texts=source_texts,
        related_titles=related_titles,
        source_id="write",
        timeout=timeout,
        source_char_max=source_char_max,
    )
    try:
        content = agent.run(deps)
        content = strip_reasoning(content) if isinstance(content, str) else content
        if content.strip():
            return content.strip() + "\n\n" + references_section
        raise RuntimeError("LLM returned empty content for write page")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"LLM synthesis failed for write page: {exc}") from exc


def _build_references_section(
    source_texts: list[tuple[str, str]],
    related_titles: list[str],
) -> str:
    """Build a References / Related section at the end of the page."""
    lines = ["## References\n"]
    for stem, _ in source_texts:
        lines.append(f"- {stem}")
    if related_titles:
        lines.append("\n## Related\n")
        for title in related_titles:
            lines.append(f"- [[{title}]]")
    return "\n".join(lines)
