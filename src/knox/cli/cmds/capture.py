"""Capture, learn, distill, and write commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import litellm
import typer

from knox.capture.capture import capture
from knox.capture.distill import distill
from knox.capture.learn import learn
from knox.cli._shared import build_cli_overrides, resolve_project_paths
from knox.cli._sources import resolve_files
from knox.cli.app import app
from knox.cli.args import CaptureText, Frontmatter, LearnText, Tags
from knox.core.config import resolve_config
from knox.core.config.keys import COMPILE_BUDGET_CAPTURE_SOURCE_CHAR_MAX, DISPLAY_RELATED_TITLES
from knox.core.models import MetadataField, WikiField
from knox.core.paths import CAPTURED_SUBDIR, DISTILLED_SUBDIR, VaultPaths
from knox.core.pipeline.ingest import _extract_text_from_cached, parse_cache_glob
from knox.core.utils.time import iso_local
from knox.llm.client import LLMClient, _strip_reasoning
from knox.logging import get_console, get_err_console, get_logger
from knox.output.markdown import MarkdownWriter
from knox.output.models import Document

logger = get_logger(__name__)


@app.command(name="capture")
def capture_cmd(
    ctx: typer.Context,
    text: CaptureText,
    tag: Tags = None,
) -> None:
    """Extract knowledge items from text via LLM; save each as a source file.

    Sends the text to an LLM, which pulls out decisions, discoveries,
    corrections, references, and notes — one file per extracted item.
    This is the LLM-extraction counterpart to 'learn' (which stores text
    verbatim without an LLM). Results are written to sources/_captured/.

    Examples:
        knox capture "We decided to use QKV projections with RoPE"
        knox capture "Discussion about layer norm placement" --tag architecture --tag training
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    sources_dir, _output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

    try:
        llm_client = LLMClient.from_config(cfg)
    except Exception:
        llm_client = None

    paths = capture(text, sources_dir=sources_dir, llm_client=llm_client, tags=tag)
    get_console().print(f"Captured {len(paths)} item(s) to {sources_dir / CAPTURED_SUBDIR}")
    for p in paths:
        get_console().print(f"  {p.name}")


@app.command(name="learn")
def learn_cmd(
    ctx: typer.Context,
    text: LearnText,
    tag: Tags = None,
) -> None:
    """Store a single fact verbatim as a source file — no LLM involved.

    The text is written exactly as given (with optional tags) to
    sources/_captured/. Use this when you already know the fact you want
    recorded; use 'capture' when you want an LLM to pull items out of
    free-form text.

    Examples:
        knox learn "The transformer uses d_model=512 and h=8"
        knox learn "QKV with RoPE" --tag attention --tag architecture
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    sources_dir, _output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

    path = learn(text, sources_dir=sources_dir, tags=tag)
    get_console().print(f"Learned fact -> {path.name}")


@app.command(name="distill")
def distill_cmd(
    ctx: typer.Context,
) -> None:
    """Merge captured notes into coherent wiki pages via LLM.

    Reads all notes from sources/_captured/, groups them by tag, and uses an
    LLM to merge each group into a single well-structured page written to
    sources/_distilled/. Run 'knox compile' afterwards to incorporate
    the distilled pages into the wiki.

    Examples:
        knox distill
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    sources_dir, _output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

    try:
        llm_client = LLMClient.from_config(cfg)
    except Exception:
        llm_client = None

    paths = distill(sources_dir, llm_client=llm_client)
    if not paths:
        get_console().print("No captured notes found to distill.")
        return
    get_console().print(f"Distilled {len(paths)} page(s) to {sources_dir / DISTILLED_SUBDIR}")
    for p in paths:
        get_console().print(f"  {p.name}")


@app.command(name="write")
def write_cmd(
    ctx: typer.Context,
    page_title: str = typer.Argument(..., help="Title of the page to create."),
    source: list[str] = typer.Option([], "--source", "-s", help="Source file(s) to include (repeatable)."),  # noqa: B008
    frontmatter: Frontmatter = None,
) -> None:
    """Create a wiki page synthesising content from one or more sources.

    Sources are resolved through the parse cache (.knox/cache/) so PDF/DOCX
    files are read as clean text, not raw bytes. When an LLM is configured,
    the page is written as continuous prose with a References section; otherwise
    source texts are concatenated.

    The page is written to wiki/write/ with default frontmatter.

    Example::

        knox write "Transformer architecture" --source attention.pdf --source bert.pdf --frontmatter language=en
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    _sources_dir, output_dir, cache_dir, _state_dir = resolve_project_paths(cfg)
    base_dir = output_dir.parent

    # ── Resolve sources through the parse cache ──────────────────────────
    source_texts: list[tuple[str, str]] = []  # (stem, clean_text)
    if source:
        source_files = resolve_files(*source, sources_dir=_sources_dir)
        for sf in source_files:
            stem = sf.stem
            # Parse cache is now content-discriminated: glob for any entry.
            matches = sorted(VaultPaths(base_dir).cache_dir.glob(parse_cache_glob(stem)))
            cached = matches[0] if matches else None
            try:
                text = _extract_text_from_cached(cached) if cached and cached.exists() else sf.read_text(encoding="utf-8", errors="replace")
                source_texts.append((stem, text.strip()))
            except Exception as exc:
                get_err_console().print(f"Warning: could not read {sf}: {exc}")

    # ── Load related wiki nodes from the last compile ────────────────────
    related_titles = _load_related_titles(base_dir, source_texts)
    related_titles = related_titles[: cfg.get(DISPLAY_RELATED_TITLES, 15)]

    # ── Build body ────────────────────────────────────────────────────────
    fm: dict[str, Any] = {}
    if frontmatter:
        for kv in frontmatter:
            if "=" in kv:
                k, v = kv.split("=", 1)
                fm[k.strip()] = v.strip()

    compiled_at = iso_local()
    fm.setdefault("title", page_title)
    fm.setdefault("kind", "page")
    fm.setdefault("compiled_at", compiled_at)
    if source_texts:
        fm.setdefault("sources", [s for s, _ in source_texts])

    if not source_texts:
        body = "_(Content pending — run with --source to populate)_"
    else:
        # Try LLM synthesis; fall back to concatenation.
        llm_model = cfg.get("compile.llm.default_model", "")
        body = _synthesize_write_page(
            page_title=page_title,
            source_texts=source_texts,
            related_titles=related_titles,
            model=llm_model,
            timeout=float(cfg.get("compile.llm.timeout", 120)),
            source_char_max=cfg.get(COMPILE_BUDGET_CAPTURE_SOURCE_CHAR_MAX, 4000),
        )

    # ── Write to wiki/write/ ─────────────────────────────────────────────
    slug = page_title.lower().replace(" ", "-")
    write_dir = output_dir / "write"
    write_dir.mkdir(parents=True, exist_ok=True)
    output_file = write_dir / f"{slug}.md"

    writer = MarkdownWriter()
    doc = Document(title=page_title, body=body, frontmatter=fm, node_type="page")
    rendered = writer.renderer.render(doc)
    output_file.write_text(rendered, encoding="utf-8")
    get_console().print(f"Written: {output_file}")


def _load_related_titles(
    base_dir: Path,
    source_texts: list[tuple[str, str]],
) -> list[str]:
    """Return titles of wiki nodes that reference any of the given sources."""
    related: list[str] = []
    wiki_cache = VaultPaths(base_dir).wiki_cache_path
    if not wiki_cache.exists() or not source_texts:
        return related
    try:
        data: dict[str, Any] = json.loads(wiki_cache.read_text(encoding="utf-8"))
        source_stems = {s for s, _ in source_texts}
        for node in data.get(WikiField.NODES, []):
            prov = node.get("metadata", {}).get(MetadataField.PROVENANCE) or []
            node_sources = {Path(str(p)).stem for p in prov if p}
            if node_sources & source_stems:
                related.append(str(node.get("title", "")))
    except Exception as exc:
        logger.debug("Failed to load last_wiki.json for related nodes: %s", exc)
    return related


def _synthesize_write_page(
    *,
    page_title: str,
    source_texts: list[tuple[str, str]],
    related_titles: list[str],
    model: str,
    timeout: float,
    source_char_max: int = 4000,
) -> str:
    """Return continuous prose for the write page.

    When an LLM is available, asks it to synthesize the sources into a
    coherent essay with a References section. Falls back to a structured
    text join when the LLM is unavailable or returns nothing.
    """
    references_section = _build_references_section(source_texts, related_titles)

    if not model:
        raise RuntimeError("No LLM model configured for write-page synthesis")

    source_block = "\n\n".join(f"### {stem}\n\n{text[:source_char_max]}" for stem, text in source_texts)
    prompt = (
        f"Write a comprehensive wiki page titled '{page_title}'.\n"
        f"Synthesise the sources below into coherent, continuous prose.\n"
        f"If the sources are too unrelated to synthesise, state that clearly.\n"
        f"Do NOT include a raw list of sources; weave citations naturally.\n\n"
        f"{source_block}"
    )
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=timeout,
        )
        content = response.choices[0].message.content or ""
        content = _strip_reasoning(content)
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
