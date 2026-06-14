"""Capture, learn, crystallize, and write commands."""

from __future__ import annotations

import typer

from knox.capture.capture import capture
from knox.capture.crystallize import crystallize
from knox.capture.learn import learn
from knox.cli._shared import build_cli_overrides, resolve_project_paths
from knox.cli._sources import resolve_files
from knox.cli.app import app
from knox.cli.args import CaptureText, Frontmatter, LearnTags, LearnText
from knox.core.config import resolve_config
from knox.llm.client import LLMClient
from knox.output.markdown import MarkdownWriter
from knox.output.models import Document


@app.command(name="capture")
def capture_cmd(
    ctx: typer.Context,
    text: CaptureText,
) -> None:
    """Extract knowledge items from text and save as source files.

    Uses LLM to extract decisions, discoveries, corrections, and references
    from conversation text or notes. Results are written to sources/_captured/.

    Examples:
        knox capture "We decided to use QKV projections with RoPE"
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    sources_dir_path, _output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

    try:
        default_model = cfg.get("compile.llm.default_model")
        llm_client = LLMClient(default_model=default_model)
    except Exception:
        llm_client = None

    paths = capture(text, sources_dir=sources_dir_path, llm_client=llm_client)
    typer.echo(f"Captured {len(paths)} item(s) to {sources_dir_path / '_captured'}")
    for p in paths:
        typer.echo(f"  {p.name}")


@app.command(name="learn")
def learn_cmd(
    ctx: typer.Context,
    text: LearnText,
    tag: LearnTags = None,
) -> None:
    """Store a single fact directly as a source file.

    Unlike 'capture', this stores the text verbatim without LLM extraction.
    Results are written to sources/_captured/.

    Examples:
        knox learn "The transformer uses d_model=512 and h=8"
        knox learn "QKV with RoPE" --tag attention --tag architecture
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    sources_dir_path, _output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

    path = learn(text, sources_dir=sources_dir_path, tags=tag)
    typer.echo(f"Learned fact -> {path.name}")


@app.command(name="crystallize")
def crystallize_cmd(
    ctx: typer.Context,
) -> None:
    """Synthesize captured notes into coherent wiki pages.

    Reads all notes from sources/_captured/, groups them by tag, and uses an
    LLM to merge each group into a single well-structured page written to
    sources/_crystallized/. Run 'knox compile' afterwards to incorporate
    the crystallized pages into the wiki.

    Examples:
        knox crystallize
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    sources_dir_path, _output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

    try:
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


@app.command(name="write")
def write_cmd(
    ctx: typer.Context,
    page_title: str = typer.Argument(..., help="Title of the page to create."),
    source: list[str] = typer.Option([], "--source", "-s", help="Source file(s) to include (repeatable)."),  # noqa: B008
    frontmatter: Frontmatter = None,
) -> None:
    """Create a one-shot wiki page from a title and optional sources.

    Example:
        knox write "Transformer architecture" --source attention.pdf --source bert.pdf --frontmatter language=en
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    _sources_dir, output_dir_path, _cache_dir, _state_dir = resolve_project_paths(cfg)

    fm: dict[str, str] = {}
    if frontmatter:
        for kv in frontmatter:
            if "=" in kv:
                k, v = kv.split("=", 1)
                fm[k.strip()] = v.strip()

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

    writer = MarkdownWriter()
    doc = Document(title=page_title, body=body, frontmatter=fm)
    rendered = writer.renderer.render(doc)

    output_path = output_dir_path / f"{page_title.lower().replace(' ', '-')}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    typer.echo(f"Written: {output_path}")
