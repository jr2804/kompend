"""Capture, learn, distill, and write commands."""

from __future__ import annotations

import typer

from knox.capture.capture import capture
from knox.capture.distill import distill
from knox.capture.learn import learn
from knox.cli._shared import build_cli_overrides, resolve_project_paths
from knox.cli._sources import resolve_files
from knox.cli.app import app
from knox.cli.args import CaptureText, Frontmatter, LearnText, Tags
from knox.core.config import resolve_config
from knox.llm.client import LLMClient
from knox.logging import get_console, get_err_console
from knox.output.markdown import MarkdownWriter
from knox.output.models import Document


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
    get_console().print(f"Captured {len(paths)} item(s) to {sources_dir / '_captured'}")
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
    get_console().print(f"Distilled {len(paths)} page(s) to {sources_dir / '_distilled'}")
    for p in paths:
        get_console().print(f"  {p.name}")


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
    _sources_dir, output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

    fm: dict[str, str] = {}
    if frontmatter:
        for kv in frontmatter:
            if "=" in kv:
                k, v = kv.split("=", 1)
                fm[k.strip()] = v.strip()

    body_parts: list[str] = []
    if source:
        source_files = resolve_files(*source, sources_dir=_sources_dir)
        for sf in source_files:
            try:
                content = sf.read_text(encoding="utf-8", errors="replace")[:2000]
                body_parts.append(f"From {sf.name}:\n\n{content}\n")
            except Exception as exc:
                get_err_console().print(f"Warning: could not read {sf}: {exc}")

    body = "\n\n".join(body_parts) if body_parts else "_(Content pending)_"

    writer = MarkdownWriter()
    doc = Document(title=page_title, body=body, frontmatter=fm)
    rendered = writer.renderer.render(doc)

    output_file = output_dir / f"{page_title.lower().replace(' ', '-')}.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(rendered, encoding="utf-8")
    get_console().print(f"Written: {output_file}")
