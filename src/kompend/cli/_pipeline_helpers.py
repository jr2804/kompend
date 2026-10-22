"""Internal helpers for pipeline CLI commands.

Private API — external callers should use ``kompend.cli.cmds.pipeline``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

import typer

from kompend.agents.doctor import DoctorAgent, DoctorDeps
from kompend.cli._post_compile import (  # noqa: F401  # re-exported for backward compat
    PostCompileOptions,
    annotate_provenance,
    cli_build_vector_index,
    mark_private,
    post_compile_actions,
    print_compile_summary,
    run_overview_synthesis,
    run_post_compile_lint,
    send_to_review_queue,
    verify_claims,
    write_wiki_cache,
)
from kompend.cli._progress import PipelineProgress
from kompend.cli._sources import download_urls, ensure_in_sources, resolve_files
from kompend.cli.args import (
    Ignore,
    Recursive,
    UrlFile,
)
from kompend.core.pipeline.orchestrator import WikiPipeline
from kompend.logging import get_console, get_err_console, get_logger, set_verbosity

logger = get_logger(__name__)


# ── progress callbacks ─────────────────────────────────────────────────


def make_progress_callback(tracker: PipelineProgress) -> Callable[[str, int, int, str], None]:  # noqa: PLR0915
    """Build a progress callback that translates pipeline events into Rich updates.

    The overall bar resets per phase. Each phase has a flat handler keyed by
    name in the dispatch table — no nested phase if/elif chain.
    - **parse**: bar tracks files (0 → N)
    - **summary**: bar tracks per-tree LLM summarization (0 → N trees)
    - **resolve**: bar resets to track chunks (0 → M)
    - **compile** / **output**: bar becomes indeterminate
    """

    def _parse(current: int, total: int, detail: str) -> None:
        if current == 0:
            tracker.start()
            tracker.reset_for_phase(total, "Parsing…")
        elif detail == "done":
            tracker.update_progress(total, f"Parsed {total} file(s)")
        else:
            tracker.update_progress(current, f"[{current}/{total}] {detail}")

    def _summary(current: int, total: int, detail: str) -> None:
        if detail == "starting":
            tracker.reset_for_phase(total, f"Summarizing sections ({total} source(s) to summarize)…")
        elif detail == "done":
            if total:
                tracker.update_progress(total, f"Summarized {total} source(s)")
            else:
                tracker.message("No sources to summarize")
        else:
            tracker.update_progress(current, f"[{current}/{total}] {detail}")

    def _index(current: int, total: int, detail: str) -> None:
        if current == 0:
            tracker.start()
            tracker.message("Downloading sources…")
        else:
            tracker.message("Download complete")

    def _resolve(current: int, total: int, detail: str) -> None:
        if current == 0:
            if detail == "vlm":
                tracker.reset_for_phase(total, f"Captioning images ({total})…")
            else:
                tracker.reset_for_phase(total, f"Resolving chunks ({total} total)…")
        elif detail == "done":
            tracker.update_progress(total, "Resolve complete")
        elif detail == "captioning":
            tracker.update_progress(current, f"Captioning image [{current}/{total}]")
        else:
            tracker.update_progress(current, f"Resolving [{current}/{total}]")

    def _start_done(start_msg: str, done_msg: str) -> Callable[[int, int, str], None]:
        """output: indeterminate bar driven by start/done events."""

        def _cb(current: int, total: int, detail: str) -> None:
            if detail == "done":
                tracker.message(done_msg)
            elif detail == "starting":
                tracker.reset_for_phase(None, start_msg)

        return _cb

    def _compile() -> Callable[[int, int, str], None]:
        """compile: determinate bar counting real work units."""
        state: dict[str, int | None] = {"total": None}

        def _cb(current: int, total: int, detail: str) -> None:
            if detail == "done":
                if state["total"] is not None:
                    tracker.update_progress(state["total"], "Compile complete")
                else:
                    tracker.message("Compile complete")
            elif detail == "starting":
                tracker.reset_for_phase(None, "Compiling wiki…")
                state["total"] = None
            elif total and total != state["total"]:
                state["total"] = total
                tracker.reset_for_phase(total, "Compiling wiki…")
                tracker.update_progress(current, detail)
            else:
                tracker.update_progress(current, detail)

        return _cb

    handlers: dict[str, Callable[[int, int, str], None]] = {
        "parse": _parse,
        "summary": _summary,
        "index": _index,
        "resolve": _resolve,
        "compile": _compile(),
        "output": _start_done("Writing output…", "Output complete"),
    }

    def _cb(phase: str, current: int, total: int, detail: str) -> None:
        handler = handlers.get(phase)
        if handler is not None:
            handler(current, total, detail)

    return _cb


def resolve_compile_inputs(
    files: list[str],
    online_urls: list[str],
    url_file: UrlFile,
    sources_dir: Path,
    recursive: Recursive,
    ignore: Ignore,
    *,
    in_place: bool = False,
    undetected: bool = True,
) -> list[Path]:
    """Resolve file arguments and URLs into a deduplicated input file list.

    When *in_place* is ``True``, local files outside *sources_dir* are
    kept at their original location instead of being copied.
    """
    if url_file is not None:
        for raw_line in url_file.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#"):
                online_urls.append(stripped)

    resolved = resolve_files(*files, recursive=recursive, ignore_patterns=ignore)

    sources_dir.mkdir(parents=True, exist_ok=True)
    local_files = ensure_in_sources(resolved, sources_dir, in_place=in_place)
    downloaded = download_urls(online_urls, sources_dir, base_dir=sources_dir.parent, undetected=undetected)

    all_input_files = list(dict.fromkeys(local_files + downloaded))
    if not all_input_files:
        all_input_files = resolve_files(
            str(sources_dir),
            recursive=recursive,
            ignore_patterns=ignore,
        )
    return all_input_files


def print_tier_listing(pipeline: WikiPipeline) -> None:
    """Print the tier listing from a WikiPipeline."""
    if not hasattr(pipeline, "list_tiers"):
        get_err_console().print("Pipeline does not support tier listing.")
        return

    get_console().print(json.dumps(pipeline.list_tiers(), indent=2, default=str))
    # ── JSON progress callback ──────────────────────────────────────────────


def make_json_progress_callback() -> Callable[[str, int, int, str], None]:
    """Build a progress callback that writes JSONL events to stdout.

    Each event is one JSON line::

        {"event": "progress", "phase": "parse", "current": 5, "total": 40, "message": "[5/40] file.pdf"}
        {"event": "done", "duration": 123.4, "message": "Compiled wiki written to ..."}

    The callback also sets INFO verbosity so pipeline log messages are visible.
    """
    set_verbosity(logging.INFO)
    start_time = time.monotonic()

    def _reset() -> None:
        nonlocal start_time
        start_time = time.monotonic()

    def _json_cb(phase: str, current: int, total: int, detail: str) -> None:
        if phase == "start":
            _reset()
        elapsed = round(time.monotonic() - start_time, 2)
        payload = {
            "event": "progress",
            "phase": phase,
            "current": current,
            "total": total,
            "message": detail,
            "elapsed": elapsed,
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    return _json_cb


def emit_json_done(duration: float, message: str) -> None:
    r"""Write the final ``{\"event\":\"done\", ...}`` line to stdout."""
    payload = {
        "event": "done",
        "duration": round(duration, 2),
        "message": message,
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def check_llm_availability(pipeline: WikiPipeline) -> None:
    """Pre-flight LLM connectivity check — fail fast before processing documents."""
    model = pipeline.config.llm.default_model
    if not model:
        return
    try:
        agent = DoctorAgent(
            model,
            timeout=pipeline.config.llm.timeout,
            llm=pipeline.config.llm,
        )
        agent.run(DoctorDeps(ping="ok"))
    except Exception as exc:
        get_err_console().print("[red]LLM not available — not starting compilation.[/red]")
        get_err_console().print(f"  {exc}")
        raise typer.Exit(1) from exc
