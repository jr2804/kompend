"""Distill pre-step helpers.

``kompend distill`` and ``kompend compile --distill`` must run the same flow:
build the agent (agentic when available, per-group otherwise) with the same
timeout/retry wiring, distill the note corpus, and report what was written.
The construction lives here so the two commands cannot drift apart.

Message text is returned as plain lines rather than printed, so each command
routes it to the correct console (stdout, or stderr under ``--json-progress``).
"""

from __future__ import annotations

from pathlib import Path

from kompend.agents.distill import DistillAgent
from kompend.agents.distill_agentic import AgenticDistillAgent, make_agentic_distill_agent
from kompend.capture.distill import distill
from kompend.core.config import ResolvedConfig
from kompend.core.paths import CAPTURED_SUBDIR, DISTILLED_SUBDIR, LEARNED_SUBDIR

#: Capture zones that distillation reads, in the order it merges them.
PENDING_NOTE_SUBDIRS: tuple[str, ...] = (CAPTURED_SUBDIR, LEARNED_SUBDIR)


def count_pending_notes(sources_dir: Path) -> int:
    """Count notes awaiting distillation.

    Mirrors what :func:`kompend.capture.distill.distill` reads: top-level
    ``*.md`` files in the capture zones (``sources/_captured/`` and
    ``sources/_learned/``), not nested files.
    """
    return sum(len(list((sources_dir / subdir).glob("*.md"))) for subdir in PENDING_NOTE_SUBDIRS)


def pending_notes_message(count: int) -> str:
    """Nudge shown when notes await distillation and the run did not request it."""
    zones = " + ".join(f"sources/{subdir}/" for subdir in PENDING_NOTE_SUBDIRS)
    return f"{count} note(s) in {zones} are not distilled — run 'kompend distill', or pass --distill."


def run_distill(sources_dir: Path, cfg: ResolvedConfig, *, archive: bool = True) -> list[Path]:
    """Distill the note corpus and return the pages written."""
    return distill(sources_dir, distill_agent=build_distill_agent(cfg), archive=archive)


def build_distill_agent(cfg: ResolvedConfig) -> DistillAgent | AgenticDistillAgent | None:
    """Build the distill agent from config.

    Prefers the agentic (one session, tool-driven) agent and falls back to the
    per-group ``DistillAgent``; returns ``None`` when no model is configured or
    neither agent can be constructed, which distills deterministically.
    """
    model = cfg.kompend.compile.llm.default_model
    if not model:
        return None
    try:
        return make_agentic_distill_agent(
            model,
            timeout=float(cfg.kompend.compile.llm.timeout),
            llm=cfg.kompend.compile.llm,
            budget=cfg.kompend.compile.budget,
        )
    except Exception:
        try:
            return DistillAgent(
                model,
                timeout=float(cfg.kompend.compile.llm.timeout),
                llm=cfg.kompend.compile.llm,
                budget=cfg.kompend.compile.budget,
            )
        except Exception:
            return None


def distill_result_lines(paths: list[Path], sources_dir: Path) -> list[str]:
    """Report lines for a completed distill run (or for an empty corpus)."""
    if not paths:
        return ["No captured notes found to distill."]
    return [
        f"Distilled {len(paths)} page(s) to {sources_dir / DISTILLED_SUBDIR}",
        *(f"  {path.name}" for path in paths),
    ]
