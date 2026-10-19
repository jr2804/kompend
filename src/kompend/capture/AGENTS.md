# DOX — kompend.capture

<!-- on entry: loaded when agent enters src/kompend/capture -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Document capture and distillation pipeline. Captures source content from URLs,
files, and stdin; distills raw content into structured knowledge; learns
patterns across captures.

## Ownership

- ``src/kompend/capture/`` — capture, distill, ingest, learn logic

## Architecture

| Module | Contents |
|--------|----------|
| ``capture.py`` | ``capture`` — LLM-extract durable items from text into ``_captured/`` |
| ``learn.py`` | ``learn`` — store a verbatim fact into ``_learned/`` |
| ``distill.py`` | ``distill`` — merge ``_captured/`` + ``_learned/`` notes by tag into ``_distilled/`` pages |
| ``consolidate.py`` | ``consolidate`` — archive near-duplicate notes across the note zones |
| ``ingest.py`` | Capture-to-ingest pipeline adapter |
| ``_utils.py`` | Capture-specific utilities (titles, slugs, sources index) |

## Rules

- Import: ``from kompend.capture.capture import capture``,
  ``from kompend.capture.distill import distill``.
- Distill wiring (agent selection, pending-note counting, result lines) used by
  the CLI lives in ``kompend.cli._distill_helpers`` — this package exposes the
  flow, not the CLI presentation.
- ``_distill_agentic`` resumes: a session that exhausts its request budget
  keeps the pages it saved, and a further session runs for the tag groups still
  without one. It stops when a round saves nothing new; groups still missing
  then merge deterministically.
- Capture agents (``CaptureAgent``, ``DistillAgent``) live in
  ``kompend.agents.capture`` — this package contains orchestration and
  non-agent logic only.
