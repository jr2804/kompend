# DOX — knox.capture

<!-- on entry: loaded when agent enters src/knox/capture -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Document capture and distillation pipeline. Captures source content from URLs,
files, and stdin; distills raw content into structured knowledge; learns
patterns across captures.

## Ownership

- ``src/knox/capture/`` — capture, distill, ingest, learn logic

## Architecture

| Module | Contents |
|--------|----------|
| ``capture.py`` | ``capture_source`` — fetch and normalise input sources |
| ``distill.py`` | ``distill_to_chunks`` — content distillation into structured data |
| ``ingest.py`` | Capture-to-ingest pipeline adapter |
| ``learn.py`` | ``learn_from_captures`` — pattern extraction across captures |
| ``_utils.py`` | Capture-specific utilities |

## Rules

- Import: ``from knox.capture.capture import capture_source``,
  ``from knox.capture.distill import distill_to_chunks``.
- Capture agents (``CaptureAgent``, ``DistillAgent``) live in
  ``knox.agents.capture`` — this package contains orchestration and
  non-agent logic only.
