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
| ``ingest.py`` | Capture-to-ingest pipeline adapter (``recursive`` defaults to on, callers pass the resolved ``paths.recursive``) |
| ``_utils.py`` | Capture-specific utilities (titles, slugs, sources index) |
| ``kinds.py`` | The capture vocabulary — ``SourceZone`` (zone, source dir, archive dir) and ``CaptureKind`` (extraction kind written to frontmatter) |

## Rules

- Import: ``from kompend.capture.capture import capture``,
  ``from kompend.capture.distill import distill``.
- Distill wiring (agent selection, pending-note counting, result lines) used by
  the CLI lives in ``kompend.cli.workflow.distill`` — this package exposes the
  flow, not the CLI presentation.
- ``_distill_groups`` resumes: a session that exhausts its request budget
  keeps the pages it saved, and a further session runs for the tag groups still
  without one. It stops when a round saves nothing new; groups still missing
  then merge deterministically.
- An agent failure is not a fallback trigger. `capture` keeps one raw note only
  on `LLMOutputError` (and keeps the caller's tags on it); `_distill_groups`
  writes the pages the failed round had already
  recorded, names the groups left without one, and re-raises. Any other
  exception is a bug and escapes. Budget exhaustion is the one resumable case:
  it takes the resume path above, distinguished down the cause chain by
  `_is_budget_exhaustion`.
- Capture agents (``CaptureAgent``, ``DistillAgent``) live in
  ``kompend.agents.capture`` — this package contains orchestration and
  non-agent logic only.
- Every page written to ``_captured/`` / ``_learned/`` / ``_distilled/``
  escapes prose pipes via ``core.text.pipes.escape_prose_pipes`` before the
  write (generation contract — see `docs/architecture/decisions.md`).
- Frontmatter values are vocabulary members from ``kinds.py``, never bare
  strings: a zone name is simultaneously the source subdirectory, the archive
  subdirectory and the ``source`` provenance value, and ``CaptureKind`` is
  pinned against the prompt template by
  ``tests/unit/test_capture_vocabulary.py``. Metadata keys come from
  ``core.models.enums.MetadataField``; pass members through
  ``output.frontmatter`` rather than formatting YAML here — it stringifies
  them, so the rendered file carries the wire value instead of a Python tag.
