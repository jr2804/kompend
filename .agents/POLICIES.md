# AGENTS.md — Policies

Boundary rules, change constraints, and response format. Always applies.

## General Policies

These rules apply to any project using the AGENTS.md framework.

### Priorities

If rules conflict, lower-numbered priority wins:

1. Correctness
2. Evidence
3. Safety
4. Minimal changes
5. Consistency
6. Performance

### Boundaries

- NEVER fabricate paths, commits, APIs, config keys, env vars, test results,
  or benchmark numbers. If you don't know, say so.
- NEVER guess at a command name, flag, or file path. Read the source or run
  `--help` first.
- NEVER delete or move files without explicit user instruction.
- NEVER run destructive commands (`rm -rf`, `git reset --hard`, `git push
  --force`) without explicit user confirmation.
- NEVER add secrets, API keys, or tokens to any file. Use env vars.
- NEVER create temporary files in the project root. Use the project's
  temp directory or `tempfile.TemporaryDirectory()`.

### Change Constraints

- Prefer minimal, surgical edits over large rewrites.
- When editing, preserve existing formatting, indentation, and style.
- Do not introduce new dependencies without explicit user instruction.
- Do not refactor unrelated code while fixing a bug or adding a feature.
- If a change requires updating tests, update only the tests directly affected.

### Completion

Before declaring completion, confirm:

- The change solves the stated problem
- Relevant validation ran or gaps are stated
- No known unintended side effects were introduced
- No secrets were added or exposed

### Response Format

Be concise and specific. No filler, intros, sycophantic openers ("Great
question!"), or restated requirements. Answer direct questions directly.

For review, debugging, or analysis outputs, use:
findings with references, conclusion, approach.
Mention caveats and unverified risks.

### Closeout

1. Run existing verification when relevant
2. Report any docs intentionally left unchanged and why

### Extension

| Who | When | What |
|-----|------|------|
| Project maintainer | Adopting new boundary rules | Boundaries |
| Project maintainer | Changing workflow philosophy | Change Constraints |
| Any contributor | Noticing a missing policy | Add accordingly |

---

## Project-specific Policies — KNOX

### Boundaries

- NEVER modify `.gitignore`, `pyproject.toml`, `ruff.toml`, or CI config
  without explicit user instruction.
- NEVER suppress `PLC0415` (inline imports), `ANN001` (missing return type),
  or `ANN202` (missing return type on public method). These are hard errors —
  no `# noqa` permitted. Refactor the code instead.
- Use `./tmp` (gitignored) or `tempfile.TemporaryDirectory()` for temp files.

### Implementation Plans

- Implementation plans, refactoring plans, and design docs go in
  `.agents/plans/` — NOT under version control (gitignored).
- NEVER commit implementation plans to git. They are working artifacts,
  not shipped documentation.
- Shipped docs (user-facing guides, API references) go in `docs/` and
  ARE committed.

### User Preferences

- Concise, direct answers with no filler or intros
- Use `uv run` for all Python execution, never direct `python` calls
- English-only for code, docs, and commits
- Conventional commits format (`feat:`, `fix:`, `chore:`, `docs:`, etc.)
