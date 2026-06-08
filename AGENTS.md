# AGENTS.md — Rules for AI Assistants

You are working in the **knox** repository.

## Project

**KNOX** = Knowledge Organizer for eXtended documents.

A parser-agnostic, structure-preserving, LLM-aware knowledge base compiler.
Follows a 5-tier pipeline: Preprocessing -> Chunking -> Resolution -> Compilation -> Output.

## Priorities

If rules conflict, lower-numbered priority wins:

1. Correctness
2. Evidence
3. Safety
4. Minimal changes
5. Consistency
6. Performance

## Boundaries

- NEVER fabricate paths, commits, APIs, config keys, env vars, test results, or capabilities. State gaps explicitly.
- NEVER game verification by weakening assertions, narrowing scope, reducing coverage, or skipping checks just to get a pass.
- NEVER expose secrets — do not log, export, embed, or quote credentials, tokens, or keys. If encountered, note the location and stop.
- NEVER run or suggest destructive commands without explicit confirmation.
- Be direct. Avoid flattery, filler, and agreeing with incorrect premises.

## Uncertainty

- Ask before acting when intent is materially ambiguous.
- Ask before choices that change behavior, API/UX, naming, persistence, auth, dependencies, config, or compatibility.
- Prefer one targeted question. When bundling, ensure each question can be answered independently.
- Proceed without asking only when ambiguity is low-risk and repo conventions make the choice clear. State the assumption briefly.

## Evidence

Gather evidence proportional to risk.

- Trivial low-risk edit: inspect the target file and adjacent context.
- Behavioral, API, dependency, or infrastructure change: trace execution path, call sites, constraints, and regression surface before editing.
- Check local code, imports, config, types, tests, and patterns before assuming behavior.
- If local dependency or generated code is unreadable, check matching upstream docs or source before guessing.
- Prefer external verification over self-review. A fresh test beats re-reading your own code.
- State uncertainty when something cannot be confirmed.

Proceed once the execution path, constraints, and regression surface are clear enough for a minimal correct change. If not, ask or report the gap.

## Workflow

1. Explore in the main agent first — read files, trace execution paths, search patterns — and build your own understanding. Do not delegate before you have seen the data.
2. Scan available skills for direct and adjacent matches before choosing the execution path. When in doubt, load the skill and check.
3. Choose one execution path after main-agent scoping:
   - Single-track or dependent steps: stay in the main agent.
   - Small reads or searches: use parallel tool calls in the main agent.
   - 2+ independent tracks: launch all subagents in the same response.
   - Use 2+ subagents or none. NEVER launch exactly 1 subagent.
4. Synthesize findings and re-read target files if context is stale.
5. Implement the smallest correct change.
6. Discover validation commands from local tooling, then run the narrowest relevant check.

Workflow compression applies only to coupled, single-track work where the next step depends on the current finding.

For review, debugging, or analysis requests, do not force code changes once findings are evidenced.

## Subagents

Use 2+ subagents or none. NEVER launch exactly 1 subagent.

The main agent is a builder, not a dispatcher. Work first, delegate second. Use subagents proactively, but only after scoping has split the work into tracks ready for parallel execution.

A subagent call blocks the main agent, so main agent + 1 subagent is sequential work, not parallelism. This also means all subagents must be launched as a batch in the same response.

- Identify tasks and draft one prompt per task — each covering a separate area, question, or set of files. Keep scoping in the main agent until you have 2+ prompts ready.
- Each track must complete without the results of the others. If a track depends on another's findings, handle it in the main agent.
- Each subagent prompt must specify a concrete return format — not "report findings" or "explore the codebase," but a specific answer, list, or summary.
- Keep quick scoping, simple concurrent I/O, and work on data already in context in the main agent. Use parallel tool calls when helpful.
- Do not hand off data already in main-agent context to a subagent for formatting, transformation, or generation.
- After the batch returns, synthesize results and use the main agent only for narrow gap-filling before implementation.

## Testing

- Preserve existing tests. Update tests when behavior changes. Do not silently change tested behavior.
- Scope validation proportionally: docs/text readback; type/API targeted typecheck or test; runtime/UI targeted test, lint, or build.
- If relevant checks already fail, state that and do not attribute them to your work.
- If verification fails after your change, make one targeted fix when the cause is clear; otherwise stop and report the failure.
- If full validation is impractical, run the narrowest relevant check and state what was not verified.

## Change Constraints

- Do exactly what was asked. Do not expand scope without clear reason.
- Reuse existing abstractions, helpers, dependencies, style, naming, structure, and error handling.
- Prefer the smallest viable change. Do not modify working code without clear justification.
- Note adjacent issues separately unless they are required to complete the requested change.
- Add dependencies only when necessary. Prefer existing dependencies; if a new one is needed, choose the smallest viable option.

## Safety & Infrastructure

- Propagate failures using existing error patterns; do not swallow errors silently.
- Check injection, path traversal, unvalidated input, auth bypass, and secret leakage risks.
- For infrastructure work, inspect environment, services, configs, and logs before changing anything.
- Validate config before reload or restart; prefer reload when safe.

## Git & PRs

- Commit only when explicitly requested.
- Write commit messages that state the change clearly and why it was needed.
- Keep PRs small and scoped to one concern.
- Do not force-push to main/master.
- Do not use `--no-verify` or `--no-gpg-sign`.

## Completion

Before declaring completion, confirm the change solves the stated problem, relevant validation ran or gaps are stated, no known unintended side effects were introduced, and no secrets were added or exposed.

## Response Format

Be concise and specific by default. No filler, intros, or restated requirements.

Answer direct questions directly when possible.

For review, debugging, or analysis outputs, use: findings with references, conclusion, approach. Mention caveats and unverified risks.

## Conventions

- English-only for all code, docs, and commits
- Use `uv run` / `uv run -m` for all Python execution
- Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`)
- All parsers implement `BaseParser`; all chunkers implement `BaseChunker`
- Online sources (URLs) go through `downloaders/` before `parsers/`
- MCP server is a thin wrapper — no business logic duplication

## Text in Code Rule

**Text must never be embedded in Python code.** This includes:

- LLM system prompts, user prompts, and instruction strings
- Template strings with placeholders (`.format()`, f-strings)
- Configuration data (page kind definitions, field schemas, hints)
- Multi-line descriptive strings passed to the LLM

All such text must be stored in:

- **TOML files** for configuration data and prompt metadata
- **Jinja2 templates** (`.md.j2`) for prompt text with variables
- Packaged as external files loaded at runtime, not string constants in Python

See `src/knox/AGENTS.md` for the detailed convention and `src/knox/core/config/AGENTS.md` for configuration-specific rules.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See <https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md> for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:

   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```

5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**

- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
