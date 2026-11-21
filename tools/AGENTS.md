# DOX — kompend tools

<!-- on entry: loaded when agent enters tools/ -->
<!-- parent: ../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md -->

## Purpose

Throwaway measurement and spike code: harnesses that run the pipeline's own
APIs against a corpus to answer a question the pipeline's behaviour cannot
answer on its own.

## Ownership

- `tools/` — one directory per spike, with the harness, its corpus and its
  result. Nothing here ships.

## Local Contracts

- **No production wiring.** Nothing under `tools/` is imported by `src/` or
  `tests/`, and a tool never changes the pipeline's behaviour. A tool that
  needs a new capability gets it in `src/` first, or the spike answers a
  narrower question.
- **A tool is measurement code.** It may call the pipeline's public APIs
  exactly as the pipeline calls them; it must not reimplement a stage to make
  a number look better.
- **Secrets come from the environment.** No key, token or password in a file
  under `tools/`; read it from the environment at run time.
- **Prompts live in `.md` files** beside the script that uses them, never in
  Python string literals.
- **Never crop.** Content that does not fit one request is windowed, never
  truncated — a harness that crops measures something else than the pipeline.
- **A spike ends in a record, not in code.** The deliverable is the verdict
  (with the numbers) on the bead that asked for it; the harness stays so the
  numbers can be reproduced.

## Work Guidance

- One directory per spike; name it after the question, not the ticket.
- Record the corpus the spike ran on, the model, the budget and the numbers
  in the result — a verdict without its inputs is not reproducible.
- State plainly what a spike could not reproduce. A substituted corpus or a
  different model belongs in the verdict, not in a footnote.
