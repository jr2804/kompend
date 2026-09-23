# DOX — knox.core.trust

<!-- on entry: loaded when agent enters src/knox/core/trust -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Actor identity and trust/privacy concerns. Resolves the current OS user
for provenance metadata, controlled by `trust.user_tracking` config.

## Modules

| Module | Contents |
|--------|----------|
| `actor.py` | `get_current_user`, `resolve_actor` |

## Rules

- Import: `from knox.core.trust.actor import resolve_actor`.
- Tracking is opt-out via `trust.user_tracking` config key.
- When disabled, `resolve_actor` returns `""` — no identity recorded.
- Privacy redaction (`src/knox/trust.py`) is separate — it lives at the
  top level, not in this submodule. Its `mark_wiki_private` records the
  private-page index in the `private_pages` table of `.knox/knox.db`
  (snapshot semantics: the previous index is replaced).
