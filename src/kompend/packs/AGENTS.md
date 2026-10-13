# DOX — kompend.packs

<!-- on entry: loaded when agent enters src/kompend/packs -->
<!-- parent: ../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Contribution packs — a named bundle of pipeline-role prompts for a domain
(academic, technical, default). A pack is data, not code: its prompts live in a
TOML manifest, so a domain expert can add or edit one without touching Python.

Surfaced today through `kompend pack list` / `kompend pack show` and the MCP
`packs_list` / `packs_show` tools. The registry also exposes `prompt_for(role,
pack_names)` for pipeline consumption; nothing in the compilation pipeline calls
it yet.

## Ownership

- `src/kompend/packs/` — the `Pack` model, `PackRegistry`, TOML loader
- `src/kompend/packs/builtins/<name>/pack.toml` — shipped packs

## Local Contracts

### Pack layout

    src/kompend/packs/builtins/<name>/__init__.py   # package marker
    src/kompend/packs/builtins/<name>/pack.toml     # manifest + prompts

Built-in packs are loaded by dotted package path
(`load_pack_toml("kompend.packs.builtins.default")`), so the directory is a
package. `_register_default()` always registers `default`.

### `pack.toml` schema

    [pack]
    name = "academic"
    description = "Prompts for academic research papers"
    version = "1.0.0"
    dependencies = ["default"]

    [roles.compile]
    prompt = "You compile knowledge from ..."

    [roles.query]
    prompt = "Answer the user's question based on ..."

- `[pack]` — `name`, `description`, `version`, `dependencies` (pack names).
- `[roles.<role>] prompt` — the only accepted role key. A role entry without
  `prompt` is ignored. Roles come from `KNOWN_ROLES` in `base.py` (`compile`,
  `resolve`, `enrich`, `query`, `lint.*`) — the enum-like source of truth; add a
  role there before using it in a manifest.
- Prompts are TOML strings: this is the sanctioned home for prompt text, not a
  "no text in code" violation (see `src/kompend/AGENTS.md`).

### Resolution

`PackRegistry.prompt_for(role, pack_names)` searches the active packs in order
and the **first pack defining the role wins**; `default` is always active and
placed first, so an explicit pack without the role falls through to it. Unknown
pack names are logged and skipped, never fatal.

## Work Guidance

- New built-in pack: add the package + `pack.toml`, and only touch
  `_register_default()` if the pack should load without being named.
- User packs live outside the source tree and are not this package's concern.
- `Pack.metadata` is currently always empty — there is no user-facing metadata
  field in the manifest yet.
