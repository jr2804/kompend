# DOX — kompend.lint

<!-- on entry: loaded when agent enters src/kompend/lint -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Linting and remediation for compiled wiki pages. Structural rules run
programmatically; LLM-powered rules use ``LintRuleAgent`` for semantic checks.
Remediation uses ``RemediationAgent`` to auto-fix findings.

## Ownership

- ``src/kompend/lint/`` — lint rules, runner, remediation, gap analysis, history

## Architecture

| Module | Contents |
|--------|----------|
| ``base.py`` | ``Severity``, ``LintRuleOutcome``, ``LintResult``, ``BaseLintRule`` ABC |
| ``runner.py`` | ``LintRunner``, ``LintReport`` — orchestrates rule execution; ``LintReportPayload``/``LintResultPayload`` are the validated wire shapes (dumped to JSON, loaded from history) |
| ``structural/`` | The programmatic rules, one module per family (see **Structural families** below); ``structural/__init__.py`` is the registry — ``STRUCTURAL_RULES``, ``RULE_BY_NAME`` and the re-exported rule classes, so ``kompend.lint.structural`` stays the one import path |
| ``flow.py`` | ``run_full_lint_flow``/``_run_pass`` (one pass over a rule set, fixes persisted), ``persist_lint_fixes``, ``_make_lint_agent``, and ``CHECK_STEP_RULES`` — the deterministic rules of the ``check`` step |
| ``llm_rules.py`` | ``LLM_RULE_BY_NAME`` — LLM-powered semantic rules |
| ``remediation.py`` | ``RemediationAutoFixer`` — auto-fix for structural + LLM findings |
| ``gap_analysis.py`` | Gap recommendations from lint reports; broken-wikilink findings stratified into ``alias`` vs ``missing`` resolutions |
| ``_remediation_gaps.py`` | Gap-specific remediation logic; generated pages route through ``MarkdownWriter`` (→ ``concepts/`` + index regen) |
| ``history.py`` | Nested pydantic records (``LintEntry``/``LintRunOptions``/``LintLlmMeta``/``LintConfigSnapshot``/``LintWikiMeta``) + ``save_lint_entry``/``load_lint_history``/``prune_lint_history``/``report_from_dict`` and ``persist_history`` (one run → one entry) — lint run history in the ``lint_runs`` table of ``.kompend/kompend.db``. ``LintEntry.from_row()``/``to_columns()`` are the only owners of the table's 7-column shape; adding a column means adding a field and a ``_COLUMNS`` name |

## Structural families

``src/kompend/lint/structural/`` — one module per rule family.

| Module | Rules |
|--------|-------|
| ``links.py`` | ``broken-wikilinks``, ``kompend-reference`` + the wikilink vocabulary |
| ``images.py`` | ``image-embed`` + the image vocabulary |
| ``structure.py`` | ``orphan-page``, ``duplicate-title``, ``empty-page``, ``heading-skip`` |
| ``metadata.py`` | ``missing-frontmatter`` |
| ``sources.py`` | ``stale-page``, ``source-not-found`` |
| ``attachments.py`` | ``orphan-attachments``, ``unreferenced-attachments`` + the two public helpers |
| ``references.py`` | ``duplicate-references`` |
| ``markdown.py`` | ``rumdl`` (+ ``RumdlFinding``, the rumdl wire shape) |

## Rules

- **An LLM rule call is long-context and slow.** `lint.llm_timeout` (default
  300 s) governs it, separately from `compile.llm.timeout`. The retry policy is
  not lint's own: `LlmLintRule._call_agent` makes one attempt and the agent base
  retries a transient failure (`compile.llm.transient_retries`). A failed call
  raises — it never returns `None`.
- **A failed unit becomes a SKIP result, not a missing one.** Every rule opens
  `UnitFailureLog(stage=self.name)` (vocabulary in `agents/AGENTS.md`), calls
  `attempted()` per unit, records each exception, emits one `SKIP` result whose
  message names the kind (`No result (empty_answer): …`), then `check()`s
  against `compile.llm.batch_failure_threshold_ratio` / `_min` before returning.
  Dropping a page silently shrinks the denominator and *raises* the health
  score — never do it.
- **One scorer.** `flow._rich_report` IS `runner._build_report`
  (severity-weighted: ERROR=3, WARNING=2, INFO=1, `100*max(0, 1-penalty/max)`).
  A second formula anywhere means two different numbers for one run.
- **The terminal shows a bounded report; the file shows everything.** A
  lint run over a big vault yields hundreds of findings, so the console prints
  the verdict plus (at most) one line per failing rule and points at
  ``.kompend/lint-log/<stamp>.md``. Never print `report_to_text` from a
  pipeline path: it is for `--verbose-lint` and `kompend lint show`.
- **Remediation writes pages, rules write JSON.** The remediation step uses
  `AgenticRemediationAgent` (free-text page writer); the lint-rule agent's
  `BaseModel | str` output cannot produce a page and fails validation. A single
  failed page is a skipped action, never an exception that ends `lint all`.
- **One definition per question.** Which rules count as the deterministic
  ``check`` step (``flow.CHECK_STEP_RULES``) and how a finished run becomes a
  history entry (``history.persist_history``) live here, never in a caller: the
  CLI commands and the MCP tools both consume them, and a copy in either layer
  drifts (the MCP tools used to import both from the CLI command module).
- **One home for the structural registry.** ``structural/__init__.py`` owns
  ``STRUCTURAL_RULES`` and ``RULE_BY_NAME``. Each family module names the rules
  it contributes (``IMAGE_RULES``, ``LINK_RULES``, ``METADATA_RULES``,
  ``PAGE_STRUCTURE_RULES``, ``SOURCE_RULES``, ``ATTACHMENT_RULES``) and the
  registry concatenates them — a family gains a rule without the registry
  changing. ``rumdl`` and ``duplicate-references`` are config-dependent
  factories: they are registered by name in ``RULE_BY_NAME`` only and
  instantiated by ``LintRunner._resolve_rules``.
- **Rule names and detail keys are contracts.** Rule names are dispatched by
  string outside this package (``gap_analysis.py``, ``runner.py``, ``flow.py``),
  and ``broken-wikilinks`` findings carry the ``alias_of``/``target`` detail
  keys that ``compilation/flows/generation`` reads to learn aliases. Renaming
  a rule or a key degrades gap analysis and alias learning silently.
- **One liveness answer.** ``live_attachment_slugs`` in
  ``core/models/attachments.py`` is the only place that decides whether an
  attachment slug belongs to a source: ``live`` for an ordinary source,
  ``protected`` for one under ``sources/_captured/`` and friends, ``dead``
  otherwise. Both destructive rules and auto-adoption call it — three copies
  used to disagree, and ``orphan-attachments`` deleted the figures of a
  captured note while ``unreferenced-attachments`` kept them. A protected slug
  is neither an orphan nor prunable, and it is not adopted.
- **The attachment rules delete files.** ``orphan-attachments`` and
  ``unreferenced-attachments`` are ``destructive`` and run only behind the
  runner's ``fix_destructive`` opt-in. Their ``fix()`` reads ``_last_sources_dir``,
  which ``check()`` set on the same instance — the runner must call ``check``
  and ``fix`` on one instance, or the fix silently no-ops.
- **A page link outranks the slug.** ``orphan-attachments`` keeps every file a
  compiled page links to, whatever its directory is named. A vault carries slug
  forms from older kompend versions (the slug grew a path prefix when sources
  moved into subdirectories) and the pages still link those directories — the
  source is alive, only the directory name is history. Deleting such a file
  breaks the wiki (371 files, 1391 links in the p863 vault). The kept files
  are reported as one PASS summary; only files no page links to stay in the
  FAIL set.
- **A link target is parsed, not pattern-matched.** ``_link_targets`` reads a
  page with markdown-it-py (``commonmark`` + ``html``) and takes the
  destination from image/link tokens plus ``src``/``href`` in raw HTML. Three
  hand-rolled misses each deleted a page's own figure: a link title
  (``![f](x.png "Figure 1")``) kept the title inside the path, an
  angle-bracket destination kept its brackets, and a reference definition was
  invisible. A destination may hold balanced parentheses — a figure named
  ``..._Assessment_(POLQA),_Standard__0b467ca.jpg`` — and the parser resolves
  that too, so no depth counter is needed. Fenced and inline code yield no
  link token: a documentation example never protects a file. A reference
  definition protects its target even when no link uses it — the definition is
  page content an author wrote, not an optimisation to drop.
- **An unreadable page cancels the destructive decision.** A page that cannot be
  decoded may link any attachment, so a run with one gets a SKIP naming the
  pages (``details["unreadable_pages"]``) and deletes nothing, rather than the
  old silent ``continue`` that read "cannot read" as "links nothing".
- **A code example is not a page link.** Both directions of the rule read only
  prose: a fenced block or an inline span that mentions
  ``attachments/<slug>/x.png`` neither protects a file the wiki dropped nor
  condemns one it shows. The same reader is the sole source for both
  destructive rules, so they cannot disagree about what a page links.
- Import: ``from kompend.lint.runner import LintRunner``,
  ``from kompend.lint.base import LintResult``.
- Structural rules are fast and run always; LLM rules are gated by config.
- ``LlmLintRule`` takes ``data_format`` — the `compile.llm.data_format` the
  rule's structured prompt payloads are encoded in. `LintRunner` owns the value
  (``data_format=``) and hands it to every LLM rule through its kwargs; a rule
  must encode its payload with ``kompend.llm.format.encode_payload`` rather
  than ``json.dumps`` so the format setting has one source.
- ``RemediationAutoFixer`` calls ``RemediationAgent`` for LLM auto-fixes.
- ``LintRuleAgent`` and ``RemediationAgent`` live in ``kompend.agents.lint``.
- ``history.py`` I/O takes a vault path and resolves ``kompend.db`` via
  ``core.store.vault_db_path`` — it no longer writes ``lint_history.jsonl``.
