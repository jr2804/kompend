# A hermetic test suite, and a CI job that runs it (2026-11-05)

## Why the suite was not trustworthy

Eight unit tests failed on a clean checkout, and every one of them was a
property of the *machine*, not of the code:

- **Five** needed `KOMP_LLM_DEFAULT_MODEL` in the shell: `PipelineConfig.from_resolved`
  refuses an empty `compile.llm.default_model`, and the value reaches the
  config through the env alias on `LLMConfig.default_model`.
- **Three** failed *because* that variable was set: for `default_model` the env
  alias outranks the project TOML, while those tests assert the file wins.

So no shell made the suite green — unset, five failures; set, three different
ones. The documented T2 command (`tests/AGENTS.md`) sets only
`KOMP_LLM_TRANSIENT_BACKOFF=0`, so the suite failed out of the box for everyone.

Two more failures were path-dependent rather than env-dependent: the CLI
assertions matched bare filenames in *wrapped* console output. Rich hard-wraps
at the terminal width and can break a filename mid-token (`".gitig\nnore"`), and
the wrap point moves with the length of the tmp path — which moves with the
pytest run counter. The eighth asserted Windows path semantics on POSIX
(`pathlib` reads a backslash as a separator only on Windows; on POSIX it is a
legal filename character).

And nothing ran the suite automatically: no workflow, no pre-commit hook. That
is why the zvec SIGILL — which made the whole suite uncollectable — was
invisible to CI, and why these eight sat unnoticed.

## What landed

- `tests/conftest.py` strips ambient `KOMP_*` variables before every test
  (autouse `_isolate_env`). A test that needs one sets it itself with
  `monkeypatch`; `KOMP_LLM_TRANSIENT_BACKOFF` survives because the T2 command
  sets it deliberately.
- `tests/helpers/cli.py` renders CLI output on a wide canvas (`COLUMNS`), so a
  substring assertion tests the message and not the wrap.
- `tests/unit/test_rel_to_vault.py` asserts the platform's own reading of a
  backslash path — the helper was never wrong, the test's premise was.
- `.github/workflows/tests.yml` runs the unit tier on every push and pull
  request, with the T2 command from `tests/AGENTS.md`.

The suite is now green in any shell: 3375 passed, 48 skipped, 0 failed.

## What was deliberately not changed

The product code stays as it is. `discover_context_window` lets a malformed
model id raise `ValueError` while `agents/base.py` catches it — but a model id
without a `provider/model` prefix is a configuration error, and failing fast
with that message beats degrading silently. The test now avoids the path
instead: `test_build_pipeline_accepts_cancel` patches
`discover_context_window` out, exactly as `tests/unit/test_agentic_wiring.py`
already does, because the token-limit lookup is a catalogue download and has
nothing to do with the cancel wiring.

The release workflow still cuts a release on every push to `main`; the test job
runs beside it and does not gate it. Gating a release on the suite is a
release-policy decision, not a test-hygiene one.
