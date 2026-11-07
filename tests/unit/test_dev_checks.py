"""The new guards must pass, and they must actually catch their bug class."""

from __future__ import annotations

import ast
import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined

from kompend.core.cache import store

ROOT = Path(__file__).resolve().parents[2]
CHECKS = ROOT / "scripts" / "checks"
GUARDS = ("check_enum_cache.py", "check_config_literals.py", "check_cli_surface.py")


#: How to empty each fact a gate judges by, one entry per ``require_facts``
#: caller (see ``test_every_require_facts_caller_has_an_empty_fact_case``).
#: ``(module, attribute, receiver, empty-value)`` — the receiver is ``None`` when
#: the attribute lives on the module itself and a nested object (e.g.
#: ``check_enum_cache.store``) otherwise.
_EMPTY_FACT_CASES: list[tuple[str, str, str | None, object]] = [
    ("check_config_literals", "_config_defaults", None, {}),
    ("check_enum_cache", "_CACHE_DATACLASSES", "store", {}),
    ("check_mcp_parity", "get_cli_commands", None, []),
]


@pytest.mark.parametrize("script", GUARDS)
def test_guard_passes_on_the_current_tree(script: str) -> None:
    result = _run(script)

    assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"


def test_all_gates_runner_lists_the_guards() -> None:
    """`all.py` must invoke the guards - a gate nobody runs is not a gate."""
    source = (CHECKS / "all.py").read_text(encoding="utf-8")

    for name in ("enum_cache", "config_literals", "cli_surface", "dox", "mcp_parity"):
        assert name in source


def test_enum_cache_guard_knows_edge_type() -> None:
    """The EdgeType regression: the enum must be decodable from a payload."""
    assert "EdgeType" in store._CACHE_ENUMS


def test_pip_install_guard_rejects_a_comment() -> None:
    """The guard must fail on an install instruction that bypasses uv."""
    sample_dir = ROOT / "tmp" / "check-antipatterns-pip"
    sample_dir.mkdir(parents=True, exist_ok=True)
    forbidden = " ".join(("uv", "pip", "install"))
    (sample_dir / "sample.py").write_text(f"VALUE = 1  # install with {forbidden} foo\n", encoding="utf-8")
    try:
        result = _run("check_antipatterns.py", "--src", str(sample_dir))

        assert result.returncode == 1
        assert "PIP-INSTALL" in result.stdout
    finally:
        shutil.rmtree(sample_dir, ignore_errors=True)


def test_noqa_scan_finds_every_spelling_with_a_real_column() -> None:
    """``#noqa`` / ``#  noqa`` are suppressions too, and the column is real."""
    module = _antipatterns()

    for line in ("x = 1  # noqa: PLC0415", "x = 1  #noqa: PLC0415", "x = 1  #  noqa: PLC0415"):
        _, comments = module._split_comments(line)
        found = module.scan_noqa_bans(Path("a.py"), comments)

        assert [v.line for v in found] == [1], line
        assert found[0].col == 8, f"column must point at the '#', not -1 (got {found[0].col})"


def test_noqa_scan_ignores_prose_and_strings() -> None:
    """A docstring quoting the rule is documentation, not a suppression."""
    module = _antipatterns()
    docstring = 'def f() -> None:\n    """Use `# noqa: PLC0415` as a last resort."""\n'

    _, comments = module._split_comments(docstring)

    assert module.scan_noqa_bans(Path("a.py"), comments) == []


def test_noqa_codes_split_on_whitespace_as_ruff_does() -> None:
    """Ruff accepts space-separated codes, so the ban must see them."""
    module = _antipatterns()

    assert module._collect_noqa_codes("# noqa: PLC0415 B008") == {"PLC0415", "B008"}
    assert module._collect_noqa_codes("# noqa: PLC0415, B008") == {"PLC0415", "B008"}


def test_magic_crops_honours_a_bare_noqa_and_ignores_string_data() -> None:
    """A crop in a string is data; a bare ``# noqa`` is a valid opt-out."""
    module = _antipatterns()

    def crops(source: str) -> int:
        code, comments = module._split_comments(source)
        return len(module.scan_magic_crops(Path("a.py"), code, comments))

    assert crops("title = text[:120]  # noqa") == 0
    assert crops('x = "text[:120]"') == 0
    assert crops('x = f"{value[:3]}"') == 1, "an f-string expression is still code"
    assert crops("title = text[:120]") == 1


def test_import_error_guard_ignores_an_import_inside_the_handler() -> None:
    """Only an import in the guarded body makes it an import guard."""
    module = _antipatterns()

    handler_only = ast.parse("try:\n    risky()\nexcept ImportError:\n    import json\n")
    real_guard = ast.parse("try:\n    import optional_pkg\nexcept ImportError:\n    optional_pkg = None\n")

    assert module.scan_import_error_guards(Path("a.py"), handler_only) == []
    assert len(module.scan_import_error_guards(Path("a.py"), real_guard)) == 1


def test_a_bare_string_reason_code_is_caught_but_the_member_is_not(tmp_path: Path) -> None:
    """The reason-code vocabulary is an enum; a bare string desyncs the log from it.

    The position is the signal, not the value: a typo'd reason ("timeouts")
    is the case worth catching, and a string that happens to match a member
    value is no more correct than one that does not — so the rule requires
    the member form rather than comparing against the enum.
    """
    module = _antipatterns()
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def f(exc):\n"
        '    logger.error("x", extra={"reason_code": "timeout"})\n'
        '    logger.error("x", reason_code="timeouts")\n'
        '    logger.error("x", extra={"reason_code": ReasonCode.TIMEOUT})\n'
        '    logger.error("x", extra={"reason_code": reason_code_for(exc)})\n',
        encoding="utf-8",
    )

    findings = module.scan_file(probe)

    assert [f.rule for f in findings] == ["REASON-CODE-LITERAL", "REASON-CODE-LITERAL"]
    assert [f.line for f in findings] == [2, 3]
    assert '"timeout"' in findings[0].message
    assert "ReasonCode member" in findings[0].message


def test_the_reason_code_rule_can_be_waived_with_a_reason(tmp_path: Path) -> None:
    """A waiver states why the site is legitimate — it is not a silent opt-out."""
    module = _antipatterns()
    probe = tmp_path / "probe.py"
    probe.write_text(
        'logger.error("x", extra={"reason_code": "custom"})  # nocheck: reason-code — vendor payload\n',
        encoding="utf-8",
    )

    assert module.scan_file(probe) == []


def test_require_facts_documents_its_precondition() -> None:
    """Zero has three meanings; the helper can only see the count.

    "declares none" (an honest empty list) and "not applicable" (a virtual
    workspace root with no [project] table) must NOT be funnelled through
    require_facts, or a silent pass becomes a loud false alarm.
    """
    sys.path.insert(0, str(CHECKS))
    try:
        common = importlib.import_module("_common")
    finally:
        sys.path.pop(0)

    doc = common.require_facts.__doc__ or ""
    assert "declares none" in doc
    assert "not applicable" in doc
    assert "shape" in doc

    # The three current callers all derive from a registry that cannot
    # honestly be empty, which is what makes the guard correct at those sites.
    for script in ("check_config_literals.py", "check_enum_cache.py", "check_mcp_parity.py"):
        assert "require_facts" in (CHECKS / script).read_text(encoding="utf-8")


def test_a_factory_default_enters_the_facts_as_its_empty_collection() -> None:
    """A factory field's repeatable default is the empty collection it builds.

    ``FieldInfo.default`` is ``PydanticUndefined`` for such a field — neither
    ``None`` nor callable — so a naive filter stored the sentinel and every
    comparison against it failed silently, dropping findings. ``LLMConfig
    .fallback_chain`` was in exactly that state.

    A sentinel is not the same as "no repeatable default" though: ``list()``
    builds a fresh object per instance, while the value a caller re-types is
    still an empty list. Both halves are asserted, because excluding the
    sentinel without adding the empty collection would just trade one silent
    miss for another.
    """
    module = _load_check("check_config_literals")
    defaults = module._config_defaults()

    factory_fields = [
        (model.__name__, name, info) for model in module.CONFIG_MODELS for name, info in model.model_fields.items() if info.default_factory is not None
    ]
    assert factory_fields, "no factory field left to guard — pick another model or drop this test"

    for model_name, field_name, info in factory_fields:
        built = info.default_factory() if info.default_factory else None
        stored = defaults.get(field_name, ())
        assert PydanticUndefined not in stored, f"{model_name}.{field_name} stored the sentinel"
        if isinstance(built, (list, dict, set, tuple)) and not built:
            assert built in stored, f"{model_name}.{field_name} builds an empty {type(built).__name__} that an empty literal repeats"
        else:
            assert built not in stored, f"{model_name}.{field_name} builds a non-literal; no caller can re-type it"


def test_an_empty_collection_literal_is_caught_but_a_call_is_not(tmp_path: Path) -> None:
    """An empty-list literal repeats the factory default; a call computes it."""
    module = _load_check("check_config_literals")

    class _Probe(BaseModel):
        plain: int = 7
        listed: list[str] = Field(default_factory=list)

    original = module.CONFIG_MODELS
    module.CONFIG_MODELS = [_Probe]
    try:
        defaults = module._config_defaults()
        assert defaults == {"plain": (7,), "listed": ([],)}

        cases = {
            "g(plain=7, listed=[])": True,
            "g(listed=list())": False,
            "g(listed=[])": True,
            # One non-repeating value does not excuse the repeating one.
            "g(plain=8, listed=[])": True,
        }
        probe = tmp_path / "probe.py"
        for source, expected in cases.items():
            probe.write_text(f"def f():\n    {source}\n", encoding="utf-8")
            findings = module._scan(probe, defaults, set())
            assert bool(findings) is expected, f"{source} -> {findings}"
            if findings:
                assert "List(" not in findings[0], "the message must print the literal, not its AST node"
    finally:
        module.CONFIG_MODELS = original


def test_a_field_name_shared_by_two_models_keeps_both_defaults(tmp_path: Path) -> None:
    """A field name is unique per model, not across the merged map.

    Keyed by name alone, a second model declaring the same field with a
    different default would overwrite the first, and a literal repeating the
    shadowed default would stop being reported — the same silent drop this
    module exists to prevent, one layer down. No name is shared today, so the
    shape must be right before it is needed.

    Asserting the tuple alone would be a shape test: it would pass if the scan
    still compared against one value. So this also scans literals for *both*
    defaults, which is the property that actually prevents the dropped finding.
    """
    module = _load_check("check_config_literals")

    class _First(BaseModel):
        timeout: int = 120

    class _Second(BaseModel):
        timeout: int = 30

    original = module.CONFIG_MODELS
    module.CONFIG_MODELS = [_First, _Second]
    probe = tmp_path / "probe.py"
    try:
        defaults = module._config_defaults()
        assert defaults == {"timeout": (120, 30)}, "one model's default must not shadow another's"

        for literal, expected in ((120, True), (30, True), (99, False)):
            probe.write_text(f"def f():\n    g(timeout={literal})\n", encoding="utf-8")
            findings = module._scan(probe, defaults, set())
            assert bool(findings) is expected, f"timeout={literal} -> {findings}"
    finally:
        module.CONFIG_MODELS = original


@pytest.mark.parametrize(("module_name", "attribute", "receiver", "empty"), _EMPTY_FACT_CASES)
def test_a_gate_exits_2_when_its_fact_source_is_empty(
    module_name: str, attribute: str, receiver: str | None, empty: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty the fact, assert exit 2 — the only test that proves the guard works.

    Parametrized from :data:`_EMPTY_FACT_CASES` so the coverage test below can
    prove every ``require_facts`` caller is exercised here, rather than merely
    imported by some test in this file.
    """
    module = _load_check(module_name)
    target = getattr(module, receiver) if receiver else module
    # A callable fact source (the default value is a function) is replaced by
    # one returning the empty value; a data attribute is replaced outright.
    replacement = mock.Mock(return_value=empty) if callable(getattr(target, attribute)) else empty
    monkeypatch.setattr(target, attribute, replacement)

    assert module.main() == 2, f"{module_name}.main() must fail with an empty {attribute}"


@pytest.mark.parametrize("module_name", [case[0] for case in _EMPTY_FACT_CASES])
def test_the_same_gate_still_passes_on_its_real_facts(module_name: str) -> None:
    """The gate still passes with real facts, so the guard is not blunt.

    Without this, a checker that unconditionally returned 2 would satisfy the
    emptiness test above.
    """
    assert _load_check(module_name).main() == 0


def test_every_require_facts_caller_has_an_empty_fact_case() -> None:
    """A guard cannot detect misuse — only a per-call-site emptying test can.

    ``require_facts`` sees a count, not a shape, so it is correct only where
    zero is genuinely impossible. Adding a fourth caller without a case in
    :data:`_EMPTY_FACT_CASES` fails here instead of silently trusting the guard.
    """
    callers = {p.stem for p in CHECKS.glob("check_*.py") if "require_facts(" in p.read_text(encoding="utf-8")}
    assert callers, "no require_facts callers found — did the helper get renamed?"

    covered = {case[0] for case in _EMPTY_FACT_CASES}
    assert callers == covered, (
        f"missing empty-fact cases: {sorted(callers - covered)}; stale cases: {sorted(covered - callers)}. "
        "Add (module, attribute, receiver, empty-value) to _EMPTY_FACT_CASES and prove exit 2 — "
        "see require_facts' docstring for why that test is the only protection."
    )


def test_a_rule_with_no_facts_refuses_to_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty facts must abort: no facts, no verdict.

    The check_persisted_paths shape, one level deeper — the scan reads its
    files but derives its judgement from an empty fact set, so it reports the
    absence of facts as the absence of violations.
    """
    _antipatterns()
    common = importlib.import_module("_common")

    with pytest.raises(SystemExit) as exc_info:
        common.require_facts(0, "widgets", "Fix: rebuild the widget registry.")

    assert exc_info.value.code == 2

    # And a non-empty set is a pass-through, not a no-op that always raises.
    assert common.require_facts(3, "widgets", "Fix: n/a") is None


def test_fact_fed_gates_abort_when_their_facts_vanish(tmp_path: Path) -> None:
    """All fact-fed gates return 2 when their fact source is empty."""
    config_literals = _load_check("check_config_literals")
    enum_cache = _load_check("check_enum_cache")
    parity = _load_check("check_mcp_parity")

    with mock.patch.object(config_literals, "_config_defaults", return_value={}):
        assert config_literals.main() == 2

    with mock.patch.object(enum_cache.store, "_CACHE_DATACLASSES", {}):
        assert enum_cache.main() == 2

    with mock.patch.object(parity, "get_cli_commands", return_value=[]):
        assert parity.main() == 2


def _load_check(name: str) -> Any:
    """Import a gate script by module name (scripts/checks is not a package)."""
    sys.path.insert(0, str(CHECKS))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.pop(0)


def test_the_verdict_does_not_depend_on_where_the_scanner_stands(tmp_path: Path) -> None:
    """Level 0: identity. A gate must not silently judge some other project.

    Absence is testable by emptying an input; substituting a different source
    is not. The test is to move the scanner somewhere else and require that it
    either still reads *this* project or refuses — never a confident verdict
    about the wrong tree.
    """
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def run_from(script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, str(CHECKS / script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=elsewhere,
            timeout=300,
            check=False,
        )

    # `repo_root()` walks up from __file__ to a pyproject marker, so these still
    # read this repository — the verdict must be identical, not merely non-empty.
    for script in ("check_antipatterns.py", "check_config_literals.py"):
        here = run_from(script)
        assert here.returncode == 0, f"{script} from a foreign cwd: {here.stderr[-400:]}"
        assert "scanned" in here.stdout

    # These two take a directory that defaults to cwd, so a foreign cwd is a
    # foreign project: refuse rather than report a clean tree.
    for script in ("check_dox.py", "check_persisted_paths.py"):
        there = run_from(script)
        assert there.returncode == 2, f"{script} must not certify a foreign cwd:\n{there.stdout}\n{there.stderr}"


def test_the_default_vault_is_the_cwd_and_nothing_else() -> None:
    """No plausible substitute: the cwd is reported as-is, or the scan fails."""
    sys.path.insert(0, str(CHECKS))
    try:
        persisted = importlib.import_module("check_persisted_paths")
    finally:
        sys.path.pop(0)

    assert persisted._default_vault() == Path.cwd()


def test_a_missing_project_root_is_not_a_pass(tmp_path: Path) -> None:
    """``--root`` pointing nowhere must abort, not report a clean DOX tree."""
    result = _run("check_dox.py", "--root", str(tmp_path / "does-not-exist"))

    assert result.returncode == 2, f"a bad --root must fail loudly:\n{result.stdout}\n{result.stderr}"
    assert "does not exist" in result.stderr


def test_a_root_with_no_dox_files_is_not_a_pass(tmp_path: Path) -> None:
    """A directory with no DOX at all cannot be certified clean."""
    empty = tmp_path / "empty"
    empty.mkdir()

    result = _run("check_dox.py", "--root", str(empty))

    assert result.returncode == 2
    assert "no DOX files" in result.stderr


def test_a_missing_source_root_aborts_rather_than_under_covering(tmp_path: Path) -> None:
    """A typo in a repeated --src must not shrink the scan silently."""
    result = _run("check_antipatterns.py", "--src", str(tmp_path / "does-not-exist"))

    assert result.returncode == 2
    assert "does not exist" in result.stderr


def test_a_single_file_source_root_is_scanned(tmp_path: Path) -> None:
    """``--src`` may name one file, which is scanned rather than skipped."""
    module = _antipatterns()
    single = tmp_path / "one.py"
    single.write_text("x = 1\n", encoding="utf-8")

    assert module.discover_py_files([single]) == [single]


def test_an_unparseable_file_aborts_the_scan(tmp_path: Path) -> None:
    """A file the gate cannot parse is a blind spot, not a clean bill."""
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "bad.py").write_text("def f(:\n", encoding="utf-8")

    result = _run("check_antipatterns.py", "--src", str(broken))

    assert result.returncode == 2
    assert "cannot parse" in result.stderr


def test_every_vocab_enum_allowlist_entry_still_names_a_strenum() -> None:
    """A typo in the allowlist silently disables checking for that enum.

    ``_vocab_enum_values`` skips a name that matches no ``StrEnum`` class, so a
    renamed or misspelt entry removes that vocabulary from the fact set and
    every bare-string comparison naming it goes unreported — the silence this
    module exists to prevent, in its own configuration. (The same rule as the
    lookup-table clause in ``.agents/CHECKS.md``: a table whose wrong entry can
    only make the scanner quieter needs a test that its entries still match.)
    """
    module = _antipatterns()

    class_names = {
        node.name
        for py in module.SRC_ROOT.rglob("*.py")
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8", errors="replace")))
        if isinstance(node, ast.ClassDef)
    }
    unmatched = sorted(set(module._VOCAB_ENUM_NAMES) - class_names)

    assert not unmatched, f"_VOCAB_ENUM_NAMES entries matching no class in src/: {unmatched}"


def test_every_module_constant_is_read_somewhere() -> None:
    """A module-level constant with no reader is unclassified, not misclassified.

    Coverage checks assume an input set, guess-direction assumes a consumer, and
    table verification assumes a lookup — all three pass by absence. Found this
    way: ``SIZE_EXEMPT`` in ``check_dox.py`` had no reader at all (root
    ``AGENTS.md`` is exempted by a different branch), which no other class can
    see.

    Deliberate limit: a cross-module re-export would read as an orphan here, so
    the check stays single-file and skips dunders. No uppercase constant in
    ``scripts/checks/`` is imported by another module today (verified), which is
    why the simple form is enough; the docstring says so rather than pretending
    the check is general.
    """
    orphans: list[str] = []
    for path in sorted(CHECKS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        declared: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                declared.append(node.targets[0].id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                declared.append(node.target.id)
        used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        orphans.extend(f"{path.name}:{name}" for name in declared if name.isupper() and name not in used)

    assert not orphans, f"module-level constants with no reader in their own file: {orphans}"


def test_every_pip_policy_file_actually_exists() -> None:
    """A stale exemption path stops covering the policy doc that needs it.

    ``scan_pip_installs`` exempts ``_PIP_POLICY_FILES`` by membership. Measured:
    a *wrong* entry (a path that does not exist) exempts nothing and the policy
    file it was meant to cover is scanned normally — so the failure is a false
    finding, loud rather than silent. This test still earns its place: the loud
    failure would appear in an unrelated file with no hint that an exemption
    drifted, and pinning existence turns that into a named error here.
    """
    module = _antipatterns()

    missing = sorted(str(p.relative_to(module.REPO_ROOT)) for p in module._PIP_POLICY_FILES if not p.is_file())

    assert not missing, f"_PIP_POLICY_FILES entries that no longer exist: {missing}"


def test_an_empty_source_root_is_not_a_pass(tmp_path: Path) -> None:
    """Scanning zero Python files must abort, not report success."""
    empty = tmp_path / "empty"
    empty.mkdir()

    result = _run("check_antipatterns.py", "--src", str(empty))

    assert result.returncode == 2, f"vacuous scan must fail loudly:\n{result.stdout}\n{result.stderr}"
    assert "nothing to scan" in result.stderr


def test_field_consistency_aborts_on_an_unparseable_file(tmp_path: Path) -> None:
    """The same blind spot in the second gate that used to skip it.

    check_field_consistency returned [] on OSError/SyntaxError — the silent-skip
    class fixed in the other gates, and an earlier audit of mine called it
    benign. It is not: a file the gate cannot parse is where a bare contract
    key could hide.
    """
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "bad.py").write_text("def f(:\n", encoding="utf-8")

    result = _run("check_field_consistency.py", "--src", str(broken))

    assert result.returncode == 2, f"must abort on a syntax error:\n{result.stdout}\n{result.stderr}"
    assert "cannot parse" in result.stderr


def test_a_fact_provider_that_cannot_read_aborts_rather_than_emptying(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A rule must not lose its facts silently.

    ``_path_filename_constants`` and ``_vocab_enum_values`` decide their rules
    by reading files outside the one being scanned. Returning an empty set on
    failure reads as "this project has no such constants", which disarms the
    rule without a word — the same class as an empty source root.
    """
    module = _antipatterns()
    missing = tmp_path / "nope.py"

    monkeypatch.setattr(module, "_PATHS_MODULE", missing)
    module._path_filename_constants.cache_clear()
    with pytest.raises(module.ScanError, match="path constants"):
        module._path_filename_constants()
    module._path_filename_constants.cache_clear()

    monkeypatch.setattr(module, "SRC_ROOT", tmp_path)
    module._vocab_enum_values.cache_clear()
    (tmp_path / "bad.py").write_text("class X(StrEnum\\n", encoding="utf-8")
    with pytest.raises(module.ScanError, match="enum values"):
        module._vocab_enum_values()
    module._vocab_enum_values.cache_clear()


# ── check_antipatterns: the scanner must not lie in either direction ────────


def _antipatterns() -> Any:
    """Import the checker module (it lives outside the package tree)."""
    sys.path.insert(0, str(CHECKS))
    try:
        return importlib.import_module("check_antipatterns")
    finally:
        sys.path.pop(0)


def _runner() -> Any:
    """Import the gate runner module (it lives outside the package tree)."""
    sys.path.insert(0, str(CHECKS))
    try:
        return importlib.import_module("all")
    finally:
        sys.path.pop(0)


def test_the_runner_tells_a_violation_from_an_unjudged_gate(tmp_path: Path) -> None:
    """Exit 1 is a violation; exit 2 is a gate that could not judge.

    The two are not the same, and the runner must not merge them: a gate that
    read nothing proved nothing, so reporting it as passed is the same
    confidently-wrong bug the exit-2 convention exists to prevent. The runner's
    own exit code follows the same convention — 1 for a violation, 2 for an
    unjudged gate, 0 only when every gate judged and found nothing.
    """
    runner = _runner()

    clean = runner._run_steps([("judged-clean", _gate_command(0))])
    assert clean == 0

    unjudged = runner._run_steps([("judged-clean", _gate_command(0)), ("could-not-judge", _gate_command(2))])
    assert unjudged == 2, "a gate that could not judge must not yield a clean run"

    violated = runner._run_steps([("judged-clean", _gate_command(0)), ("found-violations", _gate_command(1))])
    assert violated == 1, "a gate that found violations must fail the run"


def test_an_unjudged_gate_is_never_reported_as_ok(capsys: pytest.CaptureFixture[str]) -> None:
    """The runner says what it did: judged, or it did not judge."""
    runner = _runner()

    code = runner._run_steps([("judged-clean", _gate_command(0)), ("could-not-judge", _gate_command(2))])

    out = capsys.readouterr().out
    assert code == 2
    assert "[ok]" in out
    assert "[n/a]" in out, f"the unjudged gate must not be reported as ok:\n{out}"
    assert out.count("[ok]") == 1


def test_the_vault_gate_is_given_a_subject_it_can_scan(tmp_path: Path) -> None:
    """The vault gate refuses to certify a scan that read nothing.

    The repository root is not a vault, so the runner builds one — otherwise
    the documented "run every gate" command could never pass in a fresh
    checkout, because the gate would have nothing to scan.
    """
    runner = _runner()
    vault = tmp_path / "vault"

    runner._build_vault(vault)

    store = vault / ".kompend" / "kompend.db"
    assert store.is_file(), "the runner must build a real store for the vault gate to scan"

    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(CHECKS / "check_persisted_paths.py"), "--vault", str(vault)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, f"the vault gate did not judge the vault:\n{result.stdout}\n{result.stderr}"
    assert "artifact(s) scanned" in result.stdout


def _gate_command(code: int) -> list[str]:
    """An argv for a fake gate that exits with *code*."""
    return [sys.executable, "-c", f"import sys; sys.exit({code})"]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(CHECKS / script), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        timeout=180,
        check=False,
    )
