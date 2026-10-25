"""Tests for CLI doctor and status commands."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from kompend.cli.app import app
from kompend.cli.cmds.pipeline.status import collect_stats, print_status
from kompend.cli.cmds.quality import doctor
from kompend.cli.cmds.quality.doctor import CheckResult, print_doctor_results, run_doctor
from kompend.core.paths import COMPILE_STATE_FILE, KOMP_DB_FILE
from kompend.core.pipeline.state import SourceState, TierState, TierStateStore
from kompend.llm.context_window import ContextSource, ContextWindow


class TestCheckResult:
    def test_to_dict(self) -> None:
        result = CheckResult("test-check", True, "All good")
        d = result.to_dict()
        assert d["name"] == "test-check"
        assert d["passed"] is True
        assert d["message"] == "All good"

    def test_to_dict_failed(self) -> None:
        result = CheckResult("failing-check", False, "Something broke")
        d = result.to_dict()
        assert d["passed"] is False


class TestPrintDoctorResults:
    def test_all_passed_exit_code(self) -> None:
        results = [
            CheckResult("a", True, "OK"),
            CheckResult("b", True, "OK"),
        ]
        code = print_doctor_results(results)
        assert code == 0

    def test_some_failed_exit_code(self) -> None:
        results = [
            CheckResult("a", True, "OK"),
            CheckResult("b", False, "FAIL"),
        ]
        code = print_doctor_results(results)
        assert code == 1

    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [
            CheckResult("a", True, "OK"),
            CheckResult("b", False, "FAIL"),
        ]
        print_doctor_results(results, json_output=True)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["passed"] == 1
        assert data["failed"] == 1
        assert data["total"] == 2


class TestRunDoctor:
    def test_basic_run(self) -> None:
        """Doctor runs without crashing, returns CheckResult list."""
        results = run_doctor()
        assert isinstance(results, list)
        assert len(results) >= 5
        assert all(isinstance(r, CheckResult) for r in results)

    def test_with_paths(self, tmp_path: Path) -> None:
        """Doctor accepts explicit paths."""
        results = run_doctor(
            sources_dir=tmp_path,
            output_dir=tmp_path,
        )
        # sources-directory and output-directory checks should pass
        src_check = next(r for r in results if r.name == "sources-directory")
        out_check = next(r for r in results if r.name == "output-directory")
        assert src_check.passed is True
        assert out_check.passed is True

    def test_no_model_fails_llm(self) -> None:
        """Without a model, LLM connectivity check is a hard failure."""
        results = run_doctor()
        llm_check = next(r for r in results if r.name == "connectivity-llm")
        assert llm_check.passed is False
        assert "skipped" in llm_check.message

    def test_vlm_skipped_without_model(self) -> None:
        """Without a VLM model, VLM check is a warning, not a pass."""
        results = run_doctor()
        vlm_check = next(r for r in results if r.name == "connectivity-vlm")
        assert vlm_check.passed is True
        assert vlm_check.warning is True
        assert "skipped" in vlm_check.message

    def test_embed_skipped_without_model(self) -> None:
        """Without an embed model, embed check is a warning, not a pass."""
        results = run_doctor()
        embed_check = next(r for r in results if r.name == "connectivity-embed")
        assert embed_check.passed is True
        assert embed_check.warning is True
        assert "skipped" in embed_check.message


class TestDoctorWikiContext:
    """CLI-level tests for wiki-context enforcement."""

    def test_fails_without_wiki_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Doctor exits 1 when no --wiki-dir and no output dir exists."""
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1
        assert isinstance(result.exception, LookupError)

    def test_works_with_explicit_wiki_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Doctor runs (bypasses wiki-context gate) when --wiki-dir is set."""
        (tmp_path / ".kompend").mkdir()
        (tmp_path / ".kompend" / "config.toml").write_text("", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["--wiki-dir", str(tmp_path), "doctor"])
        assert "not a kompend vault root" not in result.output
        assert "checks passed" in result.output


class TestCollectStats:
    def test_no_dirs(self) -> None:
        stats = collect_stats()
        assert stats["sources"]["count"] == 0
        assert stats["articles"]["count"] == 0
        assert stats["last_compile"] is None

    def test_with_source_files(self, tmp_path: Path) -> None:
        """Source count reflects files in the sources dir."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("test content")
        stats = collect_stats(sources_dir=tmp_path)
        assert stats["sources"]["count"] == 1
        assert stats["sources"]["total_size_bytes"] > 0

    def test_with_output_files(self, tmp_path: Path) -> None:
        """Article count from wiki.md."""
        wiki = tmp_path / "wiki.md"
        wiki.write_text("# Title\n\nContent here.\n\n# Another\n\nMore words.")
        stats = collect_stats(output_dir=tmp_path)
        assert stats["articles"]["total_words"] > 0

    def test_pages_counted_from_kind_folders(self, tmp_path: Path) -> None:
        """Pages live in the kind folders - the bundle root files are not pages."""
        (tmp_path / "concepts").mkdir()
        (tmp_path / "concepts" / "speech-intelligibility.md").write_text("# Speech Intelligibility\n\nWords about speech.", encoding="utf-8")
        (tmp_path / "index.md").write_text("# Index\n\nBundle index.", encoding="utf-8")
        (tmp_path / "log.md").write_text("# Log\n\nBundle log.", encoding="utf-8")
        stats = collect_stats(output_dir=tmp_path)
        assert stats["articles"]["count"] == 1
        assert stats["articles"]["total_words"] == 6

    def test_config_file_reported(self, tmp_path: Path) -> None:
        """The vault's own config is reported - not one found in the cwd."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("", encoding="utf-8")
        stats = collect_stats(config_file=config_file)
        assert stats["config_file"] == str(config_file)

    def test_missing_config_file_reported_as_none(self, tmp_path: Path) -> None:
        stats = collect_stats(config_file=tmp_path / "absent.toml")
        assert stats["config_file"] is None

    def test_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        stats = collect_stats(sources_dir=tmp_path)
        print_status(stats, json_output=True)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["wiki_name"] == "kompend Wiki"

    def test_compile_status_not_initialized(self, tmp_path: Path) -> None:
        """No state_dir → not-initialized."""
        stats = collect_stats(sources_dir=tmp_path, output_dir=tmp_path, state_dir=None)
        assert stats["compile_status"]["status"] == "not-initialized"

    def test_compile_status_no_sources(self, tmp_path: Path) -> None:
        """state_dir exists but no sources → no-sources."""
        state_dir = tmp_path / ".kompend"
        state_dir.mkdir()
        stats = collect_stats(state_dir=state_dir)
        assert stats["compile_status"]["status"] == "no-sources"

    def test_compile_status_not_indexed(self, tmp_path: Path) -> None:
        """Sources exist but no tier state file → not-indexed."""
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "doc.pdf").write_text("x")
        state_dir = tmp_path / ".kompend"
        state_dir.mkdir()
        stats = collect_stats(sources_dir=sources, state_dir=state_dir)
        assert stats["compile_status"]["status"] == "not-indexed"
        assert stats["compile_status"]["total"] == 1

    def test_compile_status_in_progress(self, tmp_path: Path) -> None:
        """Tier state with entries at tier < 4 → in-progress."""
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "doc.pdf").write_text("x")
        state_dir = tmp_path / ".kompend"
        state_dir.mkdir()
        state = TierState(
            entries={
                "doc": SourceState(stem="doc", source_file="doc.pdf", content_hash="h", current_tier=1),
            }
        )
        TierStateStore(state_dir / COMPILE_STATE_FILE).save(state)
        stats = collect_stats(sources_dir=sources, state_dir=state_dir)
        assert stats["compile_status"]["status"] == "in-progress"
        assert stats["tier_distribution"] == {1: 1}

    def test_compile_status_compiled(self, tmp_path: Path) -> None:
        """All entries at tier 4 + wiki exists → compiled."""
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "doc.pdf").write_text("x")
        output = tmp_path / "wiki"
        output.mkdir()
        (output / "wiki.md").write_text("# Wiki\n\nContent.")
        state_dir = tmp_path / ".kompend"
        state_dir.mkdir()
        state = TierState(
            entries={
                "doc": SourceState(stem="doc", source_file="doc.pdf", content_hash="h", current_tier=4),
            }
        )
        TierStateStore(state_dir / COMPILE_STATE_FILE).save(state)
        stats = collect_stats(sources_dir=sources, output_dir=output, state_dir=state_dir)
        assert stats["compile_status"]["status"] == "compiled"
        assert stats["compile_status"]["compiled"] == 1

    def test_compile_status_broken(self, tmp_path: Path) -> None:
        """Corrupt store → broken."""
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "doc.pdf").write_text("x")
        state_dir = tmp_path / ".kompend"
        state_dir.mkdir()
        (state_dir / KOMP_DB_FILE).write_text("not a database {{{", encoding="utf-8")
        stats = collect_stats(sources_dir=sources, state_dir=state_dir)
        assert stats["compile_status"]["status"] == "broken"
        assert stats["compile_status"]["error"]


class TestRunDoctorOrder:
    """Connectivity pings must run before context-window discovery so /api/ps reports the served context (not GGUF metadata)."""

    def test_connectivity_precedes_context_window(self) -> None:
        names = [r.name for r in run_doctor()]
        if "connectivity-llm" in names and "context-window-llm" in names:
            assert names.index("connectivity-llm") < names.index("context-window-llm")
        if "connectivity-vlm" in names and "context-window-vlm" in names:
            assert names.index("connectivity-vlm") < names.index("context-window-vlm")

    def test_output_ceiling_reports_the_protocol_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "build_model", lambda model: object())
        monkeypatch.setattr(doctor, "requires_output_ceiling", lambda model: True)
        monkeypatch.setattr(doctor, "resolve_model_limits", lambda model: (1_000_000, 524_288))

        result = doctor._check_output_ceiling("minimax/MiniMax-M3.1-Flash", "llm")

        assert result.name == "output-ceiling-llm"
        assert result.passed
        assert "524288-token output limit" in result.message

    def test_output_ceiling_absent_for_openai_protocols(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "build_model", lambda model: object())
        monkeypatch.setattr(doctor, "requires_output_ceiling", lambda model: False)

        result = doctor._check_output_ceiling("ollama/qwen3", "llm")

        assert result.passed
        assert "provider default" in result.message

    def test_catalog_reports_the_cached_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A usable cache is reported with its age and TTL."""
        snapshot = _catalog_snapshot()
        monkeypatch.setattr(doctor.catalog, "ensure_fresh", lambda ttl_hours=None: False)
        monkeypatch.setattr(doctor.catalog, "snapshot", lambda refresh=False: snapshot)

        result = doctor._check_catalog()

        assert result.passed
        assert "1 providers" in result.message
        assert "cached" in result.message

    def test_catalog_refreshes_a_stale_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stale cache is refreshed once and reported as refreshed."""
        snapshot = _catalog_snapshot()
        called: list[bool] = []
        monkeypatch.setattr(doctor.catalog, "ensure_fresh", lambda ttl_hours=None: called.append(True) or True)
        monkeypatch.setattr(doctor.catalog, "snapshot", lambda refresh=False: snapshot)

        result = doctor._check_catalog()

        assert called == [True]
        assert result.passed
        assert "refreshed" in result.message

    def test_catalog_without_snapshot_is_a_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No cache and a failed download must not fail the doctor run."""
        monkeypatch.setattr(doctor.catalog, "ensure_fresh", lambda ttl_hours=None: False)
        monkeypatch.setattr(doctor.catalog, "snapshot", lambda refresh=False: None)

        result = doctor._check_catalog()

        assert not result.passed
        assert result.warning
        assert "no cached copy" in result.message

    def test_window_fit_reports_a_usable_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            doctor,
            "discover_context_window",
            lambda *a, **k: ContextWindow(max_input=32_768, max_output=32_768, source=ContextSource.OLLAMA),
        )

        result = doctor._check_window_fit("ollama/test")

        assert result.passed
        assert "8192 answer reserve" in result.message

    def test_window_fit_rejects_a_window_without_input_room(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A window whose answer reserve leaves no room for input cannot compile."""
        monkeypatch.setattr(
            doctor,
            "discover_context_window",
            lambda *a, **k: ContextWindow(max_input=600, max_output=600, source=ContextSource.OLLAMA),
        )

        result = doctor._check_window_fit("ollama/test")

        assert not result.passed
        assert "increase the model's context window" in result.message

    def test_window_fit_after_context_window(self) -> None:
        """The window-fit check reads the discovered window, so it must follow it."""
        names = [r.name for r in run_doctor()]
        if "context-window-llm" in names and "window-fit" in names:
            assert names.index("context-window-llm") < names.index("window-fit")

    def test_llm_block_precedes_vlm_block(self) -> None:
        """Per-model blocks: every llm check runs before the vlm block starts."""
        names = [r.name for r in run_doctor()]
        if "structured-output-llm" in names and "connectivity-vlm" in names:
            assert names.index("structured-output-llm") < names.index("connectivity-vlm")

    def test_embed_after_conversation_models(self) -> None:
        """Embedding runs last — loading it evicts llm/vlm from Ollama mid-run."""
        names = [r.name for r in run_doctor()]
        if "connectivity-embed" in names:
            for other in ("connectivity-llm", "structured-output-llm", "connectivity-vlm", "ollama-device"):
                if other in names:
                    assert names.index("connectivity-embed") > names.index(other)


class TestPingModelWarmup:
    """``_ping_model`` discards the cold-start run and reports warm throughput."""

    def test_second_run_is_measured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runs = 0

        class _FakeTracker:
            completion_tokens = 0

        class _FakeAgent:
            def __init__(self, **kwargs: object) -> None:
                self._tracker: Any = kwargs["usage_tracker"]

            def run(self, _deps: object) -> str:
                nonlocal runs
                runs += 1
                # Cold start emits far more tokens than the warm run; only the
                # warm delta must feed the reported rate.
                self._tracker.completion_tokens += 1000 if runs == 1 else 5
                return "ok"

        ticks = iter([10.0, 11.0])
        monkeypatch.setattr(doctor, "DoctorAgent", _FakeAgent)
        monkeypatch.setattr(doctor, "UsageTracker", _FakeTracker)
        monkeypatch.setattr(doctor, "monotonic", lambda: next(ticks))

        result = doctor._ping_model("ollama/foo", "llm")

        assert runs == 2  # warm-up + measured run
        assert result.passed is True
        assert "5.0 tok/s" in result.message


class TestCheckOllamaDevice:
    """``ollama-device`` doctor check reports GPU offload status per referenced model."""

    def test_skipped_without_ollama_model(self) -> None:

        result = doctor._check_ollama_device(default_model="openai/gpt-4o", vlm_model=None)
        assert result.passed is True
        assert result.warning is False
        assert "skipped" in result.message

    def test_reports_full_gpu_offload(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setattr(
            doctor,
            "_ollama_ps_lookup",
            lambda name: {"size": 100 * 1024 * 1024, "size_vram": 100 * 1024 * 1024, "context_length": 65536, "digest": "x"},
        )
        result = doctor._check_ollama_device("ollama/foo", None)
        assert result.passed is True
        assert result.warning is False
        assert "GPU" in result.message

    def test_warns_on_partial_offload(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setattr(
            doctor,
            "_ollama_ps_lookup",
            lambda name: {"size": 100 * 1024 * 1024, "size_vram": 30 * 1024 * 1024, "context_length": 65536, "digest": "x"},
        )
        result = doctor._check_ollama_device("ollama/foo", None)
        assert result.passed is True
        assert result.warning is True
        assert "partial" in result.message

    def test_warns_on_cpu_only(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setattr(
            doctor,
            "_ollama_ps_lookup",
            lambda name: {"size": 100 * 1024 * 1024, "size_vram": 0, "context_length": 65536, "digest": "x"},
        )
        result = doctor._check_ollama_device("ollama/foo", None)
        assert result.passed is True
        assert result.warning is True
        assert "CPU" in result.message

    def test_reports_not_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setattr(doctor, "_ollama_ps_lookup", lambda name: None)
        result = doctor._check_ollama_device("ollama/foo", None)
        assert "not loaded" in result.message

    def test_same_llm_and_vlm_model_merged_into_one_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One model serving both roles is loaded once — report it once."""
        monkeypatch.setattr(
            doctor,
            "_ollama_ps_lookup",
            lambda name: {"size": 100 * 1024 * 1024, "size_vram": 100 * 1024 * 1024, "context_length": 65536, "digest": "d1"},
        )
        result = doctor._check_ollama_device("ollama/foo:8b", "ollama/foo:8b")
        assert result.passed is True
        assert result.warning is False
        assert "llm+vlm: GPU (100/100 MiB VRAM, 100%)" in result.message

    def test_distinct_llm_and_vlm_models_not_merged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Different digests keep separate per-role lines."""
        entries = {
            "foo:8b": {"size": 100 * 1024 * 1024, "size_vram": 100 * 1024 * 1024, "context_length": 65536, "digest": "d1"},
            "bar:2b": {"size": 30 * 1024 * 1024, "size_vram": 15 * 1024 * 1024, "context_length": 4096, "digest": "d2"},
        }
        monkeypatch.setattr(doctor, "_ollama_ps_lookup", entries.get)
        result = doctor._check_ollama_device("ollama/foo:8b", "ollama/bar:2b")
        assert "llm: GPU (100/100 MiB VRAM, 100%)" in result.message
        assert "vlm: partial (15/30 MiB VRAM, 50%)" in result.message

    def test_unloaded_same_name_shared_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not-loaded entries group by name only."""
        monkeypatch.setattr(doctor, "_ollama_ps_lookup", lambda name: None)
        result = doctor._check_ollama_device("ollama/foo:8b", "ollama/foo:8b")
        assert result.message.startswith("llm+vlm: not loaded")

    def test_nvidia_unused_is_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:

        monkeypatch.setattr(
            doctor,
            "_ollama_ps_lookup",
            lambda name: {"size": 100 * 1024 * 1024, "size_vram": 0, "context_length": 65536, "digest": "x"},
        )
        monkeypatch.setattr(
            doctor,
            "_nvidia_smi_probe",
            lambda: {"name": "RTX 4090", "memory_total_mib": 24576, "memory_used_mib": 0},
        )
        result = doctor._check_ollama_device("ollama/foo", None)
        assert "RTX 4090" in result.message
        assert "not using it" in result.message

    def _ps(self, models: list) -> object:
        return SimpleNamespace(models=[SimpleNamespace(**m) for m in models])


class TestOllamaPsLookupTagMatching:
    """``_ollama_ps_lookup`` matches by full name; bare-name fallback only when untagged."""

    def test_exact_tag_match_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_ps(
            monkeypatch,
            [self._MODEL | {"model": "qwen3:8b", "digest": "d8"}, self._MODEL | {"model": "qwen3:4b", "name": "qwen3:4b", "digest": "d4"}],
        )
        entry = doctor._ollama_ps_lookup("qwen3:4b")
        assert entry is not None
        assert entry["digest"] == "d4"

    def test_tagged_name_does_not_absorb_other_variant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Configured ``qwen3:4b`` must not report ``qwen3:8b``'s stats."""
        self._patch_ps(monkeypatch, [self._MODEL | {"model": "qwen3:8b", "digest": "d8"}])
        assert doctor._ollama_ps_lookup("qwen3:4b") is None

    def test_untagged_name_falls_back_to_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_ps(monkeypatch, [self._MODEL | {"model": "qwen3:latest", "digest": "dl"}])
        entry = doctor._ollama_ps_lookup("qwen3")
        assert entry is not None
        assert entry["digest"] == "dl"

    def test_untagged_name_matches_any_loaded_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_ps(monkeypatch, [self._MODEL | {"model": "qwen3:4b", "digest": "d4"}])
        entry = doctor._ollama_ps_lookup("qwen3")
        assert entry is not None
        assert entry["digest"] == "d4"

    def _patch_ps(self, monkeypatch: pytest.MonkeyPatch, entries: list[dict]) -> None:
        ps = SimpleNamespace(models=[SimpleNamespace(**e) for e in entries])
        monkeypatch.setattr(doctor.ollama, "ps", lambda: ps)

    _MODEL = {"name": "qwen3:8b", "size": 100, "size_vram": 100, "context_length": 65536}


class TestVlmFallbackIndication:
    def test_fallback_reported_without_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No explicit VLM + vision-capable default → plain info, no warning."""
        calls = []

        def _fake_resolve(default: str | None, vlm: str | None) -> str | None:
            calls.append((default, vlm))
            return "openai/gpt-5.2" if vlm is None else vlm

        monkeypatch.setattr(doctor, "resolve_vlm_model", _fake_resolve)
        monkeypatch.setattr(doctor, "_check_sources_dir", lambda *a, **k: CheckResult("sources", True))
        monkeypatch.setattr(doctor, "_check_output_dir", lambda *a, **k: CheckResult("output", True))
        monkeypatch.setattr(doctor, "_check_parsers", lambda *a, **k: CheckResult("parsers", True))
        monkeypatch.setattr(doctor, "_check_git_repo", lambda *a, **k: CheckResult("git", True))
        monkeypatch.setattr(doctor, "_check_disk_space", lambda *a, **k: CheckResult("disk", True))
        monkeypatch.setattr(doctor, "_ping_model", lambda *a, **k: CheckResult("ping", True))
        monkeypatch.setattr(doctor, "_ping_embed_model", lambda *a, **k: CheckResult("embed", True))
        monkeypatch.setattr(doctor, "_check_context_window", lambda *a, **k: CheckResult("ctx", True))
        monkeypatch.setattr(doctor, "_check_window_fit", lambda *a, **k: CheckResult("window-fit", True))
        monkeypatch.setattr(doctor, "_check_model_capabilities", lambda *a, **k: [CheckResult("struct", True)])
        monkeypatch.setattr(doctor, "_check_custom_providers", lambda *a, **k: CheckResult("providers", True))
        monkeypatch.setattr(doctor, "_check_ollama_device", lambda *a, **k: CheckResult("device", True))

        results = run_doctor(default_model="openai/gpt-5.2", vlm_model=None)

        vlm = next(r for r in results if r.name == "connectivity-vlm")
        assert vlm.passed is True
        assert vlm.warning is False
        assert "gpt-5.2" in vlm.message
        assert "vision-capable" in vlm.message
        # The fallback resolution is consulted with the real default model.
        assert calls
        assert calls[-1] == ("openai/gpt-5.2", None)
        # No duplicate probe — the connectivity result IS the informational note.
        assert vlm.message.startswith("No vlm model configured")

    def test_explicit_vlm_still_pinged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit vlm_model keeps the normal ping path (no fallback note)."""
        pings: list[tuple[str, str]] = []

        def _fake_ping(model, model_label, **_kwargs) -> CheckResult:  # noqa: ANN001 — mock signature mirrors _ping_model
            pings.append((model, model_label))
            return CheckResult(f"connectivity-{model_label}", True, f"{model_label} OK")

        monkeypatch.setattr(doctor, "resolve_vlm_model", lambda default, vlm: vlm or default)
        monkeypatch.setattr(doctor, "_check_sources_dir", lambda *a, **k: CheckResult("sources", True))
        monkeypatch.setattr(doctor, "_check_output_dir", lambda *a, **k: CheckResult("output", True))
        monkeypatch.setattr(doctor, "_check_parsers", lambda *a, **k: CheckResult("parsers", True))
        monkeypatch.setattr(doctor, "_check_git_repo", lambda *a, **k: CheckResult("git", True))
        monkeypatch.setattr(doctor, "_check_disk_space", lambda *a, **k: CheckResult("disk", True))
        monkeypatch.setattr(doctor, "_ping_model", _fake_ping)
        monkeypatch.setattr(doctor, "_ping_embed_model", lambda *a, **k: CheckResult("embed", True))
        monkeypatch.setattr(doctor, "_check_context_window", lambda *a, **k: CheckResult("ctx", True))
        monkeypatch.setattr(doctor, "_check_window_fit", lambda *a, **k: CheckResult("window-fit", True))
        monkeypatch.setattr(doctor, "_check_model_capabilities", lambda *a, **k: [CheckResult("struct", True)])
        monkeypatch.setattr(doctor, "_check_custom_providers", lambda *a, **k: CheckResult("providers", True))
        monkeypatch.setattr(doctor, "_check_ollama_device", lambda *a, **k: CheckResult("device", True))

        results = run_doctor(default_model="openai/gpt-5.2", vlm_model="ollama/llava")

        vlm = next(r for r in results if r.name == "connectivity-vlm")
        assert ("ollama/llava", "vlm") in pings
        assert "No vlm model configured" not in vlm.message


class TestCheckCustomProviders:
    """`[providers.<name>]` override-only blocks must not require base_url."""

    def test_override_only_block_for_native_provider_passes(self) -> None:
        cfg = {"providers": {"ollama": {"context_window": 65536}}}
        result = doctor._check_custom_providers(cfg, {"ollama"})
        assert result.passed is True
        assert "missing base_url" not in result.message

    def test_standalone_provider_without_base_url_fails(self) -> None:
        cfg = {"providers": {"totally-unknown-xyz": {"context_window": 1}}}
        result = doctor._check_custom_providers(cfg, {"totally-unknown-xyz"})
        assert result.passed is False
        assert "missing base_url" in result.message

    def test_no_base_url_unreferenced_skipped(self) -> None:
        cfg = {"providers": {"ollama": {"context_window": 65536}}}
        result = doctor._check_custom_providers(cfg, {"openai"})
        assert result.passed is True


def _catalog_snapshot() -> SimpleNamespace:
    """A minimal catalogue snapshot for the doctor's reporting check."""
    return SimpleNamespace(providers={"acme": object()}, fetched_at=None)


def test_provenance_no_wiki() -> None:
    """Kompend provenance without compiled wiki should exit with error."""
    runner = CliRunner()
    result = runner.invoke(app, ["provenance", "test-page"])
    assert result.exit_code != 0
    assert result.exit_code == 1


def test_provenance_multiple_slugs_accepted() -> None:
    """Kompend provenance accepts multiple positional slugs (variadic)."""
    runner = CliRunner()
    # No wiki loaded, but Typer should accept the args without a usage error.
    result = runner.invoke(app, ["provenance", "page-a", "page-b", "page-c"])
    assert result.exit_code != 0  # no wiki → error, but NOT a usage/arg error
    assert "Got unexpected extra argument" not in result.output


def test_write_command_minimal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kompend write creates a file (or errors gracefully)."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["write", "Test Article", "--frontmatter", "lang=en"])
    assert result.exit_code in (0, 1)
    if result.exit_code == 0:
        assert "Written" in result.stdout
        result_file = tmp_path / "wiki" / "write" / "test-article.md"
        assert result_file.exists()
        content = result_file.read_text(encoding="utf-8")
        assert "lang: en" in content
