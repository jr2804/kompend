"""The capture vocabulary is one set of names, and the names do not drift.

Every value here is a wire contract: it is written into OKF frontmatter that
users commit, into archive directory names on disk, and into the prompt the
capture agent reads. The point of the module under test is that those spellings
live in one place — so the tests below pin both the spellings themselves and
the places that consume them.

The prompt test is the important one: it is what stops a prompt edit from
inventing an extraction kind the wiki has never heard of, or dropping one the
writer can no longer emit.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from kompend.capture.capture import capture
from kompend.capture.consolidate import consolidate
from kompend.capture.distill import distill
from kompend.capture.kinds import (
    CAPTURE_SOURCE_ID,
    UNGROUPED_TAG,
    CaptureKind,
    SourceZone,
)
from kompend.capture.learn import learn
from kompend.cli.args.common import CompletionShell, shell_name
from kompend.core.models.elements import element_text
from kompend.core.models.enums import MetadataField
from kompend.core.paths import (
    ARCHIVE_SUBDIR,
    CAPTURED_SUBDIR,
    DISTILLED_SUBDIR,
    LEARNED_SUBDIR,
    VaultPaths,
    run_log_file_name,
)
from kompend.core.templates import get_template_loader
from kompend.output.frontmatter import okf_frontmatter_yaml, source_frontmatter
from kompend.output.reference_links import _ref_link

PROMPT_TEMPLATE = "capture/capture_system.md.j2"


class TestWireValuesAreUnchanged:
    """These strings are on disk in users' vaults; renaming one is a migration."""

    def test_zones(self) -> None:
        assert [member.value for member in SourceZone] == [
            "capture",
            "learn",
            "manual",
            "distilled",
            "unknown",
        ]

    def test_capture_kinds(self) -> None:
        assert [member.value for member in CaptureKind] == [
            "decision",
            "discovery",
            "correction",
            "reference",
            "note",
        ]

    def test_scalars(self) -> None:
        assert UNGROUPED_TAG == "general"
        assert CAPTURE_SOURCE_ID == "capture"

    def test_metadata_keys(self) -> None:
        assert MetadataField.CAPTURED_AT.value == "captured_at"
        assert MetadataField.DISTILLED_AT.value == "distilled_at"
        assert MetadataField.SYNTHESIZED_FROM.value == "synthesized_from"


class TestPromptOffersExactlyTheKinds:
    """The prompt and :class:`CaptureKind` name the same five kinds."""

    def test_prompt_lists_every_kind(self) -> None:
        text = get_template_loader().get_text(PROMPT_TEMPLATE)
        listed = set(re.findall(r"^- `([a-z_]+)` — ", text, flags=re.MULTILINE))
        assert listed == {member.value for member in CaptureKind}

    def test_the_example_uses_a_real_kind(self) -> None:
        text = get_template_loader().get_text(PROMPT_TEMPLATE)
        for example in re.findall(r'"kind": "([a-z_]+)"', text):
            assert example in CaptureKind.__members__.values()


class TestKindMapping:
    def test_known_kinds_round_trip(self) -> None:
        for member in CaptureKind:
            assert CaptureKind.of(member.value) is member

    def test_unknown_kind_falls_back_to_note(self) -> None:
        assert CaptureKind.of("brainwave") is CaptureKind.NOTE
        assert CaptureKind.of(None) is CaptureKind.NOTE
        assert CaptureKind.of("  ") is CaptureKind.NOTE

    def test_matching_ignores_case_and_padding(self) -> None:
        assert CaptureKind.of(" Decision ") is CaptureKind.DECISION


class TestZoneMapping:
    def test_of_path_reads_the_source_zone(self) -> None:
        assert SourceZone.of_path(Path("/v") / "sources" / CAPTURED_SUBDIR / "a.md") is SourceZone.CAPTURE
        assert SourceZone.of_path(Path("/v") / "sources" / LEARNED_SUBDIR / "a.md") is SourceZone.LEARN
        assert SourceZone.of_path(Path("/v") / "sources" / DISTILLED_SUBDIR / "a.md") is SourceZone.DISTILLED

    def test_of_path_reports_unknown_outside_the_zones(self) -> None:
        assert SourceZone.of_path(Path("/v") / "sources" / "free.md") is SourceZone.UNKNOWN
        assert SourceZone.of_path(None) is SourceZone.UNKNOWN

    def test_archive_names_are_the_disk_contract(self) -> None:
        assert SourceZone.CAPTURE.archive_subdir == "capture"
        assert SourceZone.LEARN.archive_subdir == "learn"
        assert SourceZone.DISTILLED.archive_subdir == "distilled"
        assert SourceZone.UNKNOWN.archive_subdir == "unknown"

    def test_source_dirs_keep_their_underscore(self) -> None:
        assert SourceZone.CAPTURE.source_subdir == CAPTURED_SUBDIR
        assert SourceZone.LEARN.source_subdir == LEARNED_SUBDIR
        assert SourceZone.DISTILLED.source_subdir == DISTILLED_SUBDIR

    def test_manual_notes_live_in_the_learned_zone(self) -> None:
        assert SourceZone.MANUAL.source_subdir == LEARNED_SUBDIR


class TestFrontmatterCarriesTheVocabulary:
    """End-to-end: the values the writers emit are the vocabulary's values."""

    def test_capture_writes_kind_and_zone(self, tmp_path: Path) -> None:
        meta = _meta(capture("Something captured.", sources_dir=tmp_path)[0])
        assert meta["type"] == CaptureKind.NOTE.value
        assert meta["source"] == SourceZone.CAPTURE.value
        assert MetadataField.CAPTURED_AT.value in meta

    def test_learn_writes_type_and_manual_provenance(self, tmp_path: Path) -> None:
        meta = _meta(learn("A fact worth keeping.", sources_dir=tmp_path))
        assert meta["type"] == SourceZone.LEARN.value
        assert meta["source"] == SourceZone.MANUAL.value

    def test_distill_writes_type_and_provenance(self, tmp_path: Path) -> None:
        captured = tmp_path / CAPTURED_SUBDIR
        captured.mkdir(parents=True)
        (captured / "note-a.md").write_text("---\ntitle: a\ntags:\n  - alpha\n---\n\nAlpha body.\n", encoding="utf-8")
        page = distill(tmp_path, archive=False)[0]
        meta = _meta(page)
        assert meta["type"] == SourceZone.DISTILLED.value
        assert MetadataField.DISTILLED_AT.value in meta
        assert meta[MetadataField.SYNTHESIZED_FROM.value] == ["note-a.md"]

    def test_archive_directories_match_the_zone_names(self, tmp_path: Path) -> None:
        body = "The PESQ algorithm aligns utterances before scoring."
        captured = tmp_path / CAPTURED_SUBDIR
        captured.mkdir(parents=True)
        for name in ("a.md", "b.md"):
            (captured / name).write_text(f"---\ntitle: {name}\n---\n\n{body}\n", encoding="utf-8")
        groups = consolidate(tmp_path)
        assert len(groups) == 1
        assert (tmp_path / ARCHIVE_SUBDIR / SourceZone.CAPTURE.value / "b.md").exists()


class TestFrontmatterRendersMembersAsPlainStrings:
    """A StrEnum reaching yaml.dump becomes a python/object tag no reader loads."""

    def test_page_type_member_renders_as_its_value(self) -> None:
        block = okf_frontmatter_yaml(
            source_frontmatter(
                page_type=SourceZone.LEARN,
                timestamp="20260101T000000",
                timestamp_key=MetadataField.CAPTURED_AT,
                tags=[],
            )
        )
        assert "!!python/object" not in block
        assert yaml.safe_load(block[4:].split("\n---\n", 1)[0])["type"] == "learn"

    def test_enum_valued_extra_renders_as_its_value(self) -> None:
        block = okf_frontmatter_yaml(
            source_frontmatter(
                page_type=SourceZone.CAPTURE,
                timestamp="20260101T000000",
                timestamp_key=MetadataField.CAPTURED_AT,
                tags=[],
                extra={MetadataField.SOURCE: SourceZone.CAPTURE},
            )
        )
        assert yaml.safe_load(block[4:].split("\n---\n", 1)[0])["source"] == "capture"


class TestTemplateRenderers:
    """The four hand-formatted strings are functions now, and keep their bytes."""

    def test_binary_placeholder(self) -> None:
        assert element_text(b"\x89PNG") == "[binary content: 4 bytes, not text]"

    def test_run_log_file_name(self) -> None:
        assert run_log_file_name("abc") == "run-abc.log"
        assert VaultPaths(Path("/v")).run_log_file("abc").name == "run-abc.log"

    def test_reference_link(self) -> None:
        assert _ref_link("Smith 2020", "smith-2020") == "[Smith 2020](#smith-2020)"


class TestCompletionShell:
    def test_values_match_the_shells_cyclopts_accepts(self) -> None:
        assert {member.value for member in CompletionShell} == {"zsh", "bash", "fish", "powershell"}

    def test_narrowing_keeps_the_value_and_preserves_none(self) -> None:
        assert shell_name(CompletionShell.BASH) == "bash"
        assert shell_name(None) is None


def _meta(path: Path) -> dict:
    content = path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    return yaml.safe_load(content[4:].split("\n---\n", 1)[0])
