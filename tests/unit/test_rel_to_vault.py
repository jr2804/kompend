"""Tests for the ``rel_to_vault`` relativization helper and the
``check_persisted_paths`` enforcement scanner.

These guard Antipattern 6 (``src/kompend/AGENTS.md``): no absolute path may
be persisted inside a ``.kompend/`` artifact.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from kompend.core.paths import rel_to_vault


def _load_scanner() -> Any:
    """Import the check script as a module (it's not on the package path).

    The script imports its sibling ``_common``, so the checks directory must be
    importable while it executes — the same requirement :func:`main` has when
    the gate runs from the command line.
    """
    checks_dir = Path(__file__).resolve().parents[2] / "scripts" / "checks"
    script = checks_dir / "check_persisted_paths.py"
    spec = importlib.util.spec_from_file_location("check_persisted_paths", script)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered before execution: a dataclass reads sys.modules[cls.__module__]
    # while the class body is being processed, and fails on a missing entry.
    sys.modules["check_persisted_paths"] = mod
    sys.path.insert(0, str(checks_dir))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


_SCANNER = _load_scanner()

# ── rel_to_vault ───────────────────────────────────────────────────────


class TestRelToVault:
    """``rel_to_vault`` strips the vault prefix and POSIX-normalizes."""

    def test_absolute_posix_path_stripped(self, tmp_path: Path) -> None:
        base = tmp_path
        p = base / "sources" / "paper.pdf"
        assert rel_to_vault(p, base) == "sources/paper.pdf"

    def test_absolute_windows_path_stripped(self, tmp_path: Path) -> None:
        base = tmp_path / "vault"
        base.mkdir()
        p = base / "sources" / "doc.pdf"
        # as_posix() must produce forward slashes regardless of platform.
        assert rel_to_vault(p, base) == "sources/doc.pdf"

    def test_already_relative_returned_as_posix(self, tmp_path: Path) -> None:
        # A path that's already relative is returned POSIX-normalized.
        assert rel_to_vault(Path("sources/paper.pdf"), tmp_path) == "sources/paper.pdf"

    def test_backslash_path_normalized_to_posix(self, tmp_path: Path) -> None:
        """A backslash means a separator on Windows and a filename character on POSIX.

        ``pathlib`` reads the path per platform, and ``rel_to_vault`` returns
        the platform's own POSIX form of it — so the expectation differs by
        platform rather than the helper being wrong on one of them.
        """
        raw = "sources\\sub\\paper.pdf"
        assert rel_to_vault(Path(raw), tmp_path) == Path(raw).as_posix()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX: a backslash is a legal filename character")
    def test_backslash_is_a_filename_character_on_posix(self, tmp_path: Path) -> None:
        r"""On POSIX the backslash is part of the name, and must survive verbatim.

        Rewriting it would rename a real file — ``a\b.pdf`` is one file on
        POSIX, not a directory ``a`` holding ``b.pdf``.
        """
        assert rel_to_vault(Path("sources\\sub\\paper.pdf"), tmp_path) == "sources\\sub\\paper.pdf"

    def test_outside_vault_returned_as_posix(self, tmp_path: Path) -> None:
        """A path not under base_dir degrades to its POSIX form (no crash)."""
        outside = tmp_path.parent / "other" / "file.log"
        result = rel_to_vault(outside, tmp_path)
        # Not vault-relative: the full outside path is returned POSIX-normalized.
        assert result == outside.as_posix()
        assert result not in {"other/file.log", "file.log"}

    def test_accepts_string_input(self, tmp_path: Path) -> None:
        base = tmp_path
        assert rel_to_vault(str(base / "sources" / "x.md"), base) == "sources/x.md"

    def test_empty_string_returns_empty(self, tmp_path: Path) -> None:
        assert rel_to_vault("", tmp_path) == ""

    def test_none_source_file_sentinel(self, tmp_path: Path) -> None:
        """The 'unknown' sentinel used when source_file is None passes through."""
        assert rel_to_vault("unknown", tmp_path) == "unknown"

    def test_deeply_nested(self, tmp_path: Path) -> None:
        base = tmp_path
        p = base / "a" / "b" / "c" / "d" / "file.txt"
        assert rel_to_vault(p, base) == "a/b/c/d/file.txt"


# ── check_persisted_paths scanner ──────────────────────────────────────


# ── check_persisted_paths scanner ──────────────────────────────────────


class TestLooksAbsolute:
    """The ``_looks_absolute`` heuristic must catch real paths and spare benign strings."""

    def test_posix_absolute_caught(self) -> None:
        assert _SCANNER._looks_absolute("/home/user/file.pdf")
        assert _SCANNER._looks_absolute("/var/log/x")

    def test_windows_drive_caught(self) -> None:
        assert _SCANNER._looks_absolute("C:\\Users\\jan\\file.pdf")
        assert _SCANNER._looks_absolute("D:/svn/AI/kompend/sources/x.pdf")

    def test_relative_path_spared(self) -> None:
        assert not _SCANNER._looks_absolute("sources/paper.pdf")
        assert not _SCANNER._looks_absolute("paper.pdf")

    def test_short_string_spared(self) -> None:
        assert not _SCANNER._looks_absolute("/")
        assert not _SCANNER._looks_absolute("ab")

    def test_url_spared(self) -> None:
        """An http(s) URL is not flagged — only filesystem paths."""
        assert not _SCANNER._looks_absolute("https://example.com/page")

    def test_markdown_link_spared(self) -> None:
        assert not _SCANNER._looks_absolute("[paper](../sources/paper.pdf)")


class TestScanVault:
    """End-to-end scan of a fake ``.kompend/`` tree."""

    def test_clean_vault_no_violations(self, tmp_path: Path) -> None:
        vault = self._make_vault(tmp_path)
        (vault / ".kompend" / "references.json").write_text(
            json.dumps({"citing_source": "sources/paper.pdf"}),
            encoding="utf-8",
        )
        assert _SCANNER.scan_vault(vault) == []

    def test_references_json_leak_detected(self, tmp_path: Path) -> None:
        vault = self._make_vault(tmp_path)
        (vault / ".kompend" / "references.json").write_text(
            json.dumps({"citing_source": "/home/jan/vault/sources/paper.pdf"}),
            encoding="utf-8",
        )
        violations = _SCANNER.scan_vault(vault)
        assert len(violations) == 1
        assert violations[0].file == ".kompend/references.json"
        assert "citing_source" in violations[0].field_path

    def test_nested_leak_in_wiki_cache(self, tmp_path: Path) -> None:
        vault = self._make_vault(tmp_path)
        (vault / ".kompend" / "last_wiki.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {"id": "n1", "metadata": {"source_file": "C:\\Users\\jan\\vault\\sources\\x.pdf"}},
                    ],
                },
            ),
            encoding="utf-8",
        )
        violations = _SCANNER.scan_vault(vault)
        assert len(violations) == 1
        assert "source_file" in violations[0].field_path

    def test_jsonl_leak_detected(self, tmp_path: Path) -> None:
        vault = self._make_vault(tmp_path)
        (vault / ".kompend" / "runs.jsonl").write_text(
            json.dumps({"source": "/var/data/file.pdf"}) + "\n",
            encoding="utf-8",
        )
        violations = _SCANNER.scan_vault(vault)
        assert len(violations) == 1
        assert "L1" in violations[0].field_path

    def test_missing_kompend_dir_returns_empty(self, tmp_path: Path) -> None:
        # No .kompend folder at all.
        assert _SCANNER.scan_vault(tmp_path) == []

    def test_binary_files_skipped(self, tmp_path: Path) -> None:
        vault = self._make_vault(tmp_path)
        # A .db file is not scanned even if it contains an absolute path.
        (vault / ".kompend" / "memory.db").write_bytes(b"/home/jan/leak.pdf")
        assert _SCANNER.scan_vault(vault) == []

    def test_main_exits_nonzero_on_violation(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        vault = self._make_vault(tmp_path)
        (vault / ".kompend" / "assets.json").write_text(
            json.dumps({"source": "/opt/vault/sources/x.pdf"}),
            encoding="utf-8",
        )
        rc = _SCANNER.main(["--vault", str(vault)])
        assert rc == 1
        captured = capsys.readouterr()
        assert "absolute-path-leak" in captured.out

    def test_main_exits_zero_when_clean(
        self,
        tmp_path: Path,
    ) -> None:
        vault = self._make_vault(tmp_path)
        (vault / ".kompend" / "assets.json").write_text(
            json.dumps({"source": "sources/x.pdf"}),
            encoding="utf-8",
        )
        assert _SCANNER.main(["--vault", str(vault)]) == 0

    def _make_vault(self, tmp_path: Path) -> Path:
        kompend = tmp_path / ".kompend"
        kompend.mkdir()
        return tmp_path


class TestSqliteStore:
    """The vault's durable artifacts live in ``kompend.db``, so the scan must too.

    A JSON-only scanner was green forever after the store migration: zero
    artifacts read, exit 0. These tests pin that it reads the store and refuses
    a vacuous pass.
    """

    def test_a_leak_in_a_store_column_is_found(self, tmp_path: Path) -> None:
        vault = self._make_vault(tmp_path)
        self._seed_store(
            vault,
            "CREATE TABLE documents (cache_key TEXT, source_file TEXT)",
            "INSERT INTO documents VALUES ('k1', 'D:/vault/sources/x.pdf')",
        )

        violations = _SCANNER.scan_vault(vault)

        assert len(violations) == 1
        assert violations[0].file == "kompend.db"
        assert "documents" in violations[0].field_path
        assert "source_file" in violations[0].field_path

    def test_a_leak_nested_in_a_json_payload_column_is_found(self, tmp_path: Path) -> None:
        vault = self._make_vault(tmp_path)
        payload = json.dumps([{"metadata": {"image_file": "/home/jan/fig.png"}}])
        self._seed_store(
            vault,
            "CREATE TABLE documents (cache_key TEXT, elements TEXT)",
            ("INSERT INTO documents VALUES (?, ?)", ("k2", payload)),
        )

        violations = _SCANNER.scan_vault(vault)

        assert len(violations) == 1
        assert "elements[0].metadata.image_file" in violations[0].field_path

    def test_relative_paths_in_the_store_are_clean(self, tmp_path: Path) -> None:
        vault = self._make_vault(tmp_path)
        self._seed_store(
            vault,
            "CREATE TABLE documents (cache_key TEXT, source_file TEXT)",
            "INSERT INTO documents VALUES ('k3', 'sources/ok.pdf')",
        )

        assert _SCANNER.scan_vault(vault) == []
        assert _SCANNER.main(["--vault", str(vault)]) == 0

    def test_an_empty_vault_is_not_a_pass(self, tmp_path: Path) -> None:
        """Nothing scanned means nothing proved — exit 2, not 0."""
        vault = self._make_vault(tmp_path)

        assert _SCANNER.main(["--vault", str(vault)]) == 2

    def test_a_non_database_db_file_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """A stray ``.db`` with no schema carries nothing to scan."""
        vault = self._make_vault(tmp_path)
        (vault / ".kompend" / "memory.db").write_bytes(b"not a database")
        (vault / ".kompend" / "assets.json").write_text(json.dumps({"source": "sources/x.pdf"}), encoding="utf-8")

        assert _SCANNER.main(["--vault", str(vault)]) == 0

    def _seed_store(self, vault: Path, *statements: tuple[str, tuple[str, ...]] | str) -> None:
        """Create a ``kompend.db`` with *statements* executed in order."""
        conn = sqlite3.connect(vault / ".kompend" / "kompend.db")
        try:
            for statement in statements:
                if isinstance(statement, tuple):
                    sql, params = statement
                    conn.execute(sql, params)
                else:
                    conn.execute(statement)
            conn.commit()
        finally:
            conn.close()

    def _make_vault(self, tmp_path: Path) -> Path:
        kompend = tmp_path / ".kompend"
        kompend.mkdir()
        return tmp_path
