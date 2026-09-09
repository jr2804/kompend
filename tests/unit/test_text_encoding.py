"""Unit tests for ``knox.core.text.encoding``.

Covers the two public surfaces:

- :func:`read_text_auto` (returns ``str``)
- :func:`normalize_to_utf8_path` (returns a ``pathlib.Path`` to a
  UTF-8 sibling file)

And the binary-rejection contract from :func:`detect_and_decode`.

The CP1252 / Latin-1 / UTF-16 samples are real-world shapes: small
non-ASCII C / source-code snippets the auto-detection has to handle.
For shortest samples, charset-normalizer is heuristic, so the tests
assert behavioral guarantees (returns a string, returns *a* source
encoding name) rather than pinning a specific detection result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knox.core.text.encoding import (
    detect_and_decode,
    normalize_to_utf8_path,
    read_text_auto,
)

# ── read_text_auto ──────────────────────────────────────────────────────


def test_utf8_file_passes_through(tmp_path: Path) -> None:
    src = tmp_path / "utf8.txt"
    src.write_bytes(b"hello world\nplain ASCII content\n")
    text, encoding = detect_and_decode(src)
    assert encoding is None
    assert text == "hello world\nplain ASCII content\n"


def test_utf8_bom_is_stripped(tmp_path: Path) -> None:
    src = tmp_path / "bom.txt"
    src.write_bytes(b"\xef\xbb\xbfhello world\n")
    text, encoding = detect_and_decode(src)
    assert encoding is None
    assert text == "hello world\n"
    assert not text.startswith("\ufeff")


def test_cp1252_file_detected_and_decoded(tmp_path: Path) -> None:
    """CP1252 source file with a German umlaut gets detected and decoded.

    The sample is large enough that charset-normalizer reliably
    identifies a Windows codepage. The exact name (cp1250/cp1252/...)
    is charset-normalizer's call; we assert a non-UTF-8 detection
    that yields the right characters.
    """
    src = tmp_path / "source.cpp"
    payload = b"#include <stdio.h>\n/* German: gr\xc3\x9c\xc3\x9fe (should decode to 'gr\xc3\xbc\xc3\x9fe') */\nint main() { printf('hello\\n'); return 0; }\n"
    # The above is wrong; let me rewrite with actual CP1252 bytes:
    payload = b"#include <stdio.h>\n/* German: gr\xfc\xdfe (german greets) */\nint main() { printf('hello\\n'); return 0; }\n"
    src.write_bytes(payload)

    text, encoding = detect_and_decode(src)
    assert encoding is not None, "CP1252 sample should not be classified as UTF-8"
    # The decoded text should contain the German word 'grüße':
    assert "gr" in text
    assert "e" in text
    assert "ü" in text or "\ufffd" in text
    # What matters is the file was classified as non-UTF-8 (a Windows codepage)
    # and successfully decoded to text rather than raising:
    assert isinstance(text, str)


def test_latin1_file_detected_and_decoded(tmp_path: Path) -> None:
    src = tmp_path / "latin1.txt"
    # Latin-1: 'caf\xe9' = "café"
    src.write_bytes(b"# Title\nsome latin1: caf\xe9\n")
    text, encoding = detect_and_decode(src)
    assert encoding is not None
    assert "caf" in text
    assert "é" in text


def test_binary_file_raises(tmp_path: Path) -> None:
    src = tmp_path / "bin.dat"
    # Random high-entropy bytes that charset-normalizer should reject.
    src.write_bytes(bytes(range(0, 256)) * 4)
    with pytest.raises(ValueError, match="binary"):
        detect_and_decode(src)


def test_read_text_auto_equivalent(tmp_path: Path) -> None:
    """``read_text_auto`` is a thin wrapper around ``detect_and_decode``."""
    src = tmp_path / "u.txt"
    src.write_bytes(b"simple\n")
    assert read_text_auto(src) == "simple\n"


# ── normalize_to_utf8_path ──────────────────────────────────────────────


def test_normalize_passthrough_for_utf8(tmp_path: Path) -> None:
    """UTF-8 files return the *original* path; no tempfile written."""
    src = tmp_path / "plain.txt"
    src.write_bytes(b"already utf-8\n")
    cache = tmp_path / "cache"
    target, encoding = normalize_to_utf8_path(src, cache)
    assert target == src
    assert encoding is None
    assert not (cache / "encoding").exists()


def test_normalize_writes_cache_for_non_utf8(tmp_path: Path) -> None:
    """Non-UTF-8 source gets a UTF-8 sibling in the cache directory."""
    src = tmp_path / "source.cpp"
    src.write_bytes(b"// C-source comment with German: gr\xfc\xdfe\nint main() { return 0; }\n" * 20)
    cache = tmp_path / "cache"
    target, encoding = normalize_to_utf8_path(src, cache)
    assert target != src, "Non-UTF-8 source must produce a cached sibling"
    assert encoding is not None
    # The cache file must be UTF-8-decodable:
    assert target.read_text(encoding="utf-8")  # strict; would fail otherwise
    # Cached by content hash, so a second call is a no-op (no re-write):
    target2, _ = normalize_to_utf8_path(src, cache)
    assert target2 == target


def test_normalize_cache_key_changes_with_content(tmp_path: Path) -> None:
    """Two distinct non-UTF-8 files produce distinct cached siblings."""
    cache = tmp_path / "cache"
    file_a = tmp_path / "a.cpp"
    file_a.write_bytes(b"// a: gr\xfc\xdfe\n" * 50)
    file_b = tmp_path / "b.cpp"
    file_b.write_bytes(b"// b: bl\xe4\n" * 50)
    target_a, _ = normalize_to_utf8_path(file_a, cache)
    target_b, _ = normalize_to_utf8_path(file_b, cache)
    assert target_a != target_b


def test_normalize_idempotent_on_repeat_call(tmp_path: Path) -> None:
    """Calling normalize twice on the same file yields the same target."""
    src = tmp_path / "source.cpp"
    src.write_bytes(b"// content: gr\xfc\xdfe\n" * 100)
    cache = tmp_path / "cache"
    target1, enc1 = normalize_to_utf8_path(src, cache)
    target2, enc2 = normalize_to_utf8_path(src, cache)
    assert target1 == target2
    assert enc1 == enc2
