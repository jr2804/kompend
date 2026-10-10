"""Single source of truth for text-file encoding detection and normalization.

Every place in the codebase that reads a user-supplied text file as a
string — or hands a text file to a parser that needs a path — routes
through this module. Centralizing here prevents the same file from
being mojibake-decoded in one place and silently replaced in another.

Two public surfaces:

- :func:`read_text_auto` — returns ``str`` for in-process consumers
  (capture, the attach-ment parse path). Drops the BOM and
  replaces undecodable bytes with U+FFFD on detection.

- :func:`normalize_to_utf8_path` — returns a :class:`pathlib.Path`
  pointing at a UTF-8 sibling of the original file for pipeline stages that
  hand text-family sources to the parser (the knox-side binary-expecting
  parsers — pandoc, docling, marker, liteparse — are gone; parsecraft
  normalizes internally, but non-UTF-8 text sources still get a UTF-8 path
  before parse). Materializes a tempfile in the cache only when detection
  says non-UTF-8; UTF-8-clean files pass through unchanged.

Detection uses :mod:`charset-normalizer`. The dependency is declared as
a direct dep in ``pyproject.toml`` even though it was already
transitive via httpx/pypandoc — auto-conversion is contract, not
accident.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final

from charset_normalizer import from_bytes

from knox.logging import get_logger

#: Cache directory name (under ``.knox/cache/``) holding re-encoded text
#: siblings for non-UTF-8 sources. Distinct from the parse-cache so a
#: re-encode never invalidates parse-cache entries.
_ENCODING_CACHE_DIR: Final[str] = "encoding"

#: Length of the content-hash filename used in the encoding cache. 16 hex
#: chars (64 bits) matches ``cache/store.py``'s asset-id convention —
#: collision-free for any plausible vault size.
_CACHE_HASH_LEN: Final[int] = 16

#: UTF BOMs to strip on decode. ``str.encode("utf-8-sig")`` is not used
#: here because some sources legitimately contain UTF-16 BOMs mid-file;
#: we strip only at the file head.
_BOMS: Final[tuple[bytes, ...]] = (
    b"\xef\xbb\xbf",  # UTF-8 BOM
    b"\xff\xfe",  # UTF-16 LE BOM
    b"\xfe\xff",  # UTF-16 BE BOM
)

logger = get_logger(__name__)


def read_text_auto(source_path: Path) -> str:
    """Read *source_path* as text, auto-detecting its encoding.

    Thin wrapper around :func:`detect_and_decode` for string-returning
    consumers. Replaces the previous ``read_text(encoding="utf-8",
    errors="replace")`` calls — equivalent for UTF-8 inputs, correct
    for non-UTF-8 inputs.
    """
    text, _encoding = detect_and_decode(source_path)
    return text


def detect_and_decode(source_path: Path) -> tuple[str, str | None]:
    """Detect the encoding of *source_path* and return ``(text, encoding)``.

    - ``encoding is None``: file is valid UTF-8 (BOMs stripped if present).
    - ``encoding is str``: file was decoded via charset-normalizer's best
      guess; the value is the IANA name (e.g. ``"cp1252"``, ``"iso-8859-1"``).
    - Bytes charset-normalizer cannot fix are replaced with U+FFFD on
      decode (``charset-normalizer``'s ``candidates_best().output`` is
      already lossy; we pass it through verbatim).

    Binary files are detected by attempting strict UTF-8 first; if the
    file decodes as valid UTF-8 we accept it, otherwise we hand the
    bytes to charset-normalizer, which rejects truly binary inputs and
    raises ``ValueError`` for those via :func:`_assert_text`.
    """
    raw_bytes = source_path.read_bytes()
    text, encoding = _decode_bytes(raw_bytes)
    return text, encoding


def normalize_to_utf8_path(source_path: Path, cache_dir: Path) -> tuple[Path, str | None]:
    """Return a path to a UTF-8 copy of *source_path* for parsers needing a file.

    If *source_path* is already UTF-8, returns it unchanged with
    ``encoding=None``. Otherwise writes a normalized copy to
    ``cache_dir / "encoding" / <sha256>.utf8`` and returns that path
    with the detected encoding name.

    The cache key is the SHA-256 of the source bytes, truncated to
    :data:`_CACHE_HASH_LEN` — so a re-compile of an unchanged file
    hits the cache, and a changed file naturally re-encodes.

    Emits one log line per detected file via ``logger.info``.
    """
    raw_bytes = source_path.read_bytes()
    text, encoding = _decode_bytes(raw_bytes)
    if encoding is None:
        return source_path, None

    encoded_dir = cache_dir / _ENCODING_CACHE_DIR
    encoded_dir.mkdir(parents=True, exist_ok=True)
    cache_key = sha256(raw_bytes).hexdigest()[:_CACHE_HASH_LEN]
    target_path: Path = encoded_dir / f"{cache_key}.utf8"
    target_path.write_text(text, encoding="utf-8")
    logger.info(
        "encoding: %s was %s -> re-encoded UTF-8 (cache: %s)",
        source_path.name,
        encoding,
        target_path,
    )
    return target_path, encoding


def _decode_bytes(raw_bytes: bytes) -> tuple[str, str | None]:
    """Decode *raw_bytes*; return ``(text, encoding_or_None)``.

    Tries strict UTF-8 first. On ``UnicodeDecodeError`` delegates to
    charset-normalizer. Raises ``ValueError("binary")`` for inputs it
    can't classify as text.

    charset-normalizer's ``.output()`` returns silently-lossy bytes (it
    replaces undecodable bytes with ``?`` internally), so we re-decode
    the *original* ``raw_bytes`` with the detected encoding using
    ``errors='replace'`` — that keeps the user's bytes-to-characters
    mapping honest and surfaces U+FFFD only for genuinely unrecoverable
    bytes.
    """
    # Fast path: strict UTF-8 (covers ASCII + UTF-8 BOM).
    try:
        return _strip_bom(raw_bytes.decode("utf-8")), None
    except UnicodeDecodeError:
        pass

    # Fallback: charset-normalizer detects CP1252, Latin-1, etc.
    best = from_bytes(raw_bytes).best()
    if best is None or best.encoding is None:
        raise ValueError("binary")
    return _strip_bom(raw_bytes.decode(best.encoding, errors="replace")), best.encoding


def _strip_bom(text: str) -> str:
    """Strip a leading UTF-8/UTF-16 BOM that survived charset-normalizer.

    charset-normalizer's ``str()`` output drops UTF-8 BOMs but keeps
    UTF-16 BOMs as literal ``U+FEFF``. Either way, we strip a single
    leading BOM if present — anything beyond that is content.
    """
    return text[1:] if text.startswith("\ufeff") else text
