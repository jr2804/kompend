"""Statement construction in the sqlite-vec family and its two root modules.

These modules are hand-written SQL because ``vec0``/FTS5 are virtual tables with
no SQLAlchemy Core object. Each statement is therefore a constant template that
:func:`kompend.rag.sqlite_vec.schema.sql` fills, and the module-wide
``# ruff: noqa: S608`` waivers that used to cover them are gone: a waiver over a
whole module hides every later hand-built statement from bandit, which is the
regression this file guards against.

The behaviour checks pin the two facts a template refactor can silently break:
the filter clause still renders one ``EXISTS`` per required value with its
parameters in the same order, and the lexical query refuses a column the FTS
index does not hold instead of running a query that matches nothing.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

import kompend.trust as trust_module
import kompend.verification as verification_module
from kompend.rag.sqlite_vec import filters, index, manifest, query, schema
from kompend.rag.sqlite_vec.filters import filter_clause
from kompend.rag.sqlite_vec.schema import VEC_ATTR_KEYS, VEC_ATTRIBUTES, VEC_TAGS, sql
from kompend.rag.vector_store import VectorField

#: Every module that carried a module-wide S608 waiver in this family.
_WAIVERED_MODULES: tuple[ModuleType, ...] = (
    filters,
    index,
    manifest,
    query,
    schema,
    trust_module,
    verification_module,
)


class TestFilterClause:
    """The clause is still one ``EXISTS`` per value, parameters in call order."""

    def test_no_filters_render_nothing(self) -> None:
        assert filter_clause("v") == ("", [])

    def test_tags_all_renders_one_exists_per_tag(self) -> None:
        clause, params = filter_clause("v", tags_all=["x", "y"])
        assert clause.startswith("AND EXISTS")
        assert clause.count("EXISTS") == 2
        assert VEC_TAGS in clause
        assert "t.item_id = v.id" in clause
        assert params == ["x", "y"]

    def test_tags_dnf_groups_are_or_joined(self) -> None:
        clause, params = filter_clause("v", tags_dnf=[["x", "y"], ["z"]])
        assert clause.count("EXISTS") == 3
        assert " OR " in clause
        assert params == ["x", "y", "z"]

    def test_an_empty_dnf_group_is_skipped(self) -> None:
        clause, params = filter_clause("v", tags_dnf=[[], ["z"]])
        assert clause.count("EXISTS") == 1
        assert params == ["z"]

    def test_attribute_filters_name_their_side_tables(self) -> None:
        clause, params = filter_clause("f", attributes_all=["lang=rust"], attribute_keys=["lang"])
        assert VEC_ATTRIBUTES in clause
        assert VEC_ATTR_KEYS in clause
        assert "a.item_id = f.id" in clause
        assert params == ["lang=rust", "lang"]

    def test_values_are_parameters_never_sql_text(self) -> None:
        """A hostile value must not reach the statement."""
        clause, params = filter_clause("v", tags_all=["x'); DROP TABLE vec_items; --"])
        assert "DROP TABLE" not in clause
        assert params == ["x'); DROP TABLE vec_items; --"]


class TestFtsColumn:
    """The lexical query refuses a column the FTS index does not hold."""

    @pytest.mark.parametrize("column", [VectorField.TITLE, VectorField.CONTENT_PREVIEW])
    def test_indexed_columns_are_accepted(self, column: VectorField) -> None:
        assert column.value in query._FTS_COLUMN_NAMES

    def test_an_unindexed_column_is_refused_before_the_query_runs(self, connection: sqlite3.Connection) -> None:
        """A `MATCH` against a column the index lacks silently matches nothing."""
        with pytest.raises(ValueError, match="FTS column must be one of"):
            query.fts_ranked(connection, query="q", top_k=1, column="source_file")

    def test_an_empty_query_short_circuits(self, connection: sqlite3.Connection) -> None:
        """No terms means no statement at all — the store answers ``[]``."""
        assert query.fts_ranked(connection, query="---", top_k=1) == []


@pytest.fixture
def connection() -> Iterator[sqlite3.Connection]:
    """A connection that is never queried — the guards run before any SQL."""
    conn = sqlite3.connect(":memory:")
    try:
        yield conn
    finally:
        conn.close()


@pytest.mark.parametrize("module", _WAIVERED_MODULES, ids=lambda m: Path(module_file(m)).name)
def test_the_module_carries_no_ruff_waiver(module: ModuleType) -> None:
    """The conversion is the point: the waiver must not come back.

    A module-wide waiver silences bandit for the file, so any later statement
    built from an outside string lands unnoticed. The statements are templates
    filled in one place instead, and ``ruff check`` on each file is clean.
    """
    source = Path(module_file(module)).read_text(encoding="utf-8")
    assert "ruff: noqa" not in source
    assert "noqa: S608" not in source


def module_file(module: ModuleType) -> str:
    """The source file of a module — every module here is imported from disk."""
    path = module.__file__
    assert path is not None, f"{module.__name__} has no source file"
    return path


def test_sql_fills_only_the_named_placeholders() -> None:
    """``sql`` is a plain fill: a missing value fails loudly instead of leaking a brace."""
    assert sql("SELECT {column} FROM {table}", column="id", table="t") == "SELECT id FROM t"
    with pytest.raises(KeyError):
        sql("SELECT {column} FROM {table}", column="id")
