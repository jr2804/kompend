"""The tag/attribute filters, translated to parameterised SQL.

The canonical semantics live in :mod:`kompend.search.filters` — tags are an
**OR of ANDs** (DNF), attributes an **AND** of ``(key, value)`` checks. The
store evaluates the same rules as ``EXISTS`` subqueries over its side tables
so the filter runs per candidate in SQLite instead of in Python; a test pins
the two implementations to each other.

Every value travels as a bound parameter; a value never enters the SQL text.
Each fragment is a constant template whose only substitutions are the table
name constant and the caller's column alias, so the statement is auditable
without a module-wide waiver.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from kompend.rag.sqlite_vec.schema import VEC_ATTR_KEYS, VEC_ATTRIBUTES, VEC_TAGS, sql

#: One ``EXISTS`` per required value; ``alias`` is the correlated side of the
#: caller's query, ``column`` the side table's value column.
_EXISTS_BY_VALUE: Final[str] = "EXISTS (SELECT 1 FROM {table} {side} WHERE {side}.item_id = {alias}.id AND {side}.{column} = ?)"


def filter_clause(
    alias: str,
    *,
    tags_all: Sequence[str] | None = None,
    tags_dnf: Sequence[Sequence[str]] | None = None,
    attributes_all: Sequence[str] | None = None,
    attribute_keys: Sequence[str] | None = None,
) -> tuple[str, list[str]]:
    """Compose the SQL for the requested filters, with bound parameters.

    *alias* is the table alias the filter correlates against (``v`` for the
    vector query, ``f`` for the lexical one).

    ``tags_all``/``attributes_all``/``attribute_keys`` mean "the node has
    *every* one of these" — one ``EXISTS`` per value, ANDed.
    ``tags_dnf`` is an OR of ANDs (DNF): each group is a set of tags the node
    must all have, and the groups are OR-joined inside one parenthesised
    clause.

    Every value travels as a bound parameter — never as SQL text.
    """
    clauses: list[str] = []
    params: list[str] = []

    def exists(table: str, *, side: str, column: str) -> str:
        """One ``EXISTS`` against *table*, correlated to the caller's *alias*."""
        return sql(_EXISTS_BY_VALUE, table=table, side=side, column=column, alias=alias)

    for tag in tags_all or ():
        clauses.append(exists(VEC_TAGS, side="t", column="tag"))
        params.append(str(tag))

    groups: list[str] = []
    group_params: list[str] = []
    for group in tags_dnf or ():
        members = [str(tag) for tag in group]
        if not members:
            continue
        group_clauses = [exists(VEC_TAGS, side="t", column="tag") for _ in members]
        groups.append(" AND ".join(group_clauses))
        group_params.extend(members)
    if groups:
        clauses.append(f"({' OR '.join(groups)})")
        params.extend(group_params)

    for token in attributes_all or ():
        clauses.append(exists(VEC_ATTRIBUTES, side="a", column="token"))
        params.append(str(token))

    for key in attribute_keys or ():
        clauses.append(exists(VEC_ATTR_KEYS, side="k", column="key"))
        params.append(str(key))

    if not clauses:
        return "", []
    return "AND " + " AND ".join(clauses), params
