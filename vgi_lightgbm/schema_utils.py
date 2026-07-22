"""Shared Arrow-schema helpers for the LightGBM worker.

Keeps column-comment plumbing and name sanitisation in one place so every
dataset/model function exposes consistent, documented schemas to DuckDB.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import pyarrow as pa


def field(
    name: str,
    type: pa.DataType,
    comment: str,
    *,
    nullable: bool = True,
) -> pa.Field:
    """Build a ``pa.Field`` carrying a column comment in its metadata.

    The ``comment`` metadata key is the framework's transport for column
    comments -- DuckDB surfaces it via ``duckdb_columns()`` and ``DESCRIBE``.
    """
    return pa.field(
        name,
        type,
        nullable=nullable,
        metadata={b"comment": comment.encode("utf-8")},
    )


def _sql_type(t: pa.DataType) -> str:
    """Map a common Arrow type to a readable DuckDB-ish type name."""
    if pa.types.is_int64(t):
        return "BIGINT"
    if pa.types.is_int32(t):
        return "INTEGER"
    if pa.types.is_int16(t):
        return "SMALLINT"
    if pa.types.is_int8(t):
        return "TINYINT"
    if pa.types.is_float64(t):
        return "DOUBLE"
    if pa.types.is_float32(t):
        return "FLOAT"
    if pa.types.is_string(t):
        return "VARCHAR"
    if pa.types.is_boolean(t):
        return "BOOLEAN"
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return "BLOB"
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return f"{_sql_type(t.value_type)}[]"
    # A dictionary-encoded column is just its value type on the DuckDB side --
    # the encoding is a transport detail, not part of the SQL schema.
    if pa.types.is_dictionary(t):
        return _sql_type(t.value_type)
    return str(t)


def _field_comment(f: pa.Field) -> str:
    """Read a field's ``comment`` metadata (empty string when absent)."""
    if f.metadata and b"comment" in f.metadata:
        return str(f.metadata[b"comment"].decode("utf-8"))
    return ""


def result_columns_schema(schema: pa.Schema) -> str:
    """Render a table function's *static* RETURN columns as ``vgi.result_columns_schema``.

    DuckDB cannot expose a VGI table-function result schema up front, so the
    worker declares it: a JSON array of ``{name, type, description}``, one entry
    per returned column, with the type rendered as a real DuckDB type name and the
    description read from the field's ``comment`` metadata. Use this only when the
    schema does not depend on the call's arguments; otherwise see
    :func:`result_dynamic_columns_md`.
    """
    return json.dumps([{"name": f.name, "type": _sql_type(f.type), "description": _field_comment(f)} for f in schema])


def result_columns_schema_rows(rows: list[tuple[str, str, str]]) -> str:
    """Same as :func:`result_columns_schema`, from explicit ``(name, type, description)`` rows."""
    return json.dumps([{"name": n, "type": t, "description": d} for n, t, d in rows])


def result_dynamic_columns_md(
    variants: list[tuple[str, list[tuple[str, str, str]]]],
    *,
    note: str,
) -> str:
    """Render ``vgi.result_dynamic_columns_md`` for a schema that varies by argument.

    ``note`` explains *how* the schema varies; ``variants`` is a list of
    ``(caption, rows)`` pairs, one Markdown ``Name | Type | Description`` table per
    result-shape variant (the caption becomes the heading that says when that
    variant applies).
    """
    parts = [note]
    for caption, rows in variants:
        block = [f"#### {caption}", "", "| Name | Type | Description |", "| --- | --- | --- |"]
        block += [f"| `{n}` | {t} | {d} |" for n, t, d in rows]
        parts.append("\n".join(block))
    return "\n\n".join(parts)


_NON_IDENT = re.compile(r"[^0-9a-z]+")


def snake_case(name: str) -> str:
    """Normalise a feature label to a SQL-friendly column name.

    ``"sepal length (cm)"`` -> ``"sepal_length_cm"``. Collapses any run of
    non-alphanumeric characters to a single underscore and lowercases.
    """
    cleaned = _NON_IDENT.sub("_", name.strip().lower()).strip("_")
    if not cleaned:
        return "feature"
    if cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    return cleaned


def dedupe_names(names: list[str]) -> list[str]:
    """Ensure column names are unique by suffixing collisions (``_2``, ``_3`` ...)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
    return out


@dataclass(slots=True, frozen=True, kw_only=True)
class NoArgs:
    """Empty argument set for functions that take no user-facing parameters."""
