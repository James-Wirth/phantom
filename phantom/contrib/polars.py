"""
Polars operations for Phantom.

Provides DataFrame operations for LLM-driven data analysis using Polars.

Installation:
    pip install phantom[polars]

Usage:
    from phantom import Session
    from phantom.contrib.polars import polars_ops

    session = Session()
    session.register(polars_ops)  # registers ops AND inspector

    # Now use polars operations
    data = session.ref("read_csv", path="data.csv")
    selected = session.ref("select_columns", df=data, columns=["name", "value"])
    result = session.resolve(selected)
"""

# mypy: ignore-errors
# Rationale: This module uses runtime-imported polars types which mypy cannot
# properly analyze. The dynamic import via require_dependency() is necessary
# for the optional dependency pattern (pip install phantom[polars]).

from __future__ import annotations

from pathlib import Path
from typing import Any

from phantom import OperationSet, Ref
from phantom._security import SecurityPolicy

from ._base import io_policy, require_dependency

pl = require_dependency("polars", "polars", "polars")
DataFrame = pl.DataFrame
LazyFrame = pl.LazyFrame
Series = pl.Series

_IO_OPS = ["read_csv", "read_parquet", "read_json", "read_ndjson"]


def polars_policy(
    allowed_dirs: list[str | Path] | None = None,
    *,
    deny_patterns: list[str] | None = None,
    max_file_bytes: int = 50_000_000,
) -> SecurityPolicy:
    """Create a security policy for polars file operations.

    Args:
        allowed_dirs: Directories the file operations may access.  When
            ``None``, no directory restriction is applied (deny patterns
            and other guards still run).
        deny_patterns: Glob patterns to block, checked against every path
            component (default: ``["*.env", ".git"]``).
        max_file_bytes: Maximum file size for read operations (default: 50 MB).

    Returns:
        A SecurityPolicy with PathGuard and FileSizeGuard bound to I/O ops.
    """
    return io_policy(
        _IO_OPS,
        allowed_dirs=allowed_dirs,
        deny_patterns=deny_patterns,
        max_file_bytes=max_file_bytes,
    )


def _default_polars_policy() -> SecurityPolicy:
    """Build the default security policy for polars operations."""
    return polars_policy()


polars_ops = OperationSet(default_policy=_default_polars_policy())


# =============================================================================
# I/O Operations
# =============================================================================


@polars_ops.op
def read_csv(path: str) -> DataFrame:
    """Read a CSV file into a DataFrame."""
    return pl.read_csv(path)


@polars_ops.op
def read_parquet(path: str) -> DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pl.read_parquet(path)


@polars_ops.op
def read_json(path: str) -> DataFrame:
    """Read a JSON file into a DataFrame."""
    return pl.read_json(path)


@polars_ops.op
def read_ndjson(path: str) -> DataFrame:
    """Read a newline-delimited JSON file into a DataFrame."""
    return pl.read_ndjson(path)


# =============================================================================
# Selection & Filtering
# =============================================================================


@polars_ops.op
def select_columns(df: Ref[DataFrame], columns: list[str]) -> DataFrame:
    """Select specific columns from a DataFrame."""
    return df.select(columns)


@polars_ops.op
def head(df: Ref[DataFrame], n: int = 5) -> DataFrame:
    """Return the first n rows."""
    return df.head(n)


@polars_ops.op
def tail(df: Ref[DataFrame], n: int = 5) -> DataFrame:
    """Return the last n rows."""
    return df.tail(n)


@polars_ops.op
def sample_rows(
    df: Ref[DataFrame], n: int = 5, seed: int | None = None
) -> DataFrame:
    """Return a random sample of rows."""
    return df.sample(n=min(n, len(df)), seed=seed)


@polars_ops.op
def drop_columns(df: Ref[DataFrame], columns: list[str]) -> DataFrame:
    """Drop specified columns from a DataFrame."""
    return df.drop(columns)


@polars_ops.op
def drop_duplicates(
    df: Ref[DataFrame],
    subset: list[str] | None = None,
    keep: str = "first",
) -> DataFrame:
    """Remove duplicate rows. keep: 'first', 'last', or 'none' (drop all)."""
    return df.unique(subset=subset, keep=keep)


# =============================================================================
# Transformation
# =============================================================================


@polars_ops.op
def sort_values(
    df: Ref[DataFrame],
    by: str | list[str],
    descending: bool = False,
) -> DataFrame:
    """Sort by one or more columns."""
    return df.sort(by, descending=descending)


@polars_ops.op
def rename_columns(df: Ref[DataFrame], mapping: dict[str, str]) -> DataFrame:
    """Rename columns using a mapping of {old_name: new_name}."""
    return df.rename(mapping)


@polars_ops.op
def fill_null(
    df: Ref[DataFrame],
    value: Any = None,
    strategy: str | None = None,
    column: str | None = None,
) -> DataFrame:
    """
    Fill null values.

    Use value for a literal fill,
    or strategy ('forward', 'backward', 'mean', 'min', 'max').
    Optionally specify a column to fill only that column.
    """
    if column:
        if strategy:
            return df.with_columns(pl.col(column).fill_null(strategy=strategy))
        return df.with_columns(pl.col(column).fill_null(value=value))
    if strategy:
        return df.fill_null(strategy=strategy)
    return df.fill_null(value=value)


@polars_ops.op
def drop_nulls(
    df: Ref[DataFrame],
    subset: list[str] | None = None,
) -> DataFrame:
    """Remove rows with null values in specified columns (or all columns if None)."""
    return df.drop_nulls(subset=subset)


@polars_ops.op
def cast_column(df: Ref[DataFrame], column: str, dtype: str) -> DataFrame:
    """
    Convert a column to a different type.

    dtype can be: 'Int64', 'Float64', 'Utf8', 'Date', 'Datetime', 'Boolean', etc.
    """
    dtype_map = {
        "int": pl.Int64,
        "int64": pl.Int64,
        "int32": pl.Int32,
        "float": pl.Float64,
        "float64": pl.Float64,
        "float32": pl.Float32,
        "str": pl.Utf8,
        "utf8": pl.Utf8,
        "string": pl.Utf8,
        "bool": pl.Boolean,
        "boolean": pl.Boolean,
        "date": pl.Date,
        "datetime": pl.Datetime,
    }
    pl_dtype = dtype_map.get(dtype.lower())
    if pl_dtype is None:
        raise ValueError(
            f"Unknown dtype: '{dtype}'. "
            f"Supported: {sorted(dtype_map.keys())}"
        )
    return df.with_columns(pl.col(column).cast(pl_dtype))


# =============================================================================
# Aggregation
# =============================================================================


_ALLOWED_AGGS: frozenset[str] = frozenset({
    "sum", "mean", "median", "min", "max",
    "count", "len", "first", "last",
    "std", "var", "n_unique",
})


@polars_ops.op
def groupby_agg(
    df: Ref[DataFrame],
    by: str | list[str],
    aggregations: dict[str, str],
) -> DataFrame:
    """
    Group by columns and aggregate.

    aggregations: {column: 'sum'|'mean'|'count'|'min'|'max'|'first'|'last'|
    'std'|'var'|'median'|'n_unique'|'len'}.
    """
    by_cols = [by] if isinstance(by, str) else by
    agg_exprs = []
    for col_name, agg_func in aggregations.items():
        if agg_func not in _ALLOWED_AGGS:
            raise ValueError(
                f"Unsupported aggregation '{agg_func}'. "
                f"Allowed: {sorted(_ALLOWED_AGGS)}"
            )
        expr = getattr(pl.col(col_name), agg_func)()
        agg_exprs.append(expr)
    return df.group_by(by_cols).agg(agg_exprs)


@polars_ops.op
def value_counts(
    df: Ref[DataFrame],
    column: str,
    normalize: bool = False,
) -> DataFrame:
    """Count unique values in a column. Returns DataFrame with counts."""
    counts = df.select(pl.col(column).value_counts(sort=True))
    result = counts.unnest(column)
    if normalize:
        total = df.height
        result = result.with_columns((pl.col("count") / total).alias("proportion"))
    return result


@polars_ops.op
def describe(df: Ref[DataFrame]) -> DataFrame:
    """Generate descriptive statistics for numeric columns."""
    return df.describe()


# =============================================================================
# Joins
# =============================================================================


@polars_ops.op
def join(
    left: Ref[DataFrame],
    right: Ref[DataFrame],
    on: str | list[str] | None = None,
    left_on: str | list[str] | None = None,
    right_on: str | list[str] | None = None,
    how: str = "inner",
) -> DataFrame:
    """Join two DataFrames. how: 'inner', 'left', 'right', 'outer', 'cross'."""
    return left.join(right, on=on, left_on=left_on, right_on=right_on, how=how)


@polars_ops.op
def concat(
    dfs: list[Ref[DataFrame]],
    how: str = "vertical",
) -> DataFrame:
    """
    Concatenate DataFrames. how: 'vertical' (rows), 'horizontal' (columns), 'diagonal'.
    """
    return pl.concat(dfs, how=how)


# =============================================================================
# Inspector
# =============================================================================


@polars_ops.inspector(DataFrame)
def _inspect_dataframe(df: DataFrame) -> dict[str, Any]:
    """Enhanced DataFrame inspector for LLM context."""
    return {
        "type": "polars.DataFrame",
        "shape": [df.height, df.width],
        "columns": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "null_counts": {col: df[col].null_count() for col in df.columns},
        "sample": df.head(5).to_dicts(),
    }


@polars_ops.inspector(LazyFrame)
def _inspect_lazyframe(lf: LazyFrame) -> dict[str, Any]:
    """LazyFrame inspector for LLM context."""
    schema = lf.collect_schema()
    return {
        "type": "polars.LazyFrame",
        "columns": {col: str(dtype) for col, dtype in schema.items()},
        "note": "LazyFrame (unevaluated). Call .collect() to materialize.",
    }


# =============================================================================
# Public API
# =============================================================================

__all__ = ["polars_ops", "polars_policy"]
