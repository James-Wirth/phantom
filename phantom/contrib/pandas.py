"""
Pandas operations for Phantom.

Provides DataFrame operations for LLM-driven data analysis.

Installation:
    pip install phantom[pandas]

Usage:
    from phantom import Session
    from phantom.contrib.pandas import pandas_ops

    session = Session()
    session.register(pandas_ops)  # registers ops AND inspector

    # Now use pandas operations
    data = session.ref("read_csv", path="data.csv")
    filtered = session.ref("filter_rows", df=data, condition="value > 100")
    result = session.resolve(filtered)
"""

# mypy: ignore-errors
# Rationale: This module uses runtime-imported pandas types which mypy cannot
# properly analyze. The dynamic import via require_dependency() is necessary
# for the optional dependency pattern (pip install phantom[pandas]).

from __future__ import annotations

from pathlib import Path
from typing import Any

from phantom import OperationSet, Ref
from phantom._security import (
    DEFAULT_DENY_PATTERNS,
    FileSizeGuard,
    PathGuard,
    SecurityPolicy,
)

from ._base import require_dependency

pd = require_dependency("pandas", "pandas", "pandas")
DataFrame = pd.DataFrame
Series = pd.Series

_IO_OPS = ["read_csv", "read_parquet", "read_json", "read_excel"]


def pandas_policy(
    allowed_dirs: list[str | Path] | None = None,
    *,
    deny_patterns: list[str] | None = None,
    max_file_bytes: int = 50_000_000,
) -> SecurityPolicy:
    """Create a security policy for pandas file operations.

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
    if deny_patterns is None:
        deny_patterns = list(DEFAULT_DENY_PATTERNS)
    policy = SecurityPolicy()
    policy.bind(
        PathGuard(allowed_dirs, deny_patterns=deny_patterns),
        ops=_IO_OPS,
        args=["path"],
    )
    policy.bind(
        FileSizeGuard(max_bytes=max_file_bytes),
        ops=_IO_OPS,
        args=["path"],
    )
    return policy


def _default_pandas_policy() -> SecurityPolicy:
    """Build the default security policy for pandas operations."""
    return pandas_policy()


pandas_ops = OperationSet(default_policy=_default_pandas_policy())


# =============================================================================
# I/O Operations
# =============================================================================


@pandas_ops.op
def read_csv(path: str) -> DataFrame:
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(path)


@pandas_ops.op
def read_parquet(path: str) -> DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(path)


@pandas_ops.op
def read_json(path: str) -> DataFrame:
    """Read a JSON file into a DataFrame."""
    return pd.read_json(path)


@pandas_ops.op
def read_excel(path: str, sheet_name: str | int = 0) -> DataFrame:
    """Read an Excel file into a DataFrame."""
    return pd.read_excel(path, sheet_name=sheet_name)


# =============================================================================
# Selection & Filtering
# =============================================================================


@pandas_ops.op
def select_columns(df: Ref[DataFrame], columns: list[str]) -> DataFrame:
    """Select specific columns from a DataFrame."""
    return df[columns]


@pandas_ops.op
def head(df: Ref[DataFrame], n: int = 5) -> DataFrame:
    """Return the first n rows."""
    return df.head(n)


@pandas_ops.op
def tail(df: Ref[DataFrame], n: int = 5) -> DataFrame:
    """Return the last n rows."""
    return df.tail(n)


@pandas_ops.op
def sample_rows(
    df: Ref[DataFrame], n: int = 5, random_state: int | None = None
) -> DataFrame:
    """Return a random sample of rows."""
    return df.sample(n=min(n, len(df)), random_state=random_state)


@pandas_ops.op
def drop_columns(df: Ref[DataFrame], columns: list[str]) -> DataFrame:
    """Drop specified columns from a DataFrame."""
    return df.drop(columns=columns)


@pandas_ops.op
def drop_duplicates(
    df: Ref[DataFrame],
    subset: list[str] | None = None,
    keep: str = "first",
) -> DataFrame:
    """Remove duplicate rows. keep: 'first', 'last', or False (drop all)."""
    return df.drop_duplicates(subset=subset, keep=keep)


# =============================================================================
# Transformation
# =============================================================================


@pandas_ops.op
def sort_values(
    df: Ref[DataFrame],
    by: str | list[str],
    ascending: bool = True,
) -> DataFrame:
    """Sort by one or more columns."""
    return df.sort_values(by=by, ascending=ascending)


@pandas_ops.op
def rename_columns(df: Ref[DataFrame], mapping: dict[str, str]) -> DataFrame:
    """Rename columns using a mapping of {old_name: new_name}."""
    return df.rename(columns=mapping)


@pandas_ops.op
def fillna(
    df: Ref[DataFrame],
    value: Any = None,
    method: str | None = None,
    column: str | None = None,
) -> DataFrame:
    """Fill missing values. Use value or method ('ffill', 'bfill')."""
    if column:
        result = df.copy()
        if method:
            result[column] = result[column].fillna(method=method)
        else:
            result[column] = result[column].fillna(value=value)
        return result
    if method:
        return df.fillna(method=method)
    return df.fillna(value=value)


@pandas_ops.op
def dropna(
    df: Ref[DataFrame],
    subset: list[str] | None = None,
    how: str = "any",
) -> DataFrame:
    """Remove rows with missing values. how: 'any' or 'all'."""
    return df.dropna(subset=subset, how=how)


@pandas_ops.op
def astype(df: Ref[DataFrame], column: str, dtype: str) -> DataFrame:
    """
    Convert a column to a different type
    ('int', 'float', 'str', 'datetime64', 'category').
    """
    result = df.copy()
    result[column] = result[column].astype(dtype)
    return result


# =============================================================================
# Aggregation
# =============================================================================


@pandas_ops.op
def groupby_agg(
    df: Ref[DataFrame],
    by: str | list[str],
    aggregations: dict[str, str],
) -> DataFrame:
    """
    Group by columns and aggregate. aggregations:
    {column: 'sum'|'mean'|'count'|'min'|'max'}.
    """
    return df.groupby(by, as_index=False).agg(aggregations)


@pandas_ops.op
def value_counts(
    df: Ref[DataFrame],
    column: str,
    normalize: bool = False,
) -> DataFrame:
    """Count unique values in a column. Returns DataFrame with counts."""
    counts = df[column].value_counts(normalize=normalize)
    return counts.reset_index()


@pandas_ops.op
def describe(df: Ref[DataFrame]) -> DataFrame:
    """Generate descriptive statistics for numeric columns."""
    return df.describe().T.reset_index().rename(columns={"index": "column"})


# =============================================================================
# Joins
# =============================================================================


@pandas_ops.op
def merge(
    left: Ref[DataFrame],
    right: Ref[DataFrame],
    on: str | list[str] | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    how: str = "inner",
) -> DataFrame:
    """Merge two DataFrames. how: 'inner', 'left', 'right', 'outer'."""
    return left.merge(right, on=on, left_on=left_on, right_on=right_on, how=how)


@pandas_ops.op
def concat(
    dfs: list[Ref[DataFrame]],
    axis: int = 0,
    ignore_index: bool = True,
) -> DataFrame:
    """Concatenate DataFrames. axis: 0 (rows) or 1 (columns)."""
    return pd.concat(dfs, axis=axis, ignore_index=ignore_index)


# =============================================================================
# Inspector
# =============================================================================


@pandas_ops.inspector(DataFrame)
def _inspect_dataframe(df: DataFrame) -> dict[str, Any]:
    """Enhanced DataFrame inspector for LLM context."""
    return {
        "type": "dataframe",
        "shape": list(df.shape),
        "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "sample": df.head(5).to_dict("records"),
    }


# =============================================================================
# Public API
# =============================================================================

__all__ = ["pandas_ops", "pandas_policy"]
