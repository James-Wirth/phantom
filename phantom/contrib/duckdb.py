"""
DuckDB operations for Phantom.

Provides SQL-based analytics operations for LLM-driven data analysis.

Installation:
    pip install phantom[duckdb]

Usage:
    from phantom import Session
    from phantom.contrib.duckdb import duckdb_ops

    session = Session()
    session.register(duckdb_ops)

    # Query files directly with SQL
    result = session.ref("query", sql="SELECT * FROM 'data.csv' WHERE value > 100")

    # Or create tables and query them
    conn = session.ref("connect")
    session.ref("execute", conn=conn, sql="CREATE TABLE t AS FROM 'data.parquet'")
    result = session.ref("query", conn=conn, sql="SELECT * FROM t WHERE value > 100")
"""

# mypy: ignore-errors
# Rationale: This module uses runtime-imported duckdb types which mypy cannot
# properly analyze. The dynamic import via require_dependency() is necessary
# for the optional dependency pattern (pip install phantom[duckdb]).

from __future__ import annotations

from typing import Any

from phantom import OperationSet, Ref

from ._base import require_dependency

duckdb = require_dependency("duckdb", "duckdb", "duckdb")
DuckDBPyConnection = duckdb.DuckDBPyConnection
DuckDBPyRelation = duckdb.DuckDBPyRelation

duckdb_ops = OperationSet()


# =============================================================================
# Connection Management
# =============================================================================


@duckdb_ops.op
def connect(database: str = ":memory:", read_only: bool = False) -> DuckDBPyConnection:
    """
    Create a DuckDB connection.

    Args:
        database: Path to database file, or ":memory:" for in-memory (default).
        read_only: Open in read-only mode.

    Returns:
        A DuckDB connection object.
    """
    return duckdb.connect(database=database, read_only=read_only)


# =============================================================================
# Query Operations
# =============================================================================


@duckdb_ops.op
def query(
    sql: str,
    conn: Ref[DuckDBPyConnection] | None = None,
) -> DuckDBPyRelation:
    """
    Execute a SQL query and return results as a relation.

    DuckDB can query files directly:
        - CSV: SELECT * FROM 'file.csv'
        - Parquet: SELECT * FROM 'file.parquet'
        - JSON: SELECT * FROM read_json_auto('file.json')
        - Multiple files: SELECT * FROM 'data/*.parquet'

    Args:
        sql: SQL query string.
        conn: Optional connection. Uses a default connection if not provided.

    Returns:
        A DuckDB relation (lazy result set).
    """
    if conn is None:
        return duckdb.query(sql)
    return conn.query(sql)


@duckdb_ops.op
def execute(
    sql: str,
    conn: Ref[DuckDBPyConnection],
) -> None:
    """
    Execute a SQL statement (CREATE, INSERT, UPDATE, DELETE, etc.).

    Use this for statements that don't return results.

    Args:
        sql: SQL statement to execute.
        conn: DuckDB connection.
    """
    conn.execute(sql)


@duckdb_ops.op
def execute_many(
    sql: str,
    parameters: list[tuple[Any, ...]],
    conn: Ref[DuckDBPyConnection],
) -> None:
    """
    Execute a SQL statement multiple times with different parameters.

    Args:
        sql: SQL statement with placeholders (use ? or $1, $2, etc.).
        parameters: List of parameter tuples.
        conn: DuckDB connection.
    """
    conn.executemany(sql, parameters)


# =============================================================================
# Data Import
# =============================================================================


@duckdb_ops.op
def read_csv(
    path: str,
    conn: Ref[DuckDBPyConnection] | None = None,
) -> DuckDBPyRelation:
    """Read a CSV file into a relation."""
    if conn is None:
        return duckdb.read_csv(path)
    return conn.read_csv(path)


@duckdb_ops.op
def read_parquet(
    path: str,
    conn: Ref[DuckDBPyConnection] | None = None,
) -> DuckDBPyRelation:
    """
    Read a Parquet file into a relation.

    Supports glob patterns like 'data/*.parquet'.
    """
    if conn is None:
        return duckdb.read_parquet(path)
    return conn.read_parquet(path)


@duckdb_ops.op
def read_json(
    path: str,
    conn: Ref[DuckDBPyConnection] | None = None,
) -> DuckDBPyRelation:
    """Read a JSON file into a relation."""
    if conn is None:
        return duckdb.read_json(path)
    return conn.read_json(path)


# =============================================================================
# Relation Operations (Lazy/Chainable)
# =============================================================================


@duckdb_ops.op
def filter_relation(
    relation: Ref[DuckDBPyRelation],
    condition: str,
) -> DuckDBPyRelation:
    """
    Filter a relation using a SQL WHERE condition.

    Args:
        relation: Input relation.
        condition: SQL condition, e.g., "value > 100 AND status = 'active'".
    """
    return relation.filter(condition)


@duckdb_ops.op
def select_columns(
    relation: Ref[DuckDBPyRelation],
    columns: str,
) -> DuckDBPyRelation:
    """
    Select columns from a relation.

    Args:
        relation: Input relation.
        columns: Comma-separated column names or expressions,
                 e.g., "name, age" or "name, age * 2 AS double_age".
    """
    return relation.select(columns)


@duckdb_ops.op
def order_by(
    relation: Ref[DuckDBPyRelation],
    columns: str,
) -> DuckDBPyRelation:
    """
    Order a relation by columns.

    Args:
        relation: Input relation.
        columns: Comma-separated columns with optional ASC/DESC,
                 e.g., "name ASC, age DESC".
    """
    return relation.order(columns)


@duckdb_ops.op
def limit_relation(
    relation: Ref[DuckDBPyRelation],
    n: int,
    offset: int = 0,
) -> DuckDBPyRelation:
    """
    Limit the number of rows in a relation.

    Args:
        relation: Input relation.
        n: Maximum number of rows to return.
        offset: Number of rows to skip.
    """
    return relation.limit(n, offset=offset)


@duckdb_ops.op
def distinct(relation: Ref[DuckDBPyRelation]) -> DuckDBPyRelation:
    """Return distinct rows from a relation."""
    return relation.distinct()


@duckdb_ops.op
def aggregate(
    relation: Ref[DuckDBPyRelation],
    aggregations: str,
    group_by: str | None = None,
) -> DuckDBPyRelation:
    """
    Aggregate a relation.

    Args:
        relation: Input relation.
        aggregations: Comma-separated aggregations,
                      e.g., "SUM(value), AVG(price), COUNT(*)".
        group_by: Optional comma-separated columns to group by.
    """
    if group_by:
        return relation.aggregate(aggregations, group_by)
    return relation.aggregate(aggregations)


@duckdb_ops.op
def join_relations(
    left: Ref[DuckDBPyRelation],
    right: Ref[DuckDBPyRelation],
    condition: str,
    how: str = "inner",
) -> DuckDBPyRelation:
    """
    Join two relations.

    Args:
        left: Left relation.
        right: Right relation.
        condition: Join condition, e.g., "left.id = right.id".
        how: Join type - 'inner', 'left', 'right', 'outer'.
    """
    return left.join(right, condition, how=how)


@duckdb_ops.op
def union(
    relation1: Ref[DuckDBPyRelation],
    relation2: Ref[DuckDBPyRelation],
) -> DuckDBPyRelation:
    """Combine two relations (removes duplicates)."""
    return relation1.union(relation2)


@duckdb_ops.op
def union_all(
    relation1: Ref[DuckDBPyRelation],
    relation2: Ref[DuckDBPyRelation],
) -> DuckDBPyRelation:
    """Combine two relations (keeps duplicates)."""
    return relation1.union(relation2)


# =============================================================================
# Materialization
# =============================================================================


@duckdb_ops.op
def to_df(relation: Ref[DuckDBPyRelation]) -> Any:
    """Convert a relation to a pandas DataFrame."""
    return relation.df()


@duckdb_ops.op
def to_arrow(relation: Ref[DuckDBPyRelation]) -> Any:
    """Convert a relation to a PyArrow Table."""
    return relation.arrow()


@duckdb_ops.op
def to_polars(relation: Ref[DuckDBPyRelation]) -> Any:
    """Convert a relation to a Polars DataFrame."""
    return relation.pl()


@duckdb_ops.op
def fetchall(relation: Ref[DuckDBPyRelation]) -> list[tuple[Any, ...]]:
    """Fetch all rows as a list of tuples."""
    return relation.fetchall()


@duckdb_ops.op
def fetchone(relation: Ref[DuckDBPyRelation]) -> tuple[Any, ...] | None:
    """Fetch a single row as a tuple."""
    return relation.fetchone()


# =============================================================================
# Table Management
# =============================================================================


@duckdb_ops.op
def create_table_from_relation(
    conn: Ref[DuckDBPyConnection],
    table_name: str,
    relation: Ref[DuckDBPyRelation],
) -> None:
    """Create a table from a relation."""
    conn.execute(
        f"CREATE TABLE {table_name} AS SELECT * FROM relation",
        {"relation": relation},
    )


@duckdb_ops.op
def create_table_from_df(
    conn: Ref[DuckDBPyConnection],
    table_name: str,
    df: Any,  # noqa: ARG001 - DuckDB resolves 'df' by name from locals
) -> None:
    """Create a table from a pandas or polars DataFrame."""
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")


@duckdb_ops.op
def insert_into(
    conn: Ref[DuckDBPyConnection],
    table_name: str,
    relation: Ref[DuckDBPyRelation],
) -> None:
    """Insert rows from a relation into a table."""
    conn.execute(
        f"INSERT INTO {table_name} SELECT * FROM relation",
        {"relation": relation},
    )


@duckdb_ops.op
def list_tables(conn: Ref[DuckDBPyConnection]) -> list[str]:
    """List all tables in the database."""
    result = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main'"
    )
    return [row[0] for row in result.fetchall()]


@duckdb_ops.op
def describe_table(
    conn: Ref[DuckDBPyConnection],
    table_name: str,
) -> DuckDBPyRelation:
    """Get schema information for a table."""
    return conn.query(f"DESCRIBE {table_name}")


@duckdb_ops.op
def drop_table(
    conn: Ref[DuckDBPyConnection],
    table_name: str,
    if_exists: bool = True,
) -> None:
    """Drop a table from the database."""
    exists_clause = "IF EXISTS " if if_exists else ""
    conn.execute(f"DROP TABLE {exists_clause}{table_name}")


# =============================================================================
# Inspector
# =============================================================================


@duckdb_ops.inspector(DuckDBPyRelation)
def _inspect_relation(relation: DuckDBPyRelation) -> dict[str, Any]:
    """Relation inspector for LLM context."""
    columns = relation.columns
    types = relation.types

    sample = relation.limit(5).fetchall()

    return {
        "type": "duckdb.Relation",
        "columns": {col: str(dtype) for col, dtype in zip(columns, types)},
        "sample": [dict(zip(columns, row)) for row in sample],
        "note": "Lazy relation. Use to_df(), fetchall(), etc. to materialize.",
    }


@duckdb_ops.inspector(DuckDBPyConnection)
def _inspect_connection(conn: DuckDBPyConnection) -> dict[str, Any]:
    """Connection inspector for LLM context."""
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    return {
        "type": "duckdb.Connection",
        "tables": [t[0] for t in tables],
    }


# =============================================================================
# Public API
# =============================================================================

__all__ = ["duckdb_ops"]
