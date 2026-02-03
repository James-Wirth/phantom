"""
DuckDB operations for Phantom.

Provides SQL-based analytics operations for LLM-driven data analysis.

Security
--------
Connections created via ``connect()`` are hardened by default:

- ``enable_external_access=false`` — blocks all filesystem and network I/O
  from within SQL (``COPY``, ``read_csv_auto()``, ``FROM 'file.csv'``, etc.).
- ``allowed_directories`` / ``allowed_paths`` — opt-in allowlists forwarded
  from ``duckdb_policy()`` to restore access to specific locations.
- Extension auto-install/load disabled.
- ``lock_configuration=true`` — prevents SQL from reversing these settings.

Installation:
    pip install phantom[duckdb]

Usage:
    from phantom import Session
    from phantom.contrib.duckdb import duckdb_ops

    session = Session()
    session.register(duckdb_ops)

    # Query files directly with SQL (only if allowed_dirs permits)
    result = session.ref("query", sql="SELECT * FROM 'data.csv' WHERE value > 100")

    # Or create tables and query them
    conn = session.ref("connect")
    result = session.ref("query", conn=conn, sql="SELECT * FROM t WHERE value > 100")
"""

# mypy: ignore-errors
# Rationale: This module uses runtime-imported duckdb types which mypy cannot
# properly analyze. The dynamic import via require_dependency() is necessary
# for the optional dependency pattern (pip install phantom[duckdb]).

from __future__ import annotations

from pathlib import Path
from typing import Any

from phantom import OperationSet, Ref
from phantom._security import SecurityPolicy

from ._base import io_policy, require_dependency

duckdb = require_dependency("duckdb", "duckdb", "duckdb")
DuckDBPyConnection = duckdb.DuckDBPyConnection
DuckDBPyRelation = duckdb.DuckDBPyRelation

_IO_OPS = ["read_csv", "read_parquet", "read_json"]


def _quote_identifier(name: str) -> str:
    """Quote a SQL identifier to prevent injection.

    Uses double-quoting with proper escaping of embedded double-quotes.
    Rejects names containing null bytes.
    """
    if "\x00" in name:
        raise ValueError(f"Identifier contains null byte: {name!r}")
    return '"' + name.replace('"', '""') + '"'


# =============================================================================
# Secure connection helper
# =============================================================================

_SECURE_CONFIG: dict[str, str] = {
    "enable_external_access": "false",
    "autoinstall_known_extensions": "false",
    "autoload_known_extensions": "false",
}


def _secure_connect(
    database: str = ":memory:",
    *,
    read_only: bool = False,
    allowed_dirs: list[str | Path] | None = None,
    allowed_paths: list[str | Path] | None = None,
) -> DuckDBPyConnection:
    """Create a DuckDB connection with security settings applied.

    Disables external access by default, then optionally re-allows specific
    directories/paths via DuckDB's native allowlists.  Finishes by locking
    the configuration so injected SQL cannot reverse these settings.

    Note: ``allowed_directories`` and ``allowed_paths`` are ``VARCHAR[]``
    settings and must be applied via ``SET`` statements — passing them
    through the ``config`` dict causes a crash (duckdb/duckdb#17128).
    """
    conn = duckdb.connect(
        database=database, read_only=read_only, config=dict(_SECURE_CONFIG)
    )
    if allowed_dirs:
        dirs = [str(Path(d).resolve()) for d in allowed_dirs]
        conn.execute("SET allowed_directories = $dirs", {"dirs": dirs})
    if allowed_paths:
        paths = [str(Path(p).resolve()) for p in allowed_paths]
        conn.execute("SET allowed_paths = $paths", {"paths": paths})
    conn.execute("SET lock_configuration = true")
    return conn


_default_conn: DuckDBPyConnection | None = None


def _get_default_conn() -> DuckDBPyConnection:
    """Return (or lazily create) a locked-down default in-memory connection."""
    global _default_conn
    if _default_conn is None:
        _default_conn = _secure_connect()
    return _default_conn


# =============================================================================
# Policy
# =============================================================================


def duckdb_policy(
    allowed_dirs: list[str | Path] | None = None,
    *,
    allowed_paths: list[str | Path] | None = None,
    deny_patterns: list[str] | None = None,
    max_file_bytes: int = 50_000_000,
) -> SecurityPolicy:
    """Create a security policy for DuckDB operations.

    SQL safety is enforced by DuckDB's native controls
    (``enable_external_access=false``, ``lock_configuration=true``,
    extension auto-load disabled) which are applied in ``_secure_connect``.
    This policy adds PathGuard and FileSizeGuard for file I/O operations.

    Args:
        allowed_dirs: Directories the file operations may access.  When
            ``None``, no directory restriction is applied (deny patterns
            and other guards still run).  Also forwarded to DuckDB's
            ``allowed_directories`` setting on new connections.
        allowed_paths: Specific file paths to allow (forwarded to DuckDB's
            ``allowed_paths`` setting on new connections).
        deny_patterns: Glob patterns to block, checked against every path
            component (default: ``DEFAULT_DENY_PATTERNS``).
        max_file_bytes: Maximum file size for read operations (default: 50 MB).

    Returns:
        A SecurityPolicy with PathGuard and FileSizeGuard bound to I/O ops.
    """
    return io_policy(
        _IO_OPS,
        allowed_dirs=allowed_dirs,
        deny_patterns=deny_patterns,
        max_file_bytes=max_file_bytes,
        extra_path_bindings=[
            (["connect"], ["database"]),
        ],
    )


def _default_duckdb_policy() -> SecurityPolicy:
    """Build the default security policy for DuckDB operations."""
    return duckdb_policy()


duckdb_ops = OperationSet(default_policy=_default_duckdb_policy())


# =============================================================================
# Connection Management
# =============================================================================


@duckdb_ops.op
def connect(
    database: str = ":memory:",
    read_only: bool = False,
) -> DuckDBPyConnection:
    """
    Create a DuckDB connection.

    The connection is hardened: external filesystem/network access is disabled
    by default and the configuration is locked.

    Args:
        database: Path to database file, or ":memory:" for in-memory (default).
        read_only: Open in read-only mode.

    Returns:
        A DuckDB connection object.
    """
    return _secure_connect(database=database, read_only=read_only)


# =============================================================================
# Query Operations
# =============================================================================


@duckdb_ops.op
def query(
    sql: str,
    conn: Ref[DuckDBPyConnection] | None = None,
) -> DuckDBPyRelation:
    """
    Execute a read-only SQL query and return results as a relation.

    Args:
        sql: SQL query string (SELECT / WITH / EXPLAIN only).
        conn: Optional connection. Uses a locked-down default if not provided.

    Returns:
        A DuckDB relation (lazy result set).
    """
    target = conn if conn is not None else _get_default_conn()
    return target.query(sql)


@duckdb_ops.op
def execute(
    sql: str,
    conn: Ref[DuckDBPyConnection],
) -> None:
    """
    Execute a SQL statement (CREATE TABLE, etc.).

    Use this for statements that don't return results.  Dangerous
    statements (COPY, INSTALL, ATTACH, …) are blocked by DuckDB's
    native ``enable_external_access`` and ``lock_configuration`` settings.

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
    target = conn if conn is not None else _get_default_conn()
    return target.read_csv(path)


@duckdb_ops.op
def read_parquet(
    path: str,
    conn: Ref[DuckDBPyConnection] | None = None,
) -> DuckDBPyRelation:
    """
    Read a Parquet file into a relation.

    Supports glob patterns like 'data/*.parquet'.
    """
    target = conn if conn is not None else _get_default_conn()
    return target.read_parquet(path)


@duckdb_ops.op
def read_json(
    path: str,
    conn: Ref[DuckDBPyConnection] | None = None,
) -> DuckDBPyRelation:
    """Read a JSON file into a relation."""
    target = conn if conn is not None else _get_default_conn()
    return target.read_json(path)


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
    quoted = _quote_identifier(table_name)
    conn.execute(
        f"CREATE TABLE {quoted} AS SELECT * FROM relation",
        {"relation": relation},
    )


@duckdb_ops.op
def create_table_from_df(
    conn: Ref[DuckDBPyConnection],
    table_name: str,
    df: Any,  # noqa: ARG001 - DuckDB resolves 'df' by name from locals
) -> None:
    """Create a table from a pandas or polars DataFrame."""
    quoted = _quote_identifier(table_name)
    conn.execute(f"CREATE TABLE {quoted} AS SELECT * FROM df")


@duckdb_ops.op
def insert_into(
    conn: Ref[DuckDBPyConnection],
    table_name: str,
    relation: Ref[DuckDBPyRelation],
) -> None:
    """Insert rows from a relation into a table."""
    quoted = _quote_identifier(table_name)
    conn.execute(
        f"INSERT INTO {quoted} SELECT * FROM relation",
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
    quoted = _quote_identifier(table_name)
    return conn.query(f"DESCRIBE {quoted}")


@duckdb_ops.op
def drop_table(
    conn: Ref[DuckDBPyConnection],
    table_name: str,
    if_exists: bool = True,
) -> None:
    """Drop a table from the database."""
    quoted = _quote_identifier(table_name)
    exists_clause = "IF EXISTS " if if_exists else ""
    conn.execute(f"DROP TABLE {exists_clause}{quoted}")


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

__all__ = ["duckdb_ops", "duckdb_policy"]
