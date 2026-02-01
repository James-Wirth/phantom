"""
Pre-built operation sets for common libraries.

Each contrib module requires its respective library to be installed.
Install with extras: pip install phantom[pandas]

Available modules:
    - phantom.contrib.pandas: DataFrame operations (requires pandas)
    - phantom.contrib.polars: DataFrame operations (requires polars)
    - phantom.contrib.duckdb: SQL analytics (requires duckdb)
    - phantom.contrib.http: HTTP / API orchestration (requires httpx)

Example:
    from phantom import Session
    from phantom.contrib.pandas import pandas_ops

    session = Session()
    session.register(pandas_ops)  # registers ops AND inspector
"""

from ._base import MissingDependencyError

__all__ = ["MissingDependencyError", "available_modules"]


def available_modules() -> dict[str, bool]:
    """
    Check which contrib modules have dependencies installed.

    Returns:
        Dict mapping module names to availability status.

    Example:
        >>> from phantom.contrib import available_modules
        >>> available_modules()
        {'pandas': True, 'polars': False, 'files': True}
    """
    result = {}
    for name, pkg in [
        ("pandas", "pandas"),
        ("polars", "polars"),
        ("duckdb", "duckdb"),
        ("http", "httpx"),
    ]:
        try:
            __import__(pkg)
            result[name] = True
        except ImportError:
            result[name] = False
    return result
