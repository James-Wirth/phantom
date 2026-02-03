"""Shared utilities for contrib modules."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from phantom._security import (
    DEFAULT_DENY_PATTERNS,
    FileSizeGuard,
    PathGuard,
    SecurityPolicy,
)


class MissingDependencyError(ImportError):
    """
    Raised when a contrib module's required dependency is not installed.

    Provides a clear error message with installation instructions.
    """

    def __init__(self, module: str, package: str, extra: str) -> None:
        self.module = module
        self.package = package
        self.extra = extra
        super().__init__(
            f"phantom.contrib.{module} requires '{package}'. "
            f"Install with: pip install phantom[{extra}]"
        )


def require_dependency(module: str, package: str, extra: str) -> Any:
    """
    Import a required dependency or raise a clear error.

    Args:
        module: The contrib module name (e.g., "pandas")
        package: The package to import (e.g., "pandas")
        extra: The pip extra name (e.g., "pandas")

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: If the dependency is not installed.

    Example:
        pd = require_dependency("pandas", "pandas", "pandas")
    """
    try:
        return importlib.import_module(package)
    except ImportError as e:
        raise MissingDependencyError(module, package, extra) from e


def io_policy(
    io_ops: list[str],
    *,
    allowed_dirs: list[str | Path] | None = None,
    deny_patterns: list[str] | None = None,
    max_file_bytes: int = 50_000_000,
    extra_path_bindings: list[tuple[list[str], list[str]]] | None = None,
) -> SecurityPolicy:
    """Create a security policy for file I/O operations.

    Shared factory used by contrib modules that perform file I/O.  Binds
    ``PathGuard`` and ``FileSizeGuard`` to the given *io_ops* on the
    ``"path"`` argument.

    Args:
        io_ops: Operation names that perform file I/O.
        allowed_dirs: Directories the operations may access.  ``None`` means
            no directory restriction (deny patterns still apply).
        deny_patterns: Glob patterns to block, checked against every path
            component.  Defaults to ``DEFAULT_DENY_PATTERNS``.
        max_file_bytes: Maximum file size in bytes (default 50 MB).
        extra_path_bindings: Additional ``(ops, args)`` pairs that receive
            the same ``PathGuard`` (but not ``FileSizeGuard``).
            Example: ``[(["connect"], ["database"])]`` for DuckDB.

    Returns:
        A :class:`~phantom.SecurityPolicy` ready for use.
    """
    if deny_patterns is None:
        deny_patterns = list(DEFAULT_DENY_PATTERNS)

    path_guard = PathGuard(allowed_dirs, deny_patterns=deny_patterns)
    policy = SecurityPolicy()
    policy.bind(path_guard, ops=io_ops, args=["path"])

    if extra_path_bindings:
        for extra_ops, extra_args in extra_path_bindings:
            policy.bind(path_guard, ops=extra_ops, args=extra_args)

    policy.bind(
        FileSizeGuard(max_bytes=max_file_bytes),
        ops=io_ops,
        args=["path"],
    )
    return policy
