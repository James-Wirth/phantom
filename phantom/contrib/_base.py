"""Shared utilities for contrib modules."""

from __future__ import annotations

import importlib
from typing import Any


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
