"""OperationSet - Modular operation registration for phantom."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, TypeVar

T = TypeVar("T")


class OperationSet:
    """
    A collection of operations that can be registered to a session.

    Use this to define operations in separate modules and register
    them with a session later. This enables modular code organization
    while maintaining the decorator-based API.

    Example:
        # ops/data.py
        from phantom import OperationSet, Ref
        import pandas as pd

        data_ops = OperationSet()

        @data_ops.op
        def load_csv(path: str) -> pd.DataFrame:
            return pd.read_csv(path)

        @data_ops.op
        def merge(
            left: Ref[pd.DataFrame],
            right: Ref[pd.DataFrame],
            on: str
        ) -> pd.DataFrame:
            return left.merge(right, on=on)

        # main.py
        from phantom import Session
        from ops.data import data_ops

        session = Session()
        session.register(data_ops)

        # Now use the operations
        ref = session.ref("load_csv", path="data.csv")
    """

    def __init__(self) -> None:
        """Create a new empty operation set."""
        self._operations: dict[str, Callable[..., Any]] = {}

    def op(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Register a function in this operation set.

        The decorated function remains callable for direct use and testing.
        The function name becomes the operation name.

        Example:
            ops = OperationSet()

            @ops.op
            def add(x: int, y: int) -> int:
                return x + y

            # Direct call works
            result = add(1, 2)  # 3

            # Register with session later
            session.register(ops)
        """
        self._operations[func.__name__] = func
        return func

    def list_operations(self) -> list[str]:
        """List all operation names in this set."""
        return list(self._operations.keys())

    def __iter__(self) -> Iterator[tuple[str, Callable[..., Any]]]:
        """Iterate over (name, function) pairs."""
        return iter(self._operations.items())

    def __len__(self) -> int:
        """Return the number of operations in this set."""
        return len(self._operations)

    def __contains__(self, name: str) -> bool:
        """Check if an operation name is in this set."""
        return name in self._operations

    def __repr__(self) -> str:
        ops = list(self._operations.keys())
        if len(ops) <= 3:
            return f"OperationSet({ops})"
        return f"OperationSet([{ops[0]!r}, {ops[1]!r}, ... +{len(ops) - 2} more])"
