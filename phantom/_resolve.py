"""Resolve - DAG resolution engine."""

from __future__ import annotations

from typing import Any

from ._errors import CycleError, ResolutionError
from ._ref import Ref
from ._registry import get_operation, get_ref


def _topological_order(root: Ref[Any]) -> list[Ref[Any]]:
    """
    Return refs in dependency order (leaves first) with cycle detection.

    Uses depth-first search to produce a topological ordering suitable
    for iterative resolution.

    Args:
        root: The root ref to start from

    Returns:
        List of refs in execution order (dependencies before dependents)

    Raises:
        CycleError: If a cycle is detected in the graph
    """
    visited: set[str] = set()
    in_progress: set[str] = set()
    order: list[Ref[Any]] = []
    chain: list[str] = []

    def visit(ref: Ref[Any]) -> None:
        if ref.id in visited:
            return
        if ref.id in in_progress:
            raise CycleError(ref.id, chain.copy())

        in_progress.add(ref.id)
        chain.append(ref.id)

        for parent in ref.parents:
            visit(parent)

        chain.pop()
        in_progress.remove(ref.id)
        visited.add(ref.id)
        order.append(ref)

    visit(root)
    return order


def resolve(ref: Ref[Any] | str) -> Any:
    """
    Resolve a ref to its concrete value.

    Walks the computation DAG in topological order, resolving
    dependencies first, then executes operations iteratively.

    Args:
        ref: The ref to resolve (or ref ID string)

    Returns:
        The concrete value produced by the operation

    Raises:
        ResolutionError: If an operation fails, with full lineage context
        CycleError: If a cycle is detected in the computation graph

    Example:
        @phantom.op
        def double(x: int) -> int:
            return x * 2

        ref = phantom.ref("double", x=5)
        result = phantom.resolve(ref)  # Returns 10
    """
    if isinstance(ref, str):
        ref = get_ref(ref)

    order = _topological_order(ref)
    cache: dict[str, Any] = {}

    for node in order:
        resolved_args: dict[str, Any] = {}
        for key, value in node.args.items():
            if isinstance(value, Ref):
                resolved_args[key] = cache[value.id]
            else:
                resolved_args[key] = value

        try:
            op_func = get_operation(node.op)
        except KeyError as e:
            chain = [r.id for r in order[: order.index(node) + 1]]
            raise ResolutionError(
                f"Unknown operation: {node.op}",
                node,
                chain,
                e,
            ) from e

        try:
            result = op_func(**resolved_args)
        except Exception as e:
            chain = [r.id for r in order[: order.index(node) + 1]]
            raise ResolutionError(
                f"Operation '{node.op}' failed: {e}",
                node,
                chain,
                e,
            ) from e

        cache[node.id] = result

    return cache[ref.id]
