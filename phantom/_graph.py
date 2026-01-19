"""Graph - Internal DAG utilities for traversal and scheduling."""

from __future__ import annotations

from typing import Any

from ._errors import CycleError
from ._ref import Ref


def topological_order(root: Ref[Any]) -> list[Ref[Any]]:
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


def group_by_level(order: list[Ref[Any]]) -> list[list[Ref[Any]]]:
    """
    Group refs by execution level for parallel scheduling.

    Level 0: nodes with no dependencies (leaves)
    Level N: nodes whose deps are all in levels < N

    Args:
        order: Refs in topological order (from topological_order)

    Returns:
        List of levels, where each level contains refs that can run in parallel
    """
    if not order:
        return []

    levels: dict[str, int] = {}

    for node in order:
        if not node.parents:
            levels[node.id] = 0
        else:
            max_parent_level = max(levels[p.id] for p in node.parents)
            levels[node.id] = max_parent_level + 1

    max_level = max(levels.values())
    result: list[list[Ref[Any]]] = [[] for _ in range(max_level + 1)]
    for node in order:
        result[levels[node.id]].append(node)

    return result
