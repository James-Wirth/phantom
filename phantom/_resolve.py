"""Resolve - DAG resolution engine."""

from __future__ import annotations

import asyncio
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


def _group_by_level(order: list[Ref[Any]]) -> list[list[Ref[Any]]]:
    """
    Group refs by execution level for parallel scheduling.

    Level 0: nodes with no dependencies (leaves)
    Level N: nodes whose deps are all in levels < N

    Args:
        order: Refs in topological order (from _topological_order)

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

    # Group by level
    max_level = max(levels.values())
    result: list[list[Ref[Any]]] = [[] for _ in range(max_level + 1)]
    for node in order:
        result[levels[node.id]].append(node)

    return result


async def _execute_one(
    node: Ref[Any],
    cache: dict[str, Any],
    order: list[Ref[Any]],
) -> Any:
    """
    Execute a single operation, handling both sync and async.

    Args:
        node: The ref to execute
        cache: Cache of already-resolved values
        order: Full topological order (for error chain computation)

    Returns:
        The result of the operation

    Raises:
        ResolutionError: If the operation fails
    """
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
        if asyncio.iscoroutinefunction(op_func):
            result = await op_func(**resolved_args)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            func_to_run = op_func
            args_to_use = resolved_args
            result = await loop.run_in_executor(
                None, lambda: func_to_run(**args_to_use)
            )
        return result
    except ResolutionError:
        raise
    except Exception as e:
        chain = [r.id for r in order[: order.index(node) + 1]]
        raise ResolutionError(
            f"Operation '{node.op}' failed: {e}",
            node,
            chain,
            e,
        ) from e


async def aresolve(ref: Ref[Any] | str, *, parallel: bool = True) -> Any:
    """
    Resolve a ref asynchronously with optional parallel execution.

    When parallel=True, independent branches of the DAG execute concurrently
    using asyncio.gather(). This can significantly speed up I/O-bound workflows.

    Args:
        ref: The ref to resolve (or ref ID string)
        parallel: If True, execute independent branches concurrently

    Returns:
        The concrete value produced by the operation

    Raises:
        ResolutionError: If an operation fails, with full lineage context
        CycleError: If a cycle is detected in the computation graph

    Example:
        @phantom.op
        async def fetch_data(url: str) -> dict:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    return await resp.json()

        ref = phantom.ref("fetch_data", url="https://api.example.com/data")
        result = await phantom.aresolve(ref)
    """
    if isinstance(ref, str):
        ref = get_ref(ref)

    order = _topological_order(ref)
    cache: dict[str, Any] = {}

    if parallel:
        levels = _group_by_level(order)
        for level in levels:
            if len(level) == 1:
                cache[level[0].id] = await _execute_one(level[0], cache, order)
            else:
                # Execute independent ops concurrently
                results = await asyncio.gather(*[
                    _execute_one(node, cache, order) for node in level
                ])
                for node, result in zip(level, results):
                    cache[node.id] = result
    else:
        for node in order:
            cache[node.id] = await _execute_one(node, cache, order)

    return cache[ref.id]
