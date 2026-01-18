"""Resolve - DAG resolution engine."""

from __future__ import annotations

from typing import Any

from ._errors import CycleError, ResolutionError
from ._ref import Ref
from ._registry import get_operation, get_ref


def resolve(
    ref: Ref[Any] | str,
    *,
    _cache: dict[str, Any] | None = None,
    _chain: list[str] | None = None,
) -> Any:
    """
    Resolve a ref to its concrete value.

    Walks the computation DAG, resolving dependencies first,
    then executes the operation with resolved arguments.

    Args:
        ref: The ref to resolve (or ref ID string)
        _cache: Internal cache for resolved values (avoid re-computation)
        _chain: Internal chain of ref IDs for error context

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
    if _cache is None:
        _cache = {}
    if _chain is None:
        _chain = []

    if isinstance(ref, str):
        ref = get_ref(ref)

    # Cycle detection
    if ref.id in _chain:
        raise CycleError(ref.id, _chain)

    current_chain = _chain + [ref.id]

    if ref.id in _cache:
        return _cache[ref.id]

    # Get operation with error context
    try:
        op_func = get_operation(ref.op)
    except KeyError as e:
        raise ResolutionError(
            f"Unknown operation: {ref.op}",
            ref,
            current_chain,
            e,
        ) from e

    # Resolve arguments
    resolved_args: dict[str, Any] = {}
    for key, value in ref.args.items():
        if isinstance(value, Ref):
            resolved_args[key] = resolve(value, _cache=_cache, _chain=current_chain)
        else:
            resolved_args[key] = value

    # Execute operation with error context
    try:
        result = op_func(**resolved_args)
    except Exception as e:
        raise ResolutionError(
            f"Operation '{ref.op}' failed: {e}",
            ref,
            current_chain,
            e,
        ) from e

    _cache[ref.id] = result
    return result
